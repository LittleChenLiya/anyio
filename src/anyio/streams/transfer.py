from __future__ import annotations

__all__ = ("transfer_stream",)

import hashlib
import hmac
from contextlib import suppress
from typing import TYPE_CHECKING

from .._core._tasks import CancelScope

if TYPE_CHECKING:
    from ..abc import AsyncResource, ByteReceiveStream, TransactionalByteSink


async def _aclose_shielded(resource: AsyncResource) -> None:
    # Close the resource without letting an outer cancellation interrupt it.
    with CancelScope(shield=True):
        await resource.aclose()


async def transfer_stream(
    source: ByteReceiveStream,
    sink: TransactionalByteSink,
    *,
    expected_digest: bytes | str,
    max_bytes: int,
    close_source: bool = False,
) -> None:
    """
    Stream bytes from a single source to a transactional sink.

    The source is read chunk by chunk, and each chunk is written to the sink as it is
    received -- the entire content is never buffered in memory. While reading, the
    total number of bytes is counted and a SHA-256 digest is computed on the fly.

    The sink is only committed when the source reaches EOF, the total number of bytes
    does not exceed ``max_bytes`` and the computed digest matches ``expected_digest``.
    On any failure before commit (including cancellation), the sink is aborted first
    and the original error is then re-raised. A failure raised by the sink's own
    ``abort()`` is suppressed so that it cannot mask the original error.

    If the sink's ``commit()`` fails, the error is propagated as-is. The transaction
    state is owned by the sink in that case, so the utility neither aborts nor claims
    success.

    No background tasks are created; the entire transfer runs in the calling task.

    :param source: the byte stream to read from
    :param sink: the transactional sink to write to
    :param expected_digest: the expected SHA-256 digest, either as raw bytes or as a
        hexadecimal string
    :param max_bytes: the maximum number of bytes to accept (must be a positive
        integer). If the source produces more bytes than this, the transfer is aborted
        and :class:`ValueError` is raised
    :param close_source: if ``True``, the source stream is closed after the transfer
        (at most once), regardless of whether it succeeded or failed. Defaults to
        ``False``
    :raises ValueError: if ``max_bytes`` is not a positive integer, if
        ``expected_digest`` is not a valid SHA-256 digest, if the source produces more
        than ``max_bytes`` bytes, or if the computed digest does not match
        ``expected_digest``
    """
    if not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")

    if isinstance(expected_digest, str):
        expected = bytes.fromhex(expected_digest)
    else:
        expected = expected_digest

    if len(expected) != 32:
        raise ValueError("expected_digest must be a SHA-256 digest (32 bytes)")

    hasher = hashlib.sha256()
    total_bytes = 0
    try:
        async for chunk in source:
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise ValueError(
                    f"stream exceeds the maximum allowed length of {max_bytes} bytes"
                )

            hasher.update(chunk)
            await sink.write(chunk)

        if not hmac.compare_digest(hasher.digest(), expected):
            raise ValueError(
                f"digest mismatch: expected {expected.hex()}, "
                f"received {hasher.hexdigest()}"
            )
    except BaseException:
        # A failure before commit: abort the transaction. The abort is shielded
        # against cancellation so that it always gets to run, and any error it
        # raises is suppressed so that it cannot mask the original error.
        with suppress(BaseException), CancelScope(shield=True):
            await sink.abort()

        if close_source:
            with suppress(BaseException):
                await _aclose_shielded(source)

        raise

    # All checks passed: commit the transaction. If commit fails, the error is
    # propagated as-is -- the sink owns the transaction state, so we must not abort
    # or claim success. We still close the source if we own it.
    try:
        await sink.commit()
    except BaseException:
        if close_source:
            with suppress(BaseException):
                await _aclose_shielded(source)

        raise

    if close_source:
        await _aclose_shielded(source)
