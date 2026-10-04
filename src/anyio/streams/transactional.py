from __future__ import annotations

__all__ = (
    "DigestMismatchError",
    "LengthMismatchError",
    "SizeLimitExceededError",
    "TransactionalSink",
    "TransferError",
    "TransferResult",
    "transfer_to_sink",
)

import hashlib
import hmac
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .. import EndOfStream

if TYPE_CHECKING:
    from ..abc import ByteReceiveStream


class TransferError(Exception):
    """Base class for errors raised by :func:`transfer_to_sink` itself."""


class DigestMismatchError(TransferError):
    """Raised when the SHA-256 digest of the transferred data does not match the
    expected digest."""


class LengthMismatchError(TransferError):
    """Raised when the number of transferred bytes does not match the expected
    length."""


class SizeLimitExceededError(TransferError):
    """Raised when the source stream yields more bytes than the configured
    limit."""


@runtime_checkable
class TransactionalSink(Protocol):
    """
    A destination for byte stream transfers with transactional semantics.

    The sink must provide three operations:

    * :meth:`write` appends a chunk of bytes to the pending transaction
    * :meth:`commit` atomically publishes everything written so far
    * :meth:`abort` discards everything written so far

    .. note:: :func:`transfer_to_sink` only relies on these guarantees from the
        sink itself. It does **not** promise that an arbitrary writer can be
        rolled back; only sinks that genuinely implement this protocol should
        be used.
    """

    async def write(self, data: bytes) -> None:
        """Append *data* to the pending transaction."""
        ...  # pragma: no cover

    async def commit(self) -> None:
        """Atomically publish all previously written data."""
        ...  # pragma: no cover

    async def abort(self) -> None:
        """Discard all previously written data."""
        ...  # pragma: no cover


@dataclass(frozen=True)
class TransferResult:
    """The outcome of a successful :func:`transfer_to_sink` call."""

    #: the total number of bytes transferred
    bytes_transferred: int
    #: the SHA-256 hex digest of the transferred data
    sha256: str


def _normalize_expected_digest(expected: str | bytes) -> str:
    if isinstance(expected, str):
        return expected.lower()

    return bytes(expected).hex()


async def transfer_to_sink(
    source: ByteReceiveStream,
    sink: TransactionalSink,
    *,
    expected_sha256: str | bytes | None = None,
    expected_size: int | None = None,
    max_bytes: int | None = None,
    chunk_size: int = 65536,
    close_source: bool = True,
) -> TransferResult:
    """
    Transfer bytes from a single source stream into a transactional sink.

    The data is read chunk by chunk, hashed with SHA-256 on the fly and written
    straight to the sink; the full content is never collected in memory. The
    sink is committed **only** if all of the following hold:

    * the source stream reached EOF
    * the total length matches *expected_size* (if given) and never exceeded
      *max_bytes* (if given)
    * the SHA-256 digest matches *expected_sha256* (if given, as a hex string
      or as raw bytes)

    On any other outcome – a verification failure, a read/write error,
    cancellation, or a failed commit – the sink's :meth:`~TransactionalSink.abort`
    is called first (at most once) and the original error is then re-raised to
    the caller. A failure of ``abort()`` itself never masks the original error.
    If :meth:`~TransactionalSink.commit` fails, the error is propagated as-is;
    success is never reported before ``commit()`` has completed.

    The transfer is performed sequentially in the current task; no background
    tasks are spawned.

    :param source: the byte stream to read from
    :param sink: the transactional sink to write to
    :param expected_sha256: expected SHA-256 digest of the content, either as a
        hex string (case insensitive) or as raw bytes
    :param expected_size: expected total number of bytes, or ``None`` to accept
        any length (up to *max_bytes*)
    :param max_bytes: maximum number of bytes to accept, or ``None`` for no
        limit
    :param chunk_size: maximum number of bytes to read from the source at a
        time
    :param close_source: if ``True`` (the default), this function takes
        ownership of *source* and closes it exactly once when the transfer
        ends, regardless of the outcome; pass ``False`` to retain ownership
    :return: the transfer result, containing the byte count and the digest
    :raises DigestMismatchError: if the digest of the received data did not
        match *expected_sha256*
    :raises LengthMismatchError: if the number of received bytes did not match
        *expected_size*
    :raises SizeLimitExceededError: if the source yielded more than *max_bytes*
        bytes

    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    if expected_size is not None and expected_size < 0:
        raise ValueError("expected_size must not be negative")

    if max_bytes is not None and max_bytes < 0:
        raise ValueError("max_bytes must not be negative")

    expected_digest = (
        _normalize_expected_digest(expected_sha256)
        if expected_sha256 is not None
        else None
    )

    hasher = hashlib.sha256()
    total = 0
    aborted = False
    source_closed = False
    try:
        while True:
            try:
                chunk = await source.receive(chunk_size)
            except EndOfStream:
                break

            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise SizeLimitExceededError(
                    f"source stream exceeded the limit of {max_bytes} bytes"
                )

            hasher.update(chunk)
            await sink.write(chunk)

        if expected_size is not None and total != expected_size:
            raise LengthMismatchError(
                f"expected {expected_size} bytes, but received {total}"
            )

        digest = hasher.hexdigest()
        if expected_digest is not None and not hmac.compare_digest(
            digest, expected_digest
        ):
            raise DigestMismatchError(
                f"SHA-256 digest mismatch: expected {expected_digest}, got {digest}"
            )

        await sink.commit()
        return TransferResult(bytes_transferred=total, sha256=digest)
    except BaseException as exc:
        if not aborted:
            aborted = True
            try:
                await sink.abort()
            except Exception as abort_exc:
                # A failing abort() must never mask the original error.
                if hasattr(exc, "add_note"):
                    exc.add_note(f"abort() also failed: {abort_exc!r}")

        raise
    finally:
        if close_source and not source_closed:
            source_closed = True
            await source.aclose()
