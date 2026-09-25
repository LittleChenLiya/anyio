from __future__ import annotations

__all__ = (
    "RecordReceiveStream",
    "RecordStreamState",
    "RecordStreamStateError",
    "RecordTooLong",
)

from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, SupportsIndex

from .. import (
    BusyResourceError,
    ClosedResourceError,
    DelimiterNotFound,
    EndOfStream,
    IncompleteRead,
    IncompleteRecord,
)
from .buffered import BufferedByteReceiveStream

if TYPE_CHECKING:
    from ..abc import AnyByteReceiveStream


class RecordStreamState(Enum):
    """Describes the state of a :class:`RecordReceiveStream`."""

    READY = "ready"
    """The stream is ready to read the next record."""

    NEEDS_DISCARD = "needs_discard"
    """
    A record exceeded the size limit; its remainder must be discarded with
    :meth:`RecordReceiveStream.discard_until` before reading can resume.
    """


class RecordTooLong(DelimiterNotFound):
    """
    Raised during :meth:`RecordReceiveStream.read_until` when a record exceeds the
    maximum allowed number of bytes.

    When this is raised, the stream enters the
    :attr:`~RecordStreamState.NEEDS_DISCARD` state, and the remainder of the
    oversized record must be discarded with
    :meth:`RecordReceiveStream.discard_until` before new records can be read.
    """

    def __init__(self, max_bytes: int) -> None:
        Exception.__init__(
            self,
            f"The record exceeded the maximum allowed size of {max_bytes} bytes",
        )
        self.max_bytes = max_bytes


class RecordStreamStateError(RuntimeError):
    """
    Raised when a record stream operation is not allowed in the stream's current
    :class:`state <RecordStreamState>` (for example when trying to read a record
    while an oversized record is still waiting to be discarded).
    """


class RecordReceiveStream:
    """
    A record-oriented reading session on top of
    :class:`~anyio.streams.buffered.BufferedByteReceiveStream`.

    Records are read with :meth:`read_until`, which reuses the delimiter search of
    the underlying buffered stream. If a record grows beyond the per-record byte
    limit, the stream enters the :attr:`~RecordStreamState.NEEDS_DISCARD` state in
    which further reads are refused until the remainder of the oversized record has
    been discarded with :meth:`discard_until`. Discarding consumes input up to and
    including the next delimiter, optionally within a per-call work budget, and can
    be safely resumed after cancellation or budget exhaustion without ever
    re-exposing already discarded bytes or misframing the following record.

    :param receive_stream: the stream to wrap (plain receive streams are
        automatically wrapped in
        :class:`~anyio.streams.buffered.BufferedByteReceiveStream`)
    :param delimiter: the default record delimiter
    :param max_record_bytes: the default maximum size of a single record, in bytes
    :param eof_policy: what to do when the stream ends in the middle of a record:
        ``"raise"`` (the default) raises ``~anyio.IncompleteRecord``, while
        ``"return"`` returns the partial record instead
    """

    def __init__(
        self,
        receive_stream: AnyByteReceiveStream | BufferedByteReceiveStream,
        delimiter: bytes,
        max_record_bytes: int,
        *,
        eof_policy: Literal["raise", "return"] = "raise",
    ) -> None:
        if not delimiter:
            raise ValueError("delimiter must not be empty")

        if max_record_bytes < 1:
            raise ValueError("max_record_bytes must be a positive integer")

        if eof_policy not in ("raise", "return"):
            raise ValueError("eof_policy must be either 'raise' or 'return'")

        if isinstance(receive_stream, BufferedByteReceiveStream):
            self._buffered = receive_stream
        else:
            self._buffered = BufferedByteReceiveStream(receive_stream)

        self._delimiter = delimiter
        self._max_record_bytes = max_record_bytes
        self._eof_policy = eof_policy
        self._state = RecordStreamState.READY
        self._closed = False
        self._busy = False

    @property
    def state(self) -> RecordStreamState:
        """The current state of the record reading session."""
        return self._state

    @property
    def delimiter(self) -> bytes:
        """The default record delimiter."""
        return self._delimiter

    @property
    def max_record_bytes(self) -> int:
        """The default maximum size of a single record, in bytes."""
        return self._max_record_bytes

    @property
    def buffer(self) -> bytes:
        """The bytes currently in the buffer."""
        return self._buffered.buffer

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return self._buffered.extra_attributes

    def feed_data(self, data: Iterable[SupportsIndex], /) -> None:
        """
        Append data directly into the buffer.

        Any data in the buffer will be consumed by read operations before receiving
        anything from the wrapped stream.

        :param data: the data to append to the buffer (can be bytes or anything else
            that supports ``__index__()``)

        """
        self._buffered.feed_data(data)

    async def aclose(self) -> None:
        await self._buffered.aclose()
        self._closed = True

    def _begin_operation(self, action: str) -> None:
        if self._closed:
            raise ClosedResourceError

        if self._busy:
            raise BusyResourceError(action)

        self._busy = True

    async def read_until(
        self,
        delimiter: bytes | None = None,
        max_bytes: int | None = None,
        *,
        include_delimiter: bool = False,
    ) -> bytes:
        """
        Read a single record from the stream.

        :param delimiter: the record delimiter (defaults to the delimiter given to
            the constructor)
        :param max_bytes: the maximum record size in bytes (defaults to the value
            given to the constructor)
        :param include_delimiter: if ``True``, include the delimiter in the
            returned bytes
        :return: the record bytes
        :raises RecordStreamStateError: if a previous record exceeded the size
            limit and its remainder has not been discarded yet
        :raises RecordTooLong: if the record exceeded ``max_bytes`` (the stream
            enters the ``needs_discard`` state)
        :raises anyio.IncompleteRecord: if the stream ended in the middle of a
            record and the EOF policy is ``"raise"``
        :raises anyio.EndOfStream: if the stream ended with no bytes left over
        :raises anyio.BusyResourceError: if another task is already reading from or
            discarding from this stream

        """
        self._begin_operation("reading from")
        try:
            if self._state is RecordStreamState.NEEDS_DISCARD:
                raise RecordStreamStateError(
                    "the previous record exceeded the size limit; discard its "
                    "remainder with discard_until() before reading more records"
                )

            if delimiter is None:
                delimiter = self._delimiter

            if not delimiter:
                raise ValueError("delimiter must not be empty")

            if max_bytes is None:
                max_bytes = self._max_record_bytes

            try:
                record = await self._buffered.receive_until(delimiter, max_bytes)
            except DelimiterNotFound as exc:
                self._state = RecordStreamState.NEEDS_DISCARD
                raise RecordTooLong(max_bytes) from exc
            except IncompleteRead as exc:
                partial = self._buffered.buffer
                if not partial:
                    raise EndOfStream from exc

                if self._eof_policy == "return":
                    # Consume the partial record from the buffer
                    await self._buffered.receive_exactly(len(partial))
                    return partial

                raise IncompleteRecord(partial) from exc

            if include_delimiter:
                return record + delimiter

            return record
        finally:
            self._busy = False

    async def discard_until(
        self, delimiter: bytes | None = None, *, max_bytes: int | None = None
    ) -> int:
        """
        Discard the remainder of an oversized record.

        Consumes bytes up to and including the next delimiter, then returns the
        stream to the ``ready`` state. Bytes already discarded are never
        re-exposed, so a call that was cancelled or ran out of its work budget can
        simply be retried to continue discarding where it left off.

        :param delimiter: the record delimiter (defaults to the delimiter given to
            the constructor)
        :param max_bytes: the work budget for this call, that is, the maximum
            number of bytes to discard before returning; if the budget runs out
            before the delimiter is found, the stream stays in the
            ``needs_discard`` state and discarding can be resumed with another
            call
        :return: the number of bytes discarded by this call
        :raises RecordStreamStateError: if the stream is not in the
            ``needs_discard`` state
        :raises anyio.BusyResourceError: if another task is already reading from or
            discarding from this stream

        """
        self._begin_operation("discarding from")
        try:
            if self._state is not RecordStreamState.NEEDS_DISCARD:
                raise RecordStreamStateError("there is no oversized record to discard")

            if delimiter is None:
                delimiter = self._delimiter

            if not delimiter:
                raise ValueError("delimiter must not be empty")

            if max_bytes is not None and max_bytes < 1:
                raise ValueError("max_bytes must be a positive integer")

            discarded = 0
            keep = len(delimiter) - 1
            while True:
                buffer = self._buffered.buffer
                index = buffer.find(delimiter)
                if index >= 0:
                    # Found the end of the oversized record
                    end = index + len(delimiter)
                    await self._buffered.receive_exactly(end)
                    discarded += end
                    self._state = RecordStreamState.READY
                    return discarded

                if max_bytes is not None and discarded >= max_bytes:
                    return discarded

                # Discard buffered bytes, but keep any bytes that could be the
                # start of a delimiter straddling a chunk boundary
                removable = max(len(buffer) - keep, 0)
                if max_bytes is not None:
                    removable = min(removable, max_bytes - discarded)

                if removable > 0:
                    await self._buffered.receive_exactly(removable)
                    discarded += removable
                else:
                    try:
                        chunk = await self._buffered.receive_stream.receive()
                    except EndOfStream:
                        # The oversized record was never terminated; discard what
                        # is left and consider the discard complete
                        leftover = len(self._buffered.buffer)
                        if leftover:
                            await self._buffered.receive_exactly(leftover)
                            discarded += leftover

                        self._state = RecordStreamState.READY
                        return discarded

                    self._buffered.feed_data(chunk)
        finally:
            self._busy = False
