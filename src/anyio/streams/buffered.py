from __future__ import annotations

__all__ = (
    "BufferedByteReceiveStream",
    "BufferedByteStream",
    "BufferedConnectable",
    "RecordReceiveStream",
)

import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, SupportsIndex

from .. import (
    BusyResourceError,
    ClosedResourceError,
    DelimiterNotFound,
    EndOfStream,
    IncompleteRead,
    IncompleteRecord,
    RecordStateError,
)
from ..abc import (
    AnyByteReceiveStream,
    AnyByteStream,
    AnyByteStreamConnectable,
    ByteReceiveStream,
    ByteStream,
    ByteStreamConnectable,
)

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


@dataclass(eq=False)
class BufferedByteReceiveStream(ByteReceiveStream):
    """
    Wraps any bytes-based receive stream and uses a buffer to provide sophisticated
    receiving capabilities in the form of a byte stream.
    """

    receive_stream: AnyByteReceiveStream
    _buffer: bytearray = field(init=False, default_factory=bytearray)
    _closed: bool = field(init=False, default=False)

    async def aclose(self) -> None:
        await self.receive_stream.aclose()
        self._closed = True

    @property
    def buffer(self) -> bytes:
        """The bytes currently in the buffer."""
        return bytes(self._buffer)

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return self.receive_stream.extra_attributes

    def feed_data(self, data: Iterable[SupportsIndex], /) -> None:
        """
        Append data directly into the buffer.

        Any data in the buffer will be consumed by receive operations before receiving
        anything from the wrapped stream.

        :param data: the data to append to the buffer (can be bytes or anything else
            that supports ``__index__()``)

        """
        self._buffer.extend(data)

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")

        if self._closed:
            raise ClosedResourceError

        if self._buffer:
            chunk = bytes(self._buffer[:max_bytes])
            del self._buffer[:max_bytes]
            return chunk
        elif isinstance(self.receive_stream, ByteReceiveStream):
            return await self.receive_stream.receive(max_bytes)
        else:
            # With a bytes-oriented object stream, we need to handle any surplus bytes
            # we get from the receive() call
            chunk = await self.receive_stream.receive()
            if len(chunk) > max_bytes:
                # Save the surplus bytes in the buffer
                self._buffer.extend(chunk[max_bytes:])
                return chunk[:max_bytes]
            else:
                return chunk

    async def receive_exactly(self, nbytes: int) -> bytes:
        """
        Read exactly the given amount of bytes from the stream.

        :param nbytes: the number of bytes to read
        :return: the bytes read
        :raises ~anyio.IncompleteRead: if the stream was closed before the requested
            amount of bytes could be read from the stream

        """
        while True:
            remaining = nbytes - len(self._buffer)
            if remaining <= 0:
                retval = self._buffer[:nbytes]
                del self._buffer[:nbytes]
                return bytes(retval)

            try:
                if isinstance(self.receive_stream, ByteReceiveStream):
                    chunk = await self.receive_stream.receive(remaining)
                else:
                    chunk = await self.receive_stream.receive()
            except EndOfStream as exc:
                raise IncompleteRead from exc

            self._buffer.extend(chunk)

    async def receive_until(self, delimiter: bytes, max_bytes: int) -> bytes:
        """
        Read from the stream until the delimiter is found or max_bytes have been read.

        :param delimiter: the marker to look for in the stream
        :param max_bytes: maximum number of bytes that will be read before raising
            :exc:`~anyio.DelimiterNotFound`
        :return: the bytes read (not including the delimiter)
        :raises ~anyio.IncompleteRead: if the stream was closed before the delimiter
            was found
        :raises ~anyio.DelimiterNotFound: if the delimiter is not found within the
            bytes read up to the maximum allowed

        """
        offset = 0
        while True:
            # Check if the delimiter can be found in the current buffer
            index, offset = self._find_delimiter(delimiter, offset)
            if index >= 0:
                found = self._buffer[:index]
                del self._buffer[: index + len(delimiter) :]
                return bytes(found)

            # Check if the buffer is already at or over the limit
            if len(self._buffer) >= max_bytes:
                raise DelimiterNotFound(max_bytes)

            # Read more data into the buffer from the socket
            try:
                data = await self.receive_stream.receive()
            except EndOfStream as exc:
                raise IncompleteRead from exc

            # Add the new data to the buffer
            self._buffer.extend(data)

    def _find_delimiter(self, delimiter: bytes, offset: int) -> tuple[int, int]:
        """
        Search the buffer for the delimiter, starting from the given offset.

        :param delimiter: the marker to look for in the buffer
        :param offset: the buffer index to start the search from
        :return: a tuple of (index of the delimiter, or -1 if not found; the offset
            to continue the search from when more data has been appended to the
            buffer)

        """
        index = self._buffer.find(delimiter, offset)
        next_offset = max(len(self._buffer) - len(delimiter) + 1, 0)
        return index, next_offset


@dataclass(eq=False)
class RecordReceiveStream(BufferedByteReceiveStream):
    """
    Wraps any bytes-based receive stream and provides record-oriented reading on
    top of the buffering capabilities of :class:`BufferedByteReceiveStream`.

    Records are read with :meth:`read_until` which enforces a per-record byte
    limit. When a record exceeds the limit, the stream enters the
    ``needs_discard`` state where further reads are rejected until the rest of
    the oversized record has been discarded with :meth:`discard_until`. This
    ensures that the remainder of an oversized record is never mistaken for the
    next record.

    .. versionadded:: 4.16
    """

    _state: Literal["ready", "needs_discard"] = field(init=False, default="ready")
    _reading: bool = field(init=False, default=False)

    @property
    def state(self) -> str:
        """
        The current state of the record reader.

        The state is ``ready`` when records can be read with :meth:`read_until`,
        and ``needs_discard`` when an oversized record must first be discarded
        with :meth:`discard_until`.
        """
        return self._state

    def _start_read_operation(self) -> None:
        if self._closed:
            raise ClosedResourceError

        if self._reading:
            raise BusyResourceError("reading from")

        self._reading = True

    async def read_until(
        self,
        delimiter: bytes,
        max_bytes: int,
        *,
        include_delimiter: bool = False,
        on_eof: Literal["raise", "return"] = "raise",
    ) -> bytes:
        """
        Read a single record from the stream, terminated by the delimiter.

        If the record exceeds ``max_bytes`` bytes (not counting the delimiter),
        :exc:`~anyio.DelimiterNotFound` is raised and the stream enters the
        ``needs_discard`` state. While in that state, further calls to this
        method raise :exc:`~anyio.RecordStateError` until the remainder of the
        oversized record has been discarded with :meth:`discard_until`.

        This method is cancellation safe: if the read is cancelled, no bytes are
        lost and the read can be retried.

        :param delimiter: the marker terminating the record
        :param max_bytes: the maximum number of bytes a record may contain (not
            counting the delimiter) before it is considered oversized
        :param include_delimiter: if ``True``, include the delimiter in the
            returned record
        :param on_eof: the policy applied when the end of the stream is reached
            before the delimiter: ``"raise"`` to raise
            :exc:`~anyio.IncompleteRecord`, or ``"return"`` to return (and
            consume) the partial record
        :return: the record, without the delimiter (unless ``include_delimiter``
            was ``True``)
        :raises ~anyio.RecordStateError: if the stream is in the
            ``needs_discard`` state
        :raises ~anyio.DelimiterNotFound: if the record exceeds ``max_bytes``
            bytes
        :raises ~anyio.IncompleteRecord: if the end of the stream is reached
            before the delimiter and ``on_eof`` is ``"raise"``
        :raises ~anyio.BusyResourceError: if another record read or discard
            operation is already in progress

        """
        if not delimiter:
            raise ValueError("delimiter must not be empty")

        if max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer")

        if on_eof not in ("raise", "return"):
            raise ValueError("on_eof must be either 'raise' or 'return'")

        if self._closed:
            raise ClosedResourceError

        if self._state != "ready":
            raise RecordStateError(
                "an oversized record is pending discard; call discard_until() "
                "before reading more records"
            )

        self._start_read_operation()
        try:
            delimiter_size = len(delimiter)
            offset = 0
            while True:
                # Check if the delimiter can be found in the current buffer
                index, offset = self._find_delimiter(delimiter, offset)
                if index >= 0:
                    if index > max_bytes:
                        self._state = "needs_discard"
                        raise DelimiterNotFound(max_bytes)

                    end = index + delimiter_size
                    record_end = end if include_delimiter else index
                    record = bytes(self._buffer[:record_end])
                    del self._buffer[:end]
                    return record

                # Check if the record is provably over the limit, leaving room
                # for a partial delimiter at the end of the buffer
                if len(self._buffer) - (delimiter_size - 1) > max_bytes:
                    self._state = "needs_discard"
                    raise DelimiterNotFound(max_bytes)

                # Read more data into the buffer from the stream
                try:
                    data = await self.receive_stream.receive()
                except EndOfStream:
                    if on_eof == "return":
                        partial = bytes(self._buffer)
                        del self._buffer[:]
                        return partial

                    raise IncompleteRecord(bytes(self._buffer)) from None

                self._buffer.extend(data)
        finally:
            self._reading = False

    async def discard_until(self, delimiter: bytes, max_bytes: int = 65536) -> int:
        """
        Discard the remainder of an oversized record, up to and including the
        next delimiter.

        This method is used to recover from the ``needs_discard`` state (see
        :meth:`read_until`). Once the delimiter terminating the oversized record
        has been consumed, the stream returns to the ``ready`` state.

        Discarding is limited to a budget of ``max_bytes`` bytes per call. If
        the budget runs out (or the operation is cancelled) before the delimiter
        is found, the method can simply be called again to continue discarding;
        bytes already discarded are never re-exposed. The last
        ``len(delimiter) - 1`` bytes are always kept in the buffer so that a
        delimiter straddling the budget boundary is never partially discarded.

        Does nothing and returns 0 if the stream is not in the
        ``needs_discard`` state.

        :param delimiter: the marker terminating the oversized record
        :param max_bytes: the maximum number of bytes to discard in this call
        :return: the number of bytes discarded in this call (including the
            delimiter, if the end of the record was reached)
        :raises ~anyio.IncompleteRecord: if the end of the stream is reached
            before the delimiter
        :raises ~anyio.BusyResourceError: if another record read or discard
            operation is already in progress

        """
        if not delimiter:
            raise ValueError("delimiter must not be empty")

        if max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")

        self._start_read_operation()
        discarded = 0
        try:
            if self._state != "needs_discard":
                return 0

            delimiter_size = len(delimiter)
            offset = 0
            while True:
                index, offset = self._find_delimiter(delimiter, offset)
                if index >= 0 and discarded + index + delimiter_size <= max_bytes:
                    # The delimiter is within the budget: finish the record
                    end = index + delimiter_size
                    del self._buffer[:end]
                    discarded += end
                    self._state = "ready"
                    return discarded

                # Determine how many bytes can be discarded without touching a
                # (possibly partial) delimiter
                if index >= 0:
                    safe = index
                else:
                    safe = max(len(self._buffer) - delimiter_size + 1, 0)

                if safe and discarded < max_bytes:
                    amount = min(safe, max_bytes - discarded)
                    del self._buffer[:amount]
                    discarded += amount
                    offset = 0
                    continue

                if index >= 0 or discarded >= max_bytes:
                    # The delimiter is within reach but consuming it would exceed
                    # the budget, or the budget has been exhausted
                    return discarded

                # Read more data into the buffer from the stream
                try:
                    data = await self.receive_stream.receive()
                except EndOfStream:
                    raise IncompleteRecord(bytes(self._buffer)) from None

                self._buffer.extend(data)
        finally:
            self._reading = False


class BufferedByteStream(BufferedByteReceiveStream, ByteStream):
    """
    A full-duplex variant of :class:`BufferedByteReceiveStream`. All writes are passed
    through to the wrapped stream as-is.
    """

    def __init__(self, stream: AnyByteStream):
        """
        :param stream: the stream to be wrapped

        """
        super().__init__(stream)
        self._stream = stream

    @override
    async def send_eof(self) -> None:
        await self._stream.send_eof()

    @override
    async def send(self, item: bytes) -> None:
        await self._stream.send(item)


class BufferedConnectable(ByteStreamConnectable):
    """
    Wraps a byte stream connectable to produce :class:`BufferedByteStream` connections.

    Use this when you want the streams returned by :meth:`connect` to have the buffered
    receive API (e.g. :meth:`~BufferedByteReceiveStream.receive_exactly` and
    :meth:`~BufferedByteReceiveStream.receive_until`).

    :param connectable: the byte stream connectable to wrap
    """

    def __init__(self, connectable: AnyByteStreamConnectable):
        """
        :param connectable: the connectable to wrap

        """
        self.connectable = connectable

    @override
    async def connect(self) -> BufferedByteStream:
        stream = await self.connectable.connect()
        return BufferedByteStream(stream)
