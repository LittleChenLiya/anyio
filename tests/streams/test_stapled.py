from __future__ import annotations

import sys
from collections import deque
from collections.abc import Iterable
from dataclasses import InitVar, dataclass, field
from typing import TypeVar

import pytest

from anyio import (
    CancelScope,
    ClosedResourceError,
    EndOfStream,
    Event,
    create_task_group,
)
from anyio.abc import (
    ByteReceiveStream,
    ByteSendStream,
    ObjectReceiveStream,
    ObjectSendStream,
)
from anyio.streams.stapled import (
    HalfCloseableStapledByteStream,
    StapledByteStream,
    StapledObjectStream,
)

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


@dataclass
class DummyByteReceiveStream(ByteReceiveStream):
    data: InitVar[bytes]
    buffer: bytearray = field(init=False)
    _closed: bool = field(init=False, default=False)

    def __post_init__(self, data: bytes) -> None:
        self.buffer = bytearray(data)

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self._closed:
            raise ClosedResourceError

        data = bytes(self.buffer[:max_bytes])
        del self.buffer[:max_bytes]
        return data

    async def aclose(self) -> None:
        self._closed = True


@dataclass
class DummyByteSendStream(ByteSendStream):
    buffer: bytearray = field(init=False, default_factory=bytearray)
    _closed: bool = field(init=False, default=False)

    async def send(self, item: bytes) -> None:
        if self._closed:
            raise ClosedResourceError

        self.buffer.extend(item)

    async def aclose(self) -> None:
        self._closed = True


class TestStapledByteStream:
    @pytest.fixture
    def send_stream(self) -> DummyByteSendStream:
        return DummyByteSendStream()

    @pytest.fixture
    def receive_stream(self) -> DummyByteReceiveStream:
        return DummyByteReceiveStream(b"hello, world")

    @pytest.fixture
    def stapled(
        self, send_stream: DummyByteSendStream, receive_stream: DummyByteReceiveStream
    ) -> StapledByteStream:
        return StapledByteStream(send_stream, receive_stream)

    async def test_receive_send(
        self, stapled: StapledByteStream, send_stream: DummyByteSendStream
    ) -> None:
        assert await stapled.receive(3) == b"hel"
        assert await stapled.receive() == b"lo, world"
        assert await stapled.receive() == b""

        await stapled.send(b"how are you ")
        await stapled.send(b"today?")
        assert stapled.send_stream is send_stream
        assert bytes(send_stream.buffer) == b"how are you today?"

    async def test_send_eof(self, stapled: StapledByteStream) -> None:
        await stapled.send_eof()
        await stapled.send_eof()
        with pytest.raises(ClosedResourceError):
            await stapled.send(b"world")

        assert await stapled.receive() == b"hello, world"

    async def test_aclose(self, stapled: StapledByteStream) -> None:
        await stapled.aclose()
        with pytest.raises(ClosedResourceError):
            await stapled.receive()
        with pytest.raises(ClosedResourceError):
            await stapled.send(b"")

    @pytest.mark.parametrize("max_bytes", [0, -1])
    async def test_receive_invalid_max_bytes(
        self, stapled: StapledByteStream, max_bytes: int
    ) -> None:
        with pytest.raises(ValueError, match="max_bytes must be a positive integer"):
            await stapled.receive(max_bytes)


T_Item = TypeVar("T_Item")


class CountingByteSendStream(ByteSendStream):
    """A send stream that counts close calls and can fail or block on close."""

    def __init__(
        self,
        *,
        close_error: Exception | None = None,
        close_gate: Event | None = None,
    ) -> None:
        self.buffer = bytearray()
        self.close_calls = 0
        self.close_started = Event()
        self.close_error = close_error
        self.close_gate = close_gate
        self.closed = False

    async def send(self, item: bytes) -> None:
        if self.closed:
            raise ClosedResourceError

        self.buffer.extend(item)

    async def aclose(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        if self.close_gate is not None:
            await self.close_gate.wait()

        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class CountingByteReceiveStream(ByteReceiveStream):
    """A receive stream that counts close calls and can fail on close."""

    def __init__(
        self, data: bytes = b"", *, close_error: Exception | None = None
    ) -> None:
        self.buffer = bytearray(data)
        self.close_calls = 0
        self.close_error = close_error
        self.closed = False

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self.closed:
            raise ClosedResourceError

        data = bytes(self.buffer[:max_bytes])
        del self.buffer[:max_bytes]
        return data

    async def aclose(self) -> None:
        self.close_calls += 1
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class TestHalfCloseableStapledByteStream:
    @pytest.fixture
    def send_stream(self) -> CountingByteSendStream:
        return CountingByteSendStream()

    @pytest.fixture
    def receive_stream(self) -> CountingByteReceiveStream:
        return CountingByteReceiveStream(b"reply")

    @pytest.fixture
    def stream(
        self,
        send_stream: CountingByteSendStream,
        receive_stream: CountingByteReceiveStream,
    ) -> HalfCloseableStapledByteStream:
        return HalfCloseableStapledByteStream(send_stream, receive_stream)

    async def test_send_eof_only_closes_send_side(
        self,
        stream: HalfCloseableStapledByteStream,
        send_stream: CountingByteSendStream,
        receive_stream: CountingByteReceiveStream,
    ) -> None:
        await stream.send(b"hello")
        assert bytes(send_stream.buffer) == b"hello"

        await stream.send_eof()
        assert stream.send_closed
        assert not stream.receive_closed
        assert send_stream.close_calls == 1
        assert receive_stream.close_calls == 0

        # The receive side must still be usable after the send side is done
        assert await stream.receive() == b"reply"

        # Sending after send_eof() must fail
        with pytest.raises(ClosedResourceError):
            await stream.send(b"more")

    async def test_send_eof_is_idempotent(
        self,
        stream: HalfCloseableStapledByteStream,
        send_stream: CountingByteSendStream,
    ) -> None:
        await stream.send_eof()
        await stream.send_eof()
        assert send_stream.close_calls == 1

    async def test_aclose_closes_both_sides(
        self,
        stream: HalfCloseableStapledByteStream,
        send_stream: CountingByteSendStream,
        receive_stream: CountingByteReceiveStream,
    ) -> None:
        await stream.aclose()
        assert stream.send_closed
        assert stream.receive_closed
        assert send_stream.close_calls == 1
        assert receive_stream.close_calls == 1
        with pytest.raises(ClosedResourceError):
            await stream.send(b"")
        with pytest.raises(ClosedResourceError):
            await stream.receive()

    async def test_aclose_is_idempotent(
        self,
        stream: HalfCloseableStapledByteStream,
        send_stream: CountingByteSendStream,
        receive_stream: CountingByteReceiveStream,
    ) -> None:
        await stream.aclose()
        await stream.aclose()
        assert send_stream.close_calls == 1
        assert receive_stream.close_calls == 1

    async def test_aclose_send_error_still_closes_receive(self) -> None:
        send_stream = CountingByteSendStream(close_error=ValueError("send boom"))
        receive_stream = CountingByteReceiveStream(b"reply")
        stream = HalfCloseableStapledByteStream(send_stream, receive_stream)
        with pytest.raises(ValueError, match="send boom"):
            await stream.aclose()

        assert send_stream.close_calls == 1
        assert receive_stream.close_calls == 1

        # A repeat call must not re-close the underlying streams
        await stream.aclose()
        assert send_stream.close_calls == 1
        assert receive_stream.close_calls == 1

    async def test_aclose_receive_error_still_raises(self) -> None:
        send_stream = CountingByteSendStream()
        receive_stream = CountingByteReceiveStream(
            b"reply", close_error=ValueError("receive boom")
        )
        stream = HalfCloseableStapledByteStream(send_stream, receive_stream)
        with pytest.raises(ValueError, match="receive boom"):
            await stream.aclose()

        assert send_stream.close_calls == 1
        assert receive_stream.close_calls == 1

    async def test_aclose_both_errors_raised_as_group(self) -> None:
        send_stream = CountingByteSendStream(close_error=ValueError("send boom"))
        receive_stream = CountingByteReceiveStream(
            b"reply", close_error=KeyError("receive boom")
        )
        stream = HalfCloseableStapledByteStream(send_stream, receive_stream)
        with pytest.raises(ExceptionGroup) as exc_info:
            await stream.aclose()

        assert len(exc_info.value.exceptions) == 2
        assert send_stream.close_calls == 1
        assert receive_stream.close_calls == 1

    async def test_aclose_cancellation_still_finishes_cleanup(self) -> None:
        close_gate = Event()
        send_stream = CountingByteSendStream(close_gate=close_gate)
        receive_stream = CountingByteReceiveStream(b"reply")
        stream = HalfCloseableStapledByteStream(send_stream, receive_stream)

        with CancelScope() as scope:
            async with create_task_group() as task_group:
                task_group.start_soon(stream.aclose)
                await send_stream.close_started.wait()
                scope.cancel()

        # The cancellation must have been passed back to the caller...
        assert scope.cancelled_caught

        # ...but not before the cleanup was finished: both sides were closed,
        # each exactly once
        assert send_stream.close_calls == 1
        assert receive_stream.close_calls == 1
        assert stream.send_closed
        assert stream.receive_closed

    async def test_aclose_after_send_eof_closes_receive_only_once(
        self,
        stream: HalfCloseableStapledByteStream,
        send_stream: CountingByteSendStream,
        receive_stream: CountingByteReceiveStream,
    ) -> None:
        await stream.send_eof()
        await stream.aclose()
        await stream.aclose()
        assert send_stream.close_calls == 1
        assert receive_stream.close_calls == 1


T_Item = TypeVar("T_Item")


@dataclass
class DummyObjectReceiveStream(ObjectReceiveStream[T_Item]):
    data: InitVar[Iterable[T_Item]]
    buffer: deque[T_Item] = field(init=False)
    _closed: bool = field(init=False, default=False)

    def __post_init__(self, data: Iterable[T_Item]) -> None:
        self.buffer = deque(data)

    async def receive(self) -> T_Item:
        if self._closed:
            raise ClosedResourceError
        if not self.buffer:
            raise EndOfStream

        return self.buffer.popleft()

    async def aclose(self) -> None:
        self._closed = True


@dataclass
class DummyObjectSendStream(ObjectSendStream[T_Item]):
    buffer: list[T_Item] = field(init=False, default_factory=list)
    _closed: bool = field(init=False, default=False)

    async def send(self, item: T_Item) -> None:
        if self._closed:
            raise ClosedResourceError

        self.buffer.append(item)

    async def aclose(self) -> None:
        self._closed = True


class DummyObjectSendStreamWithSendNowait(DummyObjectSendStream[T_Item]):
    def send_nowait(self, item: T_Item) -> None:
        if self._closed:
            raise ClosedResourceError

        self.buffer.append(item)


class DummyObjectSendStreamWithBrokenSendNowait(DummyObjectSendStream[T_Item]):
    def send_nowait(self, item: T_Item) -> None:
        raise AttributeError("no attribute 'nonexistent'")


class TestStapledObjectStream:
    @pytest.fixture
    def receive_stream(self) -> DummyObjectReceiveStream[str]:
        return DummyObjectReceiveStream(["hello", "world"])

    @pytest.fixture
    def send_stream(self) -> DummyObjectSendStream[str]:
        return DummyObjectSendStream[str]()

    @pytest.fixture
    def send_stream_with_send_nowait(self) -> DummyObjectSendStreamWithSendNowait[str]:
        return DummyObjectSendStreamWithSendNowait[str]()

    @pytest.fixture
    def stapled(
        self,
        receive_stream: DummyObjectReceiveStream[str],
        send_stream: DummyObjectSendStream[str],
    ) -> StapledObjectStream[str]:
        return StapledObjectStream(send_stream, receive_stream)

    @pytest.fixture
    def stapled_with_send_nowait(
        self,
        receive_stream: DummyObjectReceiveStream[str],
        send_stream_with_send_nowait: DummyObjectSendStreamWithSendNowait[str],
    ) -> StapledObjectStream[str]:
        return StapledObjectStream(send_stream_with_send_nowait, receive_stream)

    async def test_receive_send(
        self, stapled: StapledObjectStream[str], send_stream: DummyObjectSendStream[str]
    ) -> None:
        assert await stapled.receive() == "hello"
        assert await stapled.receive() == "world"
        with pytest.raises(EndOfStream):
            await stapled.receive()

        await stapled.send("how are you ")
        await stapled.send("today?")
        assert stapled.send_stream is send_stream
        assert send_stream.buffer == ["how are you ", "today?"]

    def test_send_nowait_implemented(
        self,
        stapled_with_send_nowait: StapledObjectStream[str],
        send_stream_with_send_nowait: DummyObjectSendStream[str],
    ) -> None:
        stapled_with_send_nowait.send_nowait("hello")
        assert send_stream_with_send_nowait.buffer == ["hello"]

    def test_send_nowait_not_implemented(
        self, stapled: StapledObjectStream[str], send_stream: DummyObjectSendStream[str]
    ) -> None:
        with pytest.raises(
            NotImplementedError,
            match="'send_nowait' method not implemented in "
            "<class 'tests.streams.test_stapled.DummyObjectSendStream'>",
        ):
            stapled.send_nowait("hello")

    def test_send_nowait_does_not_mask_attribute_error(
        self, receive_stream: DummyObjectReceiveStream[str]
    ) -> None:
        stapled = StapledObjectStream(
            DummyObjectSendStreamWithBrokenSendNowait[str](), receive_stream
        )
        with pytest.raises(AttributeError, match="no attribute 'nonexistent'"):
            stapled.send_nowait("hello")

    async def test_send_eof(self, stapled: StapledObjectStream[str]) -> None:
        await stapled.send_eof()
        await stapled.send_eof()
        with pytest.raises(ClosedResourceError):
            await stapled.send("world")

        assert await stapled.receive() == "hello"
        assert await stapled.receive() == "world"

    async def test_aclose(self, stapled: StapledObjectStream[str]) -> None:
        await stapled.aclose()
        with pytest.raises(ClosedResourceError):
            await stapled.receive()
        with pytest.raises(ClosedResourceError):
            await stapled.send(b"")  # type: ignore[arg-type]
