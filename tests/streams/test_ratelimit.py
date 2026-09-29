from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest

from anyio import (
    ClosedResourceError,
    EndOfStream,
    create_memory_object_stream,
    current_time,
    move_on_after,
    run,
)
from anyio.abc import ByteReceiveStream
from anyio.streams.ratelimit import RateLimitedByteReceiveStream


class InfiniteByteStream(ByteReceiveStream):
    """A byte stream that produces an endless supply of bytes."""

    def __init__(self) -> None:
        self.received_max_bytes: list[int] = []
        self.closed = False

    async def receive(self, max_bytes: int = 65536) -> bytes:
        self.received_max_bytes.append(max_bytes)
        return b"x" * max_bytes

    async def aclose(self) -> None:
        self.closed = True

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return {}


class OneShotObjectStream:
    """A bytes-oriented object stream that yields a single chunk, then EOF."""

    def __init__(self, chunk: bytes) -> None:
        self._chunk: bytes | None = chunk
        self.receive_calls = 0
        self.closed = False

    async def receive(self) -> bytes:
        self.receive_calls += 1
        if self._chunk is None:
            raise EndOfStream

        chunk, self._chunk = self._chunk, None
        return chunk

    async def aclose(self) -> None:
        self.closed = True

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return {}


def test_rate_limit_with_fake_clock() -> None:
    """
    Test that with a rate of 10 bytes/second and a burst size of 5, the cumulative
    number of delivered bytes never exceeds 5 + 10 * elapsed_seconds.
    """
    pytest.importorskip("trio")
    from trio.testing import MockClock

    clock = MockClock(autojump_threshold=0)

    async def main() -> list[tuple[float, int]]:
        stream = RateLimitedByteReceiveStream(InfiniteByteStream(), 10, 5)
        deliveries = []
        for _ in range(8):
            chunk = await stream.receive(3)
            deliveries.append((current_time(), len(chunk)))

        return deliveries

    deliveries = run(main, backend="trio", backend_options={"clock": clock})

    # The burst allowance of 5 is delivered immediately (chunks of 3 + 2), after
    # which one byte is delivered every 0.1 seconds
    expected_sizes = [3, 2, 1, 1, 1, 1, 1, 1]
    expected_times = [0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert [size for _, size in deliveries] == expected_sizes
    assert [time for time, _ in deliveries] == pytest.approx(expected_times)

    # Verify the cumulative delivery cap at every point in time
    cumulative = 0
    for timestamp, size in deliveries:
        cumulative += size
        assert cumulative <= 5 + 10 * timestamp + 1e-9

    assert cumulative == 11


async def test_cancelled_receive_then_reread() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RateLimitedByteReceiveStream(receive_stream, 100, 5)
    await send_stream.send(b"12345")
    assert await stream.receive() == b"12345"  # drains the burst allowance

    # The next receive has to wait for new allowance; cancel it mid-wait
    with move_on_after(0.05) as scope:
        await stream.receive()

    assert scope.cancelled_caught

    # Re-reading must deliver exactly the sent bytes, with none lost or duplicated
    await send_stream.send(b"abc")
    received = b""
    while len(received) < 3:
        received += await stream.receive()

    assert received == b"abc"

    send_stream.close()
    receive_stream.close()


async def test_oversized_chunk_is_buffered() -> None:
    """Surplus bytes from the wrapped stream must be buffered for later reads."""
    object_stream = OneShotObjectStream(b"x" * 25)
    stream = RateLimitedByteReceiveStream(object_stream, 100000, 10)

    received = await stream.receive(5)
    assert received == b"x" * 5
    assert object_stream.receive_calls == 1

    # The remaining 20 bytes must come from the internal buffer
    while len(received) < 25:
        received += await stream.receive(10)

    assert received == b"x" * 25
    assert object_stream.receive_calls == 1


async def test_end_of_stream() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](1)
    stream = RateLimitedByteReceiveStream(receive_stream, 1000, 10)
    await send_stream.send(b"abc")
    await send_stream.aclose()
    assert await stream.receive(2) == b"ab"
    assert await stream.receive(2) == b"c"  # delivered from the buffer
    with pytest.raises(EndOfStream):
        await stream.receive()

    receive_stream.close()


async def test_close() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](1)
    stream = RateLimitedByteReceiveStream(receive_stream, 1000, 10)
    await stream.aclose()
    with pytest.raises(ClosedResourceError):
        await stream.receive()

    # The wrapped stream must have been closed too
    with pytest.raises(ClosedResourceError):
        await receive_stream.receive()

    send_stream.close()


async def test_invalid_arguments() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](1)
    with pytest.raises(ValueError, match="bytes_per_second"):
        RateLimitedByteReceiveStream(receive_stream, 0, 5)

    with pytest.raises(ValueError, match="bytes_per_second"):
        RateLimitedByteReceiveStream(receive_stream, -1.5, 5)

    with pytest.raises(ValueError, match="burst_size"):
        RateLimitedByteReceiveStream(receive_stream, 10, 0)

    with pytest.raises(ValueError, match="burst_size"):
        RateLimitedByteReceiveStream(receive_stream, 10, -3)

    stream = RateLimitedByteReceiveStream(receive_stream, 10, 5)
    with pytest.raises(ValueError, match="max_bytes"):
        await stream.receive(0)

    with pytest.raises(ValueError, match="max_bytes"):
        await stream.receive(-1)

    send_stream.close()
    receive_stream.close()
