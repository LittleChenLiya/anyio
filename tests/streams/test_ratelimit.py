from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import pytest

from anyio import (
    ClosedResourceError,
    EndOfStream,
    Event,
    create_memory_object_stream,
    create_task_group,
    wait_all_tasks_blocked,
)
from anyio.abc import ByteReceiveStream
from anyio.streams.ratelimit import RateLimitedByteReceiveStream


class FakeClock:
    """Deterministic clock driven manually by the test.

    Time only moves when ``advance()`` is called. Sleeping tasks register
    their deadline and block on an event; advancing the clock pops the due
    deadlines in order and signals them. After each signal the driver waits
    for the resumed task to register its next deadline (or become blocked on
    something else), which is robust against the scheduling checkpoints that
    memory object stream receives perform.
    """

    def __init__(self) -> None:
        self.time = 0.0
        self._deadlines: deque[tuple[float, Event]] = deque()
        self._registered: Event | None = None

    def current_time(self) -> float:
        return self.time

    async def sleep_until(self, deadline: float) -> None:
        if deadline <= self.time + 1e-9:
            return

        wake = Event()
        self._deadlines.append((deadline, wake))
        if self._registered is not None:
            registered = self._registered
            self._registered = None
            registered.set()

        await wake.wait()

    async def advance(self, seconds: float) -> None:
        target_time = round(self.time + seconds, 9)

        # Ensure tasks have registered their deadlines at the current time
        await self._await_next_registration()

        while self._deadlines and self._deadlines[0][0] <= target_time + 1e-9:
            deadline, wake = self._deadlines.popleft()
            self.time = max(self.time, min(deadline, target_time))
            wake.set()
            await self._await_next_registration()

        self.time = target_time
        await wait_all_tasks_blocked()

    async def _await_next_registration(self) -> None:
        """Run tasks until one registers a deadline or all are blocked.

        Tasks that traverse intermediate checkpoints block and resume without
        registering a clock deadline; ``wait_all_tasks_blocked()`` flushes such
        progress. The barrier event only fires on an actual registration.
        """
        self._registered = Event()
        registered = self._registered
        await wait_all_tasks_blocked()
        if self._registered is None:
            await registered.wait()
        else:
            # No registration; the tasks settled elsewhere
            self._registered = None


@pytest.fixture
def fake_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterable[FakeClock]:
    clock = FakeClock()
    monkeypatch.setattr(
        "anyio.streams.ratelimit._eventloop.current_time", clock.current_time
    )
    monkeypatch.setattr(
        "anyio.streams.ratelimit._eventloop.sleep_until", clock.sleep_until
    )
    yield clock


@dataclass(eq=False)
class ScriptedByteReceiveStream(ByteReceiveStream):
    """Byte receive stream that hands out scripted oversized chunks."""

    chunks: deque[bytes]
    receive_calls: int = 0
    requested_sizes: list[int] = field(default_factory=list)
    closed: bool = False

    @property
    def extra_attributes(self) -> dict[object, Callable[[], object]]:
        return {}

    async def aclose(self) -> None:
        self.closed = True

    async def receive(self, max_bytes: int = 65536) -> bytes:
        self.receive_calls += 1
        self.requested_sizes.append(max_bytes)
        if self.chunks:
            return self.chunks.popleft()

        raise EndOfStream


async def test_rate_limiting(fake_clock: FakeClock) -> None:
    """At 10 bytes/s with a burst of 5, cumulative delivery follows the bucket."""
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    await send_stream.send(b"x" * 100)
    limited_stream = RateLimitedByteReceiveStream(
        receive_stream, rate=10, burst_bytes=5
    )

    received = bytearray()

    async def consume() -> None:
        while len(received) < 40:
            received.extend(await limited_stream.receive(65536))

    async with create_task_group() as task_group:
        task_group.start_soon(consume)
        await wait_all_tasks_blocked()

        # The full burst is available immediately
        assert len(received) == 5

        # Exactly one token accumulates per 0.1 seconds
        for step in range(1, 36):
            await fake_clock.advance(0.1)
            assert len(received) == 5 + step

        # Steady state delivers exactly 10 bytes per second
        assert len(received) == 5 + int(fake_clock.time * 10)
        task_group.cancel_scope.cancel()

    assert bytes(received) == b"x" * len(received)
    send_stream.close()
    receive_stream.close()


async def test_delivered_bytes_never_exceed_tokens(fake_clock: FakeClock) -> None:
    """Check the cumulative upper bound at several fractional time points."""
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    await send_stream.send(b"y" * 1000)
    limited_stream = RateLimitedByteReceiveStream(
        receive_stream, rate=10, burst_bytes=5
    )

    received = bytearray()

    async def consume() -> None:
        while True:
            received.extend(await limited_stream.receive(1000))

    time_points = (0.0, 0.1, 0.25, 0.37, 0.9, 1.23, 2.5, 3.33)
    async with create_task_group() as task_group:
        task_group.start_soon(consume)
        await wait_all_tasks_blocked()
        assert len(received) == 5

        for target_time in time_points[1:]:
            while fake_clock.time < target_time - 1e-9:
                await fake_clock.advance(min(0.01, target_time - fake_clock.time))

            assert fake_clock.time == pytest.approx(target_time)
            assert len(received) <= 5 + int(target_time * 10)

        assert len(received) == 5 + int(3.33 * 10)
        task_group.cancel_scope.cancel()

    assert bytes(received) == b"y" * len(received)
    send_stream.close()
    receive_stream.close()


async def test_oversized_chunk_is_buffered(fake_clock: FakeClock) -> None:
    """A chunk larger than the available tokens is split across receive calls."""
    send_stream, receive_stream = create_memory_object_stream[bytes](1)
    await send_stream.send(b"abcdefghij")
    limited_stream = RateLimitedByteReceiveStream(
        receive_stream, rate=10, burst_bytes=5
    )

    assert await limited_stream.receive(65536) == b"abcde"
    assert limited_stream.buffer == b"fghij"

    await fake_clock.advance(0.3)
    assert await limited_stream.receive(65536) == b"fgh"
    assert limited_stream.buffer == b"ij"

    await fake_clock.advance(0.2)
    assert await limited_stream.receive(65536) == b"ij"
    assert limited_stream.buffer == b""
    send_stream.close()
    receive_stream.close()


async def test_oversized_byte_stream_chunk_bounded_buffer(
    fake_clock: FakeClock,
) -> None:
    """The buffer holds the surplus of one chunk and no more chunks are pulled.

    A :class:`~anyio.abc.ByteReceiveStream` is asked for at most the available
    number of tokens, so the buffer can never exceed one chunk's worth of data.
    """
    underlying = ScriptedByteReceiveStream(deque([b"abcdefghijklmnopqrst", b"zzz"]))
    limited_stream = RateLimitedByteReceiveStream(underlying, rate=10, burst_bytes=5)

    assert await limited_stream.receive(65536) == b"abcde"
    assert limited_stream.buffer == b"fghijklmnopqrst"
    assert underlying.receive_calls == 1
    assert underlying.requested_sizes == [5]

    await fake_clock.advance(0.5)
    assert await limited_stream.receive(65536) == b"fghij"
    assert underlying.receive_calls == 1

    await fake_clock.advance(1.0)
    assert await limited_stream.receive(65536) == b"klmno"
    assert underlying.receive_calls == 1

    await fake_clock.advance(1.0)
    assert await limited_stream.receive(65536) == b"pqrst"
    assert underlying.receive_calls == 1
    assert limited_stream.buffer == b""


@dataclass(eq=False)
class GreedyByteReceiveStream(ByteReceiveStream):
    """Always returns its entire payload, ignoring ``max_bytes``."""

    payload: bytes
    read_count: int = 0
    closed: bool = False

    @property
    def extra_attributes(self) -> dict[object, Callable[[], object]]:
        return {}

    async def aclose(self) -> None:
        self.closed = True

    async def receive(self, max_bytes: int = 65536) -> bytes:
        self.read_count += 1
        if self.payload:
            chunk = self.payload
            self.payload = b""
            return chunk

        raise EndOfStream


async def test_buffer_stays_bounded_with_oversized_pull(
    fake_clock: FakeClock,
) -> None:
    """Surplus from an oversized pull stays bounded; no extra reads happen."""
    payload = b"a" * 1000
    underlying = GreedyByteReceiveStream(payload)
    limited_stream = RateLimitedByteReceiveStream(underlying, rate=100, burst_bytes=50)

    first = await limited_stream.receive()
    assert first == b"a" * 50
    assert underlying.read_count == 1

    # While the surplus is drained, no more reads reach the wrapped stream and
    # the buffer shrinks monotonically
    previous_buffer_size = len(limited_stream.buffer)
    assert previous_buffer_size == 950
    for _ in range(5):
        await fake_clock.advance(0.1)
        chunk = await limited_stream.receive()
        assert chunk
        assert len(limited_stream.buffer) < previous_buffer_size
        previous_buffer_size = len(limited_stream.buffer)
        assert underlying.read_count == 1


async def test_eof_after_buffer_drained(fake_clock: FakeClock) -> None:
    """EndOfStream is not signaled until buffered bytes have been delivered."""
    underlying = ScriptedByteReceiveStream(deque([b"abcdefghij"]))
    limited_stream = RateLimitedByteReceiveStream(underlying, rate=10, burst_bytes=5)

    assert await limited_stream.receive() == b"abcde"

    received = bytearray()
    async with create_task_group() as task_group:

        async def consume() -> None:
            with pytest.raises(EndOfStream):
                while True:
                    received.extend(await limited_stream.receive())

        task_group.start_soon(consume)
        await wait_all_tasks_blocked()
        assert bytes(received) == b""

        for _ in range(10):
            await fake_clock.advance(0.1)

        assert bytes(received) == b"fghij"


async def test_wait_cancelled_then_reread(fake_clock: FakeClock) -> None:
    """Cancelling while waiting for tokens neither loses nor duplicates bytes."""
    send_stream, receive_stream = create_memory_object_stream[bytes](1)
    await send_stream.send(b"abcdefghij")
    limited_stream = RateLimitedByteReceiveStream(
        receive_stream, rate=10, burst_bytes=5
    )

    assert await limited_stream.receive() == b"abcde"
    assert limited_stream.buffer == b"fghij"

    async with create_task_group() as task_group:

        async def blocked_receive() -> None:
            await limited_stream.receive()

        task_group.start_soon(blocked_receive)
        await wait_all_tasks_blocked()
        task_group.cancel_scope.cancel()

    # The buffer is untouched by the cancelled wait
    assert limited_stream.buffer == b"fghij"

    await fake_clock.advance(0.5)
    assert await limited_stream.receive() == b"fghij"
    assert limited_stream.buffer == b""
    send_stream.close()
    receive_stream.close()


async def test_immediate_eof() -> None:
    underlying = ScriptedByteReceiveStream(deque())
    limited_stream = RateLimitedByteReceiveStream(underlying, rate=10, burst_bytes=5)
    with pytest.raises(EndOfStream):
        await limited_stream.receive()


async def test_close() -> None:
    underlying = ScriptedByteReceiveStream(deque())
    limited_stream = RateLimitedByteReceiveStream(underlying, rate=10, burst_bytes=5)
    await limited_stream.aclose()
    assert underlying.closed
    with pytest.raises(ClosedResourceError):
        await limited_stream.receive()


@pytest.mark.parametrize(
    "rate,burst_bytes",
    [(0, 5), (-1, 5), (10, 0), (10, -1), (float("inf"), 5), (10, float("nan"))],
)
async def test_invalid_parameters(rate: float, burst_bytes: float) -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](1)
    with pytest.raises(ValueError):
        RateLimitedByteReceiveStream(receive_stream, rate=rate, burst_bytes=burst_bytes)

    send_stream.close()
    receive_stream.close()


@pytest.mark.parametrize("max_bytes", [0, -1])
async def test_receive_invalid_max_bytes(max_bytes: int) -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](1)
    limited_stream = RateLimitedByteReceiveStream(
        receive_stream, rate=10, burst_bytes=5
    )
    with pytest.raises(ValueError, match="max_bytes must be a positive integer"):
        await limited_stream.receive(max_bytes)

    send_stream.close()
    receive_stream.close()


async def test_full_stream_byte_accounting() -> None:
    """Every source byte is delivered exactly once across several large chunks."""
    chunks = deque([b"a" * 50, b"b" * 30, b"c" * 20])
    underlying = ScriptedByteReceiveStream(chunks)
    limited_stream = RateLimitedByteReceiveStream(
        underlying, rate=1000, burst_bytes=1000
    )

    received = bytearray()
    with pytest.raises(EndOfStream):
        while True:
            chunk = await limited_stream.receive(100)
            received.extend(chunk)
            assert len(chunk) <= 100

    assert len(received) == 100
    assert bytes(received) == b"a" * 50 + b"b" * 30 + b"c" * 20
