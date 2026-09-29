from __future__ import annotations

__all__ = ("RateLimitedByteReceiveStream",)

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .. import ClosedResourceError, current_time, sleep
from ..abc import AnyByteReceiveStream, ByteReceiveStream

# Tolerance for floating point inaccuracy when comparing the allowance against
# whole bytes; without it, a minuscule float error could require a wait shorter
# than the resolution of the monotonic clock, causing an infinite loop
_ALLOWANCE_EPSILON = 1e-9


@dataclass(eq=False)
class RateLimitedByteReceiveStream(ByteReceiveStream):
    """
    Wraps any bytes-based receive stream and limits the rate at which bytes are
    delivered to the caller.

    The rate limit is enforced with a token bucket algorithm: the allowance is
    replenished at `bytes_per_second` (measured with a monotonic clock), up to a
    maximum of `burst_size` bytes. Each :meth:`receive` call waits until the
    allowance permits at least one more byte, and never delivers more bytes than
    the current allowance. If the wrapped stream delivers more bytes than the
    allowance permits, the surplus is kept in a bounded internal buffer and
    delivered by subsequent :meth:`receive` calls.

    :param receive_stream: the stream to be wrapped
    :param bytes_per_second: the maximum sustained rate, in bytes per second, at
        which bytes are delivered (must be a positive number)
    :param burst_size: the maximum number of bytes that can be delivered at once
        after a period of inactivity (must be a positive integer)
    """

    receive_stream: AnyByteReceiveStream
    bytes_per_second: float
    burst_size: int
    _buffer: bytearray = field(init=False, default_factory=bytearray)
    _allowance: float = field(init=False)
    _last_check: float | None = field(init=False, default=None)
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if not self.bytes_per_second > 0:
            raise ValueError("bytes_per_second must be a positive number")

        if self.burst_size < 1:
            raise ValueError("burst_size must be a positive integer")

        self._allowance = float(self.burst_size)

    async def aclose(self) -> None:
        await self.receive_stream.aclose()
        self._closed = True

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return self.receive_stream.extra_attributes

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")

        if self._closed:
            raise ClosedResourceError

        # Replenish the allowance based on the elapsed monotonic time, then wait
        # until at least one byte's worth of allowance is available
        while True:
            now = current_time()
            if self._last_check is not None:
                elapsed = now - self._last_check
                self._allowance = min(
                    float(self.burst_size),
                    self._allowance + elapsed * self.bytes_per_second,
                )

            self._last_check = now
            if self._allowance >= 1 - _ALLOWANCE_EPSILON:
                break

            await sleep((1 - self._allowance) / self.bytes_per_second)

        max_allowed = min(int(self._allowance + _ALLOWANCE_EPSILON), max_bytes)
        if self._buffer:
            chunk = bytes(self._buffer[:max_allowed])
            del self._buffer[:max_allowed]
        elif isinstance(self.receive_stream, ByteReceiveStream):
            chunk = await self.receive_stream.receive(max_allowed)
        else:
            # With a bytes-oriented object stream, we need to handle any surplus
            # bytes we get from the receive() call
            chunk = await self.receive_stream.receive()
            if len(chunk) > max_allowed:
                # Save the surplus bytes in the buffer
                self._buffer.extend(chunk[max_allowed:])
                chunk = chunk[:max_allowed]

        self._allowance = max(0.0, self._allowance - len(chunk))
        return chunk
