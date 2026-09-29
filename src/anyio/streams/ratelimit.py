from __future__ import annotations

__all__ = ("RateLimitedByteReceiveStream",)

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .. import ClosedResourceError
from .._core import _eventloop
from ..abc import AnyByteReceiveStream, ByteReceiveStream


@dataclass(eq=False)
class RateLimitedByteReceiveStream(ByteReceiveStream):
    """
    Wraps a bytes-based receive stream and limits the rate at which bytes are
    delivered to the caller.

    Rate limiting is implemented as a token bucket: a token, representing one
    byte, accumulates at the given ``rate`` up to a maximum bucket size of
    ``burst_bytes``. A :meth:`receive` call never returns more bytes than the
    current number of whole tokens, waiting until at least one token is
    available if the bucket is empty. If the wrapped stream yields more bytes
    than can be delivered at the moment, the surplus is stored in a bounded
    buffer and served by subsequent receive calls instead of reading more data
    from the wrapped stream.

    The bucket is measured against the event loop's monotonic clock, so the
    wrapper behaves identically on all supported backends.

    :param receive_stream: the bytes-based receive stream to wrap
    :param rate: the sustained delivery rate, in bytes per second (must be > 0)
    :param burst_bytes: the maximum amount of bytes that may be delivered in a
        burst, i.e. the size of the token bucket (must be > 0)

    """

    receive_stream: AnyByteReceiveStream
    rate: float
    burst_bytes: float
    _tokens: float = field(init=False)
    _last_refill: float | None = field(init=False, default=None)
    _buffer: bytearray = field(init=False, default_factory=bytearray)
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.rate, int | float)
            or not math.isfinite(self.rate)
            or self.rate <= 0
        ):
            raise ValueError("rate must be a finite, positive number")

        if (
            not isinstance(self.burst_bytes, int | float)
            or not math.isfinite(self.burst_bytes)
            or self.burst_bytes <= 0
        ):
            raise ValueError("burst_bytes must be a finite, positive number")

        self.rate = float(self.rate)
        self.burst_bytes = float(self.burst_bytes)
        self._tokens = self.burst_bytes

    @property
    def buffer(self) -> bytes:
        """The bytes currently buffered from the wrapped stream."""
        return bytes(self._buffer)

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return self.receive_stream.extra_attributes

    async def aclose(self) -> None:
        await self.receive_stream.aclose()
        self._closed = True

    def _refill(self) -> float:
        """Lazily refill the bucket according to the monotonic clock."""
        now = _eventloop.current_time()
        if self._last_refill is None:
            self._last_refill = now
        elif now > self._last_refill:
            self._tokens = min(
                self.burst_bytes,
                self._tokens + (now - self._last_refill) * self.rate,
            )
            self._last_refill = now

        return now

    def _whole_tokens(self) -> int:
        """Return the number of currently available whole tokens.

        A tiny tolerance is applied before flooring so that floating point
        rounding at exact time boundaries does not cost a whole token.
        """
        return math.floor(self._tokens + 1e-9 * max(1.0, abs(self._tokens)))

    def _consume(self, count: int) -> None:
        self._tokens = max(0.0, self._tokens - count)

    async def _wait_for_whole_token(self, now: float) -> None:
        """Sleep until the next whole token becomes available."""
        needed = math.floor(self._tokens) + 1 - self._tokens
        self._last_refill = now
        await _eventloop.sleep_until(now + needed / self.rate)

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")

        if self._closed:
            raise ClosedResourceError

        while True:
            now = self._refill()
            available = self._whole_tokens()

            # Serve buffered bytes first so that nothing is pulled from the
            # wrapped stream while surplus bytes are still waiting
            if self._buffer:
                if available < 1:
                    await self._wait_for_whole_token(now)
                    continue

                deliverable = min(len(self._buffer), max_bytes, available)
                chunk = bytes(self._buffer[:deliverable])
                del self._buffer[:deliverable]
                self._consume(deliverable)
                return chunk

            if available < 1:
                await self._wait_for_whole_token(now)
                continue

            request_size = min(max_bytes, available)
            if isinstance(self.receive_stream, ByteReceiveStream):
                chunk = await self.receive_stream.receive(request_size)
            else:
                chunk = await self.receive_stream.receive()

            if not chunk:
                return b""

            # The wrapped read may take real time, so refill before accounting
            self._refill()
            available = self._whole_tokens()
            deliverable = min(len(chunk), max_bytes, available)
            if len(chunk) > deliverable:
                self._buffer.extend(chunk[deliverable:])

            self._consume(deliverable)
            return chunk[:deliverable]
