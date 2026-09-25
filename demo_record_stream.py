"""
Demonstration of the record reading session in anyio.streams.records.

Runs well within 90 seconds and shows:
1. normal record framing over chunked CRLF-delimited delivery
2. cancellation recovery (a cancelled read loses no bytes)
3. oversized record handling: budgeted discard, cancellation mid-discard and
   resumption without re-exposing discarded bytes
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import anyio
from anyio import EndOfStream, Event, move_on_after
from anyio.abc import ByteReceiveStream
from anyio.lowlevel import checkpoint
from anyio.streams.records import (
    RecordReceiveStream,
    RecordStreamState,
    RecordTooLong,
)


class ChunkedReceiveStream(ByteReceiveStream):
    """Yields preset chunks one at a time, waiting on any Events in between."""

    def __init__(self, items: Iterable[bytes | Event]) -> None:
        self._items: deque[bytes | Event] = deque(items)

    async def receive(self, max_bytes: int = 65536) -> bytes:
        await checkpoint()
        while self._items and isinstance(self._items[0], Event):
            event = self._items.popleft()
            await event.wait()
            await checkpoint()

        if not self._items:
            raise EndOfStream

        item = self._items.popleft()
        assert isinstance(item, bytes)
        return item

    async def aclose(self) -> None:
        pass


async def demo_normal_framing() -> None:
    print("--- normal framing (chunks: 'a\\r' + '\\n b\\r\\n', delimiter CRLF) ---")
    reader = RecordReceiveStream(
        ChunkedReceiveStream([b"a\r", b"\n b\r\n"]), b"\r\n", 100
    )
    print("record 1:", await reader.read_until())
    print("record 2:", await reader.read_until())


async def demo_cancel_recovery() -> None:
    print("--- cancel recovery (read cancelled before the second chunk) ---")
    gate = Event()
    reader = RecordReceiveStream(
        ChunkedReceiveStream([b"a\r", gate, b"\n b\r\n"]), b"\r\n", 100
    )
    with move_on_after(0.2):
        await reader.read_until()

    print("cancelled; residual buffer:", reader.buffer)
    gate.set()
    print("record 1 after resume:", await reader.read_until())
    print("record 2 after resume:", await reader.read_until())


async def demo_oversized_record() -> None:
    print("--- oversized record: budgeted discard, cancel and resume ---")
    gate = Event()
    reader = RecordReceiveStream(
        ChunkedReceiveStream([b"x" * 10, gate, b"\r\n", b"ok\r\n"]), b"\r\n", 4
    )
    try:
        await reader.read_until()
    except RecordTooLong as exc:
        print("read refused:", exc)

    print("state:", reader.state.value)
    print("discarded (budget 6):", await reader.discard_until(max_bytes=6))
    print("buffer now:", reader.buffer)
    with move_on_after(0.2):
        await reader.discard_until()

    print("discard cancelled; buffer:", reader.buffer)
    gate.set()
    print("discarded after resume:", await reader.discard_until())
    print("state:", reader.state.value)
    assert reader.state is RecordStreamState.READY
    print("next record:", await reader.read_until())


async def main() -> None:
    await demo_normal_framing()
    await demo_cancel_recovery()
    await demo_oversized_record()


if __name__ == "__main__":
    for backend in ("asyncio", "trio"):
        print(f"=== backend: {backend} ===")
        anyio.run(main, backend=backend)

    print("DEMO OK")
