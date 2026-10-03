"""Demonstrates explicit half-close with HalfCloseableStapledByteStream.

Two independent byte streams (one for sending, one for receiving) are stapled
together. After sending "hello" and half-closing the send side with
``send_eof()``, the receive side is still able to read the peer's "reply".
The state of both ends is printed at each step.
"""

from __future__ import annotations

from anyio import ClosedResourceError, run
from anyio.abc import ByteReceiveStream, ByteSendStream
from anyio.streams.stapled import HalfCloseableStapledByteStream


class DemoByteSendStream(ByteSendStream):
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.closed = False

    async def send(self, item: bytes) -> None:
        if self.closed:
            raise ClosedResourceError

        self.buffer.extend(item)

    async def aclose(self) -> None:
        self.closed = True


class DemoByteReceiveStream(ByteReceiveStream):
    def __init__(self, data: bytes) -> None:
        self.buffer = bytearray(data)
        self.closed = False

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self.closed:
            raise ClosedResourceError

        data = bytes(self.buffer[:max_bytes])
        del self.buffer[:max_bytes]
        return data

    async def aclose(self) -> None:
        self.closed = True


def print_state(stream: HalfCloseableStapledByteStream, label: str) -> None:
    print(
        f"{label}: send_closed={stream.send_closed}, "
        f"receive_closed={stream.receive_closed}"
    )


async def main() -> None:
    send_stream = DemoByteSendStream()
    receive_stream = DemoByteReceiveStream(b"reply")
    stream = HalfCloseableStapledByteStream(send_stream, receive_stream)
    print_state(stream, "initial")

    await stream.send(b"hello")
    print(f"sent: {bytes(send_stream.buffer)!r}")

    # Half-close: only the send side is closed
    await stream.send_eof()
    print_state(stream, "after send_eof")

    # The receive side is still open and delivers the peer's reply
    reply = await stream.receive()
    print(f"received after send_eof: {reply!r}")

    # Sending is no longer possible
    try:
        await stream.send(b"more")
    except ClosedResourceError:
        print("sending after send_eof: ClosedResourceError (as expected)")

    # Full close shuts down both ends (idempotently)
    await stream.aclose()
    await stream.aclose()
    print_state(stream, "after aclose")


if __name__ == "__main__":
    run(main)
