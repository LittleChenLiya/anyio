# ruff: noqa: T201
"""
Demonstration of record-oriented reading with RecordReceiveStream.

Covers normal CRLF framing over chunked delivery, cancellation recovery,
oversized-record discard with a byte budget, and resuming a cancelled discard.
Runs on both asyncio and Trio (well under 90 seconds).
"""

from __future__ import annotations

from anyio import (
    DelimiterNotFound,
    create_memory_object_stream,
    move_on_after,
    run,
)
from anyio.streams.buffered import RecordReceiveStream

CRLF = b"\r\n"


async def demo_normal_framing() -> None:
    print("=== Normal framing (chunked CRLF delivery) ===")
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)

    # The peer sends "a\r" and "\n b\r\n" in two separate chunks
    await send_stream.send(b"a\r")
    await send_stream.send(b"\n b\r\n")

    record = await stream.read_until(CRLF, 100)
    print(f"record 1: {record!r}")
    record = await stream.read_until(CRLF, 100)
    print(f"record 2: {record!r}")
    await send_stream.send(CRLF)
    record = await stream.read_until(CRLF, 100, include_delimiter=True)
    print(f"record 3 (empty record, delimiter included): {record!r}")

    send_stream.close()
    receive_stream.close()


async def demo_cancel_recovery() -> None:
    print("=== Cancellation recovery ===")
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)
    await send_stream.send(b"a\r")

    # The read is cancelled while waiting for the second chunk
    with move_on_after(0.1) as scope:
        await stream.read_until(CRLF, 100)

    print(f"cancelled: {scope.cancelled_caught}, buffer preserved: {stream.buffer!r}")

    # Retrying the read does not lose a single byte
    await send_stream.send(b"\n b\r\n")
    record = await stream.read_until(CRLF, 100)
    print(f"record 1 after retry: {record!r}")
    record = await stream.read_until(CRLF, 100)
    print(f"record 2 after retry: {record!r}")

    send_stream.close()
    receive_stream.close()


async def demo_oversized_discard() -> None:
    print("=== Oversized record: budgeted discard and recovery ===")
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)
    await send_stream.send(b"aaaaaaaa\r\nok\r\n")

    try:
        await stream.read_until(CRLF, 4)
    except DelimiterNotFound:
        print(f"record exceeded the limit, state: {stream.state}")

    # Discard the oversized record in two budgeted steps
    discarded = await stream.discard_until(CRLF, 5)
    print(f"discarded {discarded} bytes, state: {stream.state}")
    discarded += await stream.discard_until(CRLF, 5)
    print(f"discarded {discarded} bytes total, state: {stream.state}")

    # The following record is read back intact
    record = await stream.read_until(CRLF, 100)
    print(f"next record: {record!r}")

    send_stream.close()
    receive_stream.close()


async def demo_cancelled_discard() -> None:
    print("=== Cancelled discard can be resumed ===")
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)
    await send_stream.send(b"xxxxxx")

    try:
        await stream.read_until(CRLF, 3)
    except DelimiterNotFound:
        print(f"record exceeded the limit, state: {stream.state}")

    with move_on_after(0.1) as scope:
        await stream.discard_until(CRLF)

    print(f"discard cancelled: {scope.cancelled_caught}, buffer: {stream.buffer!r}")

    await send_stream.send(b"yy\r\nok\r\n")
    discarded = await stream.discard_until(CRLF)
    print(f"resumed discard dropped {discarded} more bytes, state: {stream.state}")
    record = await stream.read_until(CRLF, 100)
    print(f"next record: {record!r}")

    send_stream.close()
    receive_stream.close()


async def main() -> None:
    await demo_normal_framing()
    await demo_cancel_recovery()
    await demo_oversized_discard()
    await demo_cancelled_discard()


if __name__ == "__main__":
    for backend in ("asyncio", "trio"):
        print(f"--- backend: {backend} ---")
        run(main, backend=backend)
        print()
