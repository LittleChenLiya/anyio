from __future__ import annotations

import pytest

from anyio import (
    BusyResourceError,
    ClosedResourceError,
    DelimiterNotFound,
    IncompleteRecord,
    RecordStateError,
    create_memory_object_stream,
    create_task_group,
    move_on_after,
    wait_all_tasks_blocked,
)
from anyio.streams.buffered import RecordReceiveStream


async def test_read_until_chunked_crlf() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)
    await send_stream.send(b"a\r")
    await send_stream.send(b"\n b\r\n")

    assert await stream.read_until(b"\r\n", 100) == b"a"
    assert await stream.read_until(b"\r\n", 100) == b" b"
    assert stream.buffer == b""
    assert stream.state == "ready"

    send_stream.close()
    receive_stream.close()


async def test_read_until_cancelled_mid_record() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)
    await send_stream.send(b"a\r")

    # Cancel the read while it is waiting for the second chunk
    with move_on_after(0.1) as scope:
        await stream.read_until(b"\r\n", 100)

    assert scope.cancelled_caught
    # No bytes may be lost by the cancellation
    assert stream.buffer == b"a\r"
    assert stream.state == "ready"

    await send_stream.send(b"\n b\r\n")
    assert await stream.read_until(b"\r\n", 100) == b"a"
    assert await stream.read_until(b"\r\n", 100) == b" b"

    send_stream.close()
    receive_stream.close()


async def test_read_until_include_delimiter() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)
    await send_stream.send(b"a\r\nb\r\n")

    record = await stream.read_until(b"\r\n", 100, include_delimiter=True)
    assert record == b"a\r\n"
    record = await stream.read_until(b"\r\n", 100)
    assert record == b"b"

    send_stream.close()
    receive_stream.close()


async def test_read_until_self_overlapping_delimiter() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)

    # "aab" overlaps itself and straddles the chunk boundary here
    await send_stream.send(b"aa")
    await send_stream.send(b"ab")
    assert await stream.read_until(b"aab", 100) == b"a"

    # The delimiter arrives split across two chunks
    await send_stream.send(b"xxab")
    await send_stream.send(b"c")
    assert await stream.read_until(b"abc", 100) == b"xx"

    send_stream.close()
    receive_stream.close()


async def test_read_until_exact_limit() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)

    # A record of exactly max_bytes bytes is accepted, even when the
    # delimiter arrives in a later chunk
    await send_stream.send(b"ab")
    await send_stream.send(b"\r\n")
    assert await stream.read_until(b"\r\n", 2) == b"ab"
    assert stream.state == "ready"

    # One byte more exceeds the limit
    await send_stream.send(b"abc\r\n")
    with pytest.raises(DelimiterNotFound):
        await stream.read_until(b"\r\n", 2)

    assert stream.state == "needs_discard"
    assert await stream.discard_until(b"\r\n") == 5
    assert stream.state == "ready"

    send_stream.close()
    receive_stream.close()


async def test_read_until_oversized_requires_discard() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)
    await send_stream.send(b"xxxxxx\r\n")

    with pytest.raises(DelimiterNotFound):
        await stream.read_until(b"\r\n", 3)

    assert stream.state == "needs_discard"

    # Reads are rejected while the oversized record is pending discard
    with pytest.raises(RecordStateError):
        await stream.read_until(b"\r\n", 3)

    # The rest of the oversized record must not be mistaken for a new record
    assert await stream.discard_until(b"\r\n") == 8
    assert stream.state == "ready"
    assert stream.buffer == b""

    send_stream.close()
    receive_stream.close()


async def test_discard_until_budgeted() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)
    await send_stream.send(b"aaaaaaaa\r\nok\r\n")

    with pytest.raises(DelimiterNotFound):
        await stream.read_until(b"\r\n", 4)

    # Discard the oversized record in two limited-budget steps
    assert await stream.discard_until(b"\r\n", 5) == 5
    assert stream.state == "needs_discard"
    assert await stream.discard_until(b"\r\n", 5) == 5
    assert stream.state == "ready"

    # The following record must be read back intact
    assert await stream.read_until(b"\r\n", 100) == b"ok"

    send_stream.close()
    receive_stream.close()


async def test_discard_until_cancelled_and_resumed() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)
    await send_stream.send(b"xxxxxx")

    with pytest.raises(DelimiterNotFound):
        await stream.read_until(b"\r\n", 3)

    # Cancel the discard while it is waiting for more data
    with move_on_after(0.1) as scope:
        await stream.discard_until(b"\r\n")

    assert scope.cancelled_caught
    assert stream.state == "needs_discard"
    # Five of the six bytes have already been discarded; the last one is kept
    # because it could be part of a delimiter straddling the boundary
    assert stream.buffer == b"x"

    # Resuming the discard must not re-expose the already discarded bytes
    await send_stream.send(b"yy\r\nok\r\n")
    assert await stream.discard_until(b"\r\n") == 5
    assert stream.state == "ready"
    assert await stream.read_until(b"\r\n", 100) == b"ok"

    send_stream.close()
    receive_stream.close()


async def test_discard_until_when_ready() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)
    assert stream.state == "ready"
    assert await stream.discard_until(b"\r\n") == 0

    send_stream.close()
    receive_stream.close()


async def test_read_until_eof_raise() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)
    await send_stream.send(b"abc")
    await send_stream.aclose()

    with pytest.raises(IncompleteRecord) as exc_info:
        await stream.read_until(b"\r\n", 100)

    assert exc_info.value.partial == b"abc"
    # The partial record is left in the buffer
    assert stream.buffer == b"abc"

    receive_stream.close()


async def test_read_until_eof_return() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)
    await send_stream.send(b"abc")
    await send_stream.aclose()

    record = await stream.read_until(b"\r\n", 100, on_eof="return")
    assert record == b"abc"
    assert stream.buffer == b""

    receive_stream.close()


async def test_discard_until_eof() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)
    await send_stream.send(b"xxxxxx")

    with pytest.raises(DelimiterNotFound):
        await stream.read_until(b"\r\n", 3)

    await send_stream.aclose()
    with pytest.raises(IncompleteRecord) as exc_info:
        await stream.discard_until(b"\r\n")

    assert exc_info.value.partial == b"x"
    assert stream.state == "needs_discard"

    receive_stream.close()


async def test_concurrent_read_operations() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)

    async with create_task_group() as task_group:
        task_group.start_soon(stream.read_until, b"\r\n", 100)
        await wait_all_tasks_blocked()
        with pytest.raises(BusyResourceError):
            await stream.read_until(b"\r\n", 100)

        with pytest.raises(BusyResourceError):
            await stream.discard_until(b"\r\n")

        task_group.cancel_scope.cancel()

    send_stream.close()
    receive_stream.close()


async def test_closed_stream() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)
    await send_stream.send(b"xxxxxx\r\n")

    with pytest.raises(DelimiterNotFound):
        await stream.read_until(b"\r\n", 3)

    await stream.aclose()
    with pytest.raises(ClosedResourceError):
        await stream.read_until(b"\r\n", 3)

    with pytest.raises(ClosedResourceError):
        await stream.discard_until(b"\r\n")

    send_stream.close()


async def test_invalid_arguments() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)

    with pytest.raises(ValueError, match="delimiter must not be empty"):
        await stream.read_until(b"", 100)

    with pytest.raises(ValueError, match="max_bytes must be a non-negative"):
        await stream.read_until(b"\r\n", -1)

    with pytest.raises(ValueError, match="on_eof must be either"):
        await stream.read_until(b"\r\n", 100, on_eof="bogus")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="delimiter must not be empty"):
        await stream.discard_until(b"")

    with pytest.raises(ValueError, match="max_bytes must be a positive"):
        await stream.discard_until(b"\r\n", 0)

    send_stream.close()
    receive_stream.close()


async def test_buffered_api_still_works() -> None:
    send_stream, receive_stream = create_memory_object_stream[bytes](10)
    stream = RecordReceiveStream(receive_stream)
    await send_stream.send(b"abcdefgh")

    assert await stream.receive_exactly(2) == b"ab"
    assert await stream.receive_until(b"de", 10) == b"c"
    assert await stream.receive(2) == b"fg"
    assert await stream.receive(2) == b"h"

    send_stream.close()
    receive_stream.close()
