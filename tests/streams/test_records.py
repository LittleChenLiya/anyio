from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import pytest

from anyio import (
    BusyResourceError,
    ClosedResourceError,
    DelimiterNotFound,
    EndOfStream,
    Event,
    IncompleteRecord,
    create_task_group,
    move_on_after,
    wait_all_tasks_blocked,
)
from anyio.abc import ByteReceiveStream
from anyio.lowlevel import checkpoint
from anyio.streams.buffered import BufferedByteReceiveStream
from anyio.streams.records import (
    RecordReceiveStream,
    RecordStreamState,
    RecordStreamStateError,
    RecordTooLong,
)


class ChunkedReceiveStream(ByteReceiveStream):
    """
    A receive stream that yields preset chunks one at a time.

    Any Event in the item list is waited on before the chunks after it are
    delivered.
    """

    def __init__(self, items: Iterable[bytes | Event]) -> None:
        self._items: deque[bytes | Event] = deque(items)
        self._closed = False

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self._closed:
            raise ClosedResourceError

        await checkpoint()
        while self._items and isinstance(self._items[0], Event):
            event = self._items.popleft()
            assert isinstance(event, Event)
            await event.wait()
            await checkpoint()

        if not self._items:
            raise EndOfStream

        item = self._items.popleft()
        assert isinstance(item, bytes)
        return item

    async def aclose(self) -> None:
        self._closed = True


def assert_state(reader: RecordReceiveStream, state: RecordStreamState) -> None:
    assert reader.state is state


async def test_read_records_crlf_chunked() -> None:
    stream = ChunkedReceiveStream([b"a\r", b"\n b\r\n"])
    reader = RecordReceiveStream(stream, b"\r\n", 100)
    assert_state(reader, RecordStreamState.READY)
    assert await reader.read_until() == b"a"
    assert await reader.read_until() == b" b"
    assert reader.buffer == b""
    with pytest.raises(EndOfStream):
        await reader.read_until()


async def test_cancel_before_second_chunk() -> None:
    gate = Event()
    stream = ChunkedReceiveStream([b"a\r", gate, b"\n b\r\n"])
    reader = RecordReceiveStream(stream, b"\r\n", 100)

    # The read is cancelled while waiting for the second chunk
    with move_on_after(0.5):
        await reader.read_until()

    # No bytes may be lost: the partial record is still in the buffer
    assert reader.buffer == b"a\r"

    gate.set()
    assert await reader.read_until() == b"a"
    assert await reader.read_until() == b" b"
    assert reader.buffer == b""


async def test_include_delimiter() -> None:
    stream = ChunkedReceiveStream([b"abc\r\ndef\r\n"])
    reader = RecordReceiveStream(stream, b"\r\n", 100)
    assert await reader.read_until(include_delimiter=True) == b"abc\r\n"
    assert await reader.read_until() == b"def"


async def test_self_overlapping_delimiter() -> None:
    stream = ChunkedReceiveStream([b"a###b##"])
    reader = RecordReceiveStream(stream, b"##", 100)
    assert await reader.read_until() == b"a"
    assert reader.buffer == b"#b##"
    assert await reader.read_until() == b"#b"
    assert reader.buffer == b""


async def test_exact_size_limit() -> None:
    stream = ChunkedReceiveStream([b"abc\r\n"])
    reader = RecordReceiveStream(stream, b"\r\n", 3)
    assert await reader.read_until() == b"abc"
    assert_state(reader, RecordStreamState.READY)


async def test_record_too_long_requires_discard() -> None:
    stream = ChunkedReceiveStream([b"xxxxxxxxxx", b"\r\n", b"ok\r\n"])
    reader = RecordReceiveStream(stream, b"\r\n", 4)
    with pytest.raises(RecordTooLong):
        await reader.read_until()

    assert_state(reader, RecordStreamState.NEEDS_DISCARD)

    # Reading is refused while the oversized record has not been discarded
    with pytest.raises(RecordStreamStateError):
        await reader.read_until()

    # 10 record bytes + 2 delimiter bytes are discarded
    assert await reader.discard_until() == 12
    assert_state(reader, RecordStreamState.READY)
    assert await reader.read_until() == b"ok"


async def test_record_too_long_is_delimiter_not_found() -> None:
    stream = ChunkedReceiveStream([b"xxxxxxxxxx"])
    reader = RecordReceiveStream(stream, b"\r\n", 4)
    with pytest.raises(DelimiterNotFound):
        await reader.read_until()


async def test_budgeted_discard_two_passes() -> None:
    stream = ChunkedReceiveStream([b"xxxxxxxxxx", b"\r\n", b"ok\r\n"])
    reader = RecordReceiveStream(stream, b"\r\n", 4)
    with pytest.raises(RecordTooLong):
        await reader.read_until()

    assert await reader.discard_until(max_bytes=6) == 6
    assert_state(reader, RecordStreamState.NEEDS_DISCARD)

    # The discarded bytes are gone; the rest of the record remains buffered
    assert reader.buffer == b"xxxx"

    assert await reader.discard_until(max_bytes=6) == 6
    assert_state(reader, RecordStreamState.READY)

    # The record following the oversized one is read completely
    assert await reader.read_until() == b"ok"


async def test_cancelled_discard_can_resume() -> None:
    gate = Event()
    stream = ChunkedReceiveStream([b"xxxxxxxxxx", gate, b"\r\n", b"ok\r\n"])
    reader = RecordReceiveStream(stream, b"\r\n", 4)
    with pytest.raises(RecordTooLong):
        await reader.read_until()

    assert await reader.discard_until(max_bytes=6) == 6
    assert reader.buffer == b"xxxx"

    # This discard gets cancelled while waiting for more data
    with move_on_after(0.5):
        await reader.discard_until()

    # Already discarded bytes must not be re-exposed
    assert_state(reader, RecordStreamState.NEEDS_DISCARD)
    assert reader.buffer == b"x"

    gate.set()
    assert await reader.discard_until() == 3
    assert_state(reader, RecordStreamState.READY)
    assert await reader.read_until() == b"ok"


async def test_eof_policy_raise() -> None:
    stream = ChunkedReceiveStream([b"abc"])
    reader = RecordReceiveStream(stream, b"\r\n", 100, eof_policy="raise")
    with pytest.raises(IncompleteRecord) as excinfo:
        await reader.read_until()

    assert excinfo.value.partial == b"abc"
    assert reader.buffer == b"abc"


async def test_eof_policy_return() -> None:
    stream = ChunkedReceiveStream([b"abc"])
    reader = RecordReceiveStream(stream, b"\r\n", 100, eof_policy="return")
    assert await reader.read_until() == b"abc"
    assert reader.buffer == b""
    with pytest.raises(EndOfStream):
        await reader.read_until()


async def test_eof_with_empty_buffer() -> None:
    stream = ChunkedReceiveStream([])
    reader = RecordReceiveStream(stream, b"\r\n", 100)
    with pytest.raises(EndOfStream):
        await reader.read_until()


async def test_eof_during_discard() -> None:
    stream = ChunkedReceiveStream([b"xxxxxxxxxx"])
    reader = RecordReceiveStream(stream, b"\r\n", 4)
    with pytest.raises(RecordTooLong):
        await reader.read_until()

    # The unterminated oversized record is discarded in full at EOF
    assert await reader.discard_until() == 10
    assert_state(reader, RecordStreamState.READY)
    with pytest.raises(EndOfStream):
        await reader.read_until()


async def test_close() -> None:
    stream = ChunkedReceiveStream([b"abc\r\n"])
    reader = RecordReceiveStream(stream, b"\r\n", 100)
    await reader.aclose()
    with pytest.raises(ClosedResourceError):
        await reader.read_until()

    with pytest.raises(ClosedResourceError):
        await reader.discard_until()


async def test_concurrent_reads_forbidden() -> None:
    gate = Event()
    stream = ChunkedReceiveStream([gate, b"abc\r\n"])
    reader = RecordReceiveStream(stream, b"\r\n", 100)
    async with create_task_group() as tg:
        tg.start_soon(reader.read_until)
        await wait_all_tasks_blocked()
        with pytest.raises(BusyResourceError):
            await reader.read_until()

        gate.set()


async def test_concurrent_read_and_discard_forbidden() -> None:
    gate = Event()
    stream = ChunkedReceiveStream([b"xxxxxxxxxx", gate, b"\r\n"])
    reader = RecordReceiveStream(stream, b"\r\n", 4)
    with pytest.raises(RecordTooLong):
        await reader.read_until()

    async with create_task_group() as tg:
        tg.start_soon(reader.discard_until)
        await wait_all_tasks_blocked()
        with pytest.raises(BusyResourceError):
            await reader.read_until()

        gate.set()


async def test_discard_in_ready_state_refused() -> None:
    stream = ChunkedReceiveStream([b"abc\r\n"])
    reader = RecordReceiveStream(stream, b"\r\n", 100)
    with pytest.raises(RecordStreamStateError):
        await reader.discard_until()


async def test_wraps_existing_buffered_stream() -> None:
    stream = ChunkedReceiveStream([b"abc\r\n"])
    buffered = BufferedByteReceiveStream(stream)
    reader = RecordReceiveStream(buffered, b"\r\n", 100)
    reader.feed_data(b"x")
    assert await reader.read_until() == b"xabc"


def test_invalid_constructor_arguments() -> None:
    stream = ChunkedReceiveStream([])
    with pytest.raises(ValueError, match="delimiter must not be empty"):
        RecordReceiveStream(stream, b"", 100)

    with pytest.raises(ValueError, match="max_record_bytes must be a positive integer"):
        RecordReceiveStream(stream, b"\r\n", 0)

    with pytest.raises(ValueError, match="eof_policy"):
        RecordReceiveStream(stream, b"\r\n", 100, eof_policy="ignore")  # type: ignore[arg-type]
