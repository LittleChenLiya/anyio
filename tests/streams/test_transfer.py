from __future__ import annotations

import hashlib
import re
import sys
from contextlib import contextmanager
from functools import partial

import pytest

from anyio import (
    BrokenResourceError,
    EndOfStream,
    Event,
    TaskHandle,
    create_memory_object_stream,
    create_task_group,
)
from anyio.abc import ByteReceiveStream, TransactionalByteSink
from anyio.streams.transfer import transfer_stream

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup


@contextmanager
def raises_grouped(exc_type: type, match: str | None = None):
    """
    Assert that a :class:`BaseExceptionGroup` containing a matching exception is
    raised from a task group.
    """
    with pytest.raises(BaseExceptionGroup) as exc_info:
        yield

    matches = [
        e
        for e in exc_info.value.exceptions
        if isinstance(e, exc_type) and (match is None or re.search(match, str(e)))
    ]
    assert matches, f"no {exc_type.__name__} matching {match!r} in {exc_info.value!r}"


class BarrierByteSource(ByteReceiveStream):
    """
    A byte source that blocks on a barrier before delivering each chunk.

    Each chunk has its own (ready, release) event pair. The test waits for
    :meth:`wait_until_ready` to return and then calls :meth:`release` to deliver
    the chunk.
    """

    def __init__(self, *chunks: bytes) -> None:
        self._chunks = list(chunks)
        self._index = 0
        self._barriers: list[tuple[Event, Event]] = [(Event(), Event()) for _ in chunks]
        self.closed = False

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self._index >= len(self._chunks):
            raise EndOfStream

        ready, release = self._barriers[self._index]
        ready.set()
        await release.wait()
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    async def wait_until_ready(self, chunk_index: int) -> None:
        await self._barriers[chunk_index][0].wait()

    def release(self, chunk_index: int) -> None:
        self._barriers[chunk_index][1].set()

    async def aclose(self) -> None:
        self.closed = True


class MemoryTransactionalSink(TransactionalByteSink):
    """An in-memory transactional sink that counts commits and aborts."""

    def __init__(self, *, fail_commit: bool = False, fail_abort: bool = False) -> None:
        self._buffer = bytearray()
        self.published = bytearray()
        self.commit_count = 0
        self.abort_count = 0
        self.fail_commit = fail_commit
        self.fail_abort = fail_abort
        self.finalized = False

    async def write(self, item: bytes) -> None:
        if self.finalized:
            raise BrokenResourceError("transaction already finalized")

        self._buffer.extend(item)

    async def commit(self) -> None:
        if self.finalized:
            raise BrokenResourceError("transaction already finalized")

        if self.fail_commit:
            raise RuntimeError("commit failed")

        self.published.extend(self._buffer)
        self._buffer.clear()
        self.finalized = True
        self.commit_count += 1

    async def abort(self) -> None:
        if self.finalized:
            raise BrokenResourceError("transaction already finalized")

        if self.fail_abort:
            raise RuntimeError("abort failed")

        self._buffer.clear()
        self.finalized = True
        self.abort_count += 1


async def test_transfer_commits_once_for_streamed_chunks() -> None:
    sink = MemoryTransactionalSink()
    source = BarrierByteSource(b"a", b"b", b"c")
    digest = hashlib.sha256(b"abc").digest()
    async with create_task_group() as tg:
        tg.start_soon(
            partial(
                transfer_stream, source, sink, expected_digest=digest, max_bytes=100
            )
        )
        for i in range(3):
            await source.wait_until_ready(i)
            source.release(i)

    assert sink.commit_count == 1
    assert sink.abort_count == 0
    assert bytes(sink.published) == b"abc"


async def test_transfer_from_memory_object_stream() -> None:
    send, receive = create_memory_object_stream[bytes]()
    sink = MemoryTransactionalSink()
    digest = hashlib.sha256(b"abc").digest()
    async with create_task_group() as tg:
        tg.start_soon(
            partial(
                transfer_stream,
                receive,
                sink,
                expected_digest=digest,
                max_bytes=100,
                close_source=True,
            )
        )
        async with send:
            await send.send(b"a")
            await send.send(b"b")
            await send.send(b"c")

    assert sink.commit_count == 1
    assert sink.abort_count == 0
    assert bytes(sink.published) == b"abc"


async def test_wrong_digest_aborts_without_publishing() -> None:
    sink = MemoryTransactionalSink()
    source = BarrierByteSource(b"a", b"b", b"c")
    with raises_grouped(ValueError, match="digest mismatch"):
        async with create_task_group() as tg:
            tg.start_soon(
                partial(
                    transfer_stream,
                    source,
                    sink,
                    expected_digest=b"\x00" * 32,
                    max_bytes=100,
                )
            )
            for i in range(3):
                await source.wait_until_ready(i)
                source.release(i)

    assert sink.commit_count == 0
    assert sink.abort_count == 1
    assert bytes(sink.published) == b""


async def test_hex_digest_is_accepted() -> None:
    sink = MemoryTransactionalSink()
    source = BarrierByteSource(b"a", b"b", b"c")
    digest = hashlib.sha256(b"abc").hexdigest()
    async with create_task_group() as tg:
        tg.start_soon(
            partial(
                transfer_stream, source, sink, expected_digest=digest, max_bytes=100
            )
        )
        for i in range(3):
            await source.wait_until_ready(i)
            source.release(i)

    assert sink.commit_count == 1
    assert bytes(sink.published) == b"abc"


async def test_overlong_input_aborts_without_publishing() -> None:
    sink = MemoryTransactionalSink()
    source = BarrierByteSource(b"a", b"b", b"c")
    digest = hashlib.sha256(b"abc").digest()
    with raises_grouped(ValueError, match="maximum allowed length"):
        async with create_task_group() as tg:
            tg.start_soon(
                partial(
                    transfer_stream, source, sink, expected_digest=digest, max_bytes=2
                )
            )
            for i in range(3):
                await source.wait_until_ready(i)
                source.release(i)

    assert sink.commit_count == 0
    assert sink.abort_count == 1
    assert bytes(sink.published) == b""


async def test_cancel_while_waiting_for_second_chunk_aborts() -> None:
    sink = MemoryTransactionalSink()
    source = BarrierByteSource(b"a", b"b", b"c")
    digest = hashlib.sha256(b"abc").digest()
    async with create_task_group() as tg:
        handle = tg.start_soon(
            partial(
                transfer_stream, source, sink, expected_digest=digest, max_bytes=100
            )
        )
        await source.wait_until_ready(0)
        source.release(0)
        await source.wait_until_ready(1)
        tg.cancel_scope.cancel()

    assert handle.status == TaskHandle.Status.CANCELLED
    assert sink.commit_count == 0
    assert sink.abort_count == 1
    assert bytes(sink.published) == b""


async def test_commit_failure_propagates_without_claiming_success() -> None:
    sink = MemoryTransactionalSink(fail_commit=True)
    source = BarrierByteSource(b"a", b"b", b"c")
    digest = hashlib.sha256(b"abc").digest()
    with raises_grouped(RuntimeError, match="commit failed"):
        async with create_task_group() as tg:
            tg.start_soon(
                partial(
                    transfer_stream, source, sink, expected_digest=digest, max_bytes=100
                )
            )
            for i in range(3):
                await source.wait_until_ready(i)
                source.release(i)

    assert sink.commit_count == 0
    assert sink.abort_count == 0
    assert bytes(sink.published) == b""


async def test_abort_failure_does_not_mask_original_error() -> None:
    sink = MemoryTransactionalSink(fail_abort=True)
    source = BarrierByteSource(b"a", b"b", b"c")
    with raises_grouped(ValueError, match="digest mismatch"):
        async with create_task_group() as tg:
            tg.start_soon(
                partial(
                    transfer_stream,
                    source,
                    sink,
                    expected_digest=b"\x00" * 32,
                    max_bytes=100,
                )
            )
            for i in range(3):
                await source.wait_until_ready(i)
                source.release(i)

    assert sink.commit_count == 0
    assert sink.abort_count == 0
    assert bytes(sink.published) == b""


async def test_close_source_after_success() -> None:
    sink = MemoryTransactionalSink()
    source = BarrierByteSource(b"a")
    digest = hashlib.sha256(b"a").digest()
    async with create_task_group() as tg:
        tg.start_soon(
            partial(
                transfer_stream,
                source,
                sink,
                expected_digest=digest,
                max_bytes=100,
                close_source=True,
            )
        )
        await source.wait_until_ready(0)
        source.release(0)

    assert source.closed
    assert sink.commit_count == 1


async def test_close_source_after_failure() -> None:
    sink = MemoryTransactionalSink()
    source = BarrierByteSource(b"a")
    with raises_grouped(ValueError):
        async with create_task_group() as tg:
            tg.start_soon(
                partial(
                    transfer_stream,
                    source,
                    sink,
                    expected_digest=b"\x00" * 32,
                    max_bytes=100,
                    close_source=True,
                )
            )
            await source.wait_until_ready(0)
            source.release(0)

    assert source.closed
    assert sink.abort_count == 1


async def test_invalid_max_bytes() -> None:
    sink = MemoryTransactionalSink()
    source = BarrierByteSource(b"a")
    with pytest.raises(ValueError, match="max_bytes"):
        await transfer_stream(source, sink, expected_digest=b"\x00" * 32, max_bytes=0)


async def test_invalid_digest_length() -> None:
    sink = MemoryTransactionalSink()
    source = BarrierByteSource(b"a")
    with pytest.raises(ValueError, match="SHA-256"):
        await transfer_stream(source, sink, expected_digest=b"\x00", max_bytes=10)


async def test_close_source_after_commit_failure() -> None:
    sink = MemoryTransactionalSink(fail_commit=True)
    source = BarrierByteSource(b"a")
    digest = hashlib.sha256(b"a").digest()
    with raises_grouped(RuntimeError, match="commit failed"):
        async with create_task_group() as tg:
            tg.start_soon(
                partial(
                    transfer_stream,
                    source,
                    sink,
                    expected_digest=digest,
                    max_bytes=100,
                    close_source=True,
                )
            )
            await source.wait_until_ready(0)
            source.release(0)

    assert source.closed
    assert sink.commit_count == 0
    assert sink.abort_count == 0
