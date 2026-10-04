from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from anyio import EndOfStream, Event, create_task_group
from anyio.abc import ByteReceiveStream
from anyio.streams.transactional import (
    DigestMismatchError,
    LengthMismatchError,
    SizeLimitExceededError,
    TransactionalSink,
    TransferResult,
    transfer_to_sink,
)

pytestmark = pytest.mark.anyio


class ChunkedByteStream(ByteReceiveStream):
    """A minimal in-memory byte receive stream fed from a list of chunks."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        gate: Event | None = None,
        gate_before: int | None = None,
        entered_gate: Event | None = None,
    ) -> None:
        self._chunks = chunks
        self._gate = gate
        self._gate_before = gate_before
        self._entered_gate = entered_gate
        self.close_count = 0

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if not self._chunks:
            raise EndOfStream

        if self._gate is not None and len(self._chunks) == self._gate_before:
            if self._entered_gate is not None:
                self._entered_gate.set()

            await self._gate.wait()

        return self._chunks.pop(0)

    async def aclose(self) -> None:
        self.close_count += 1

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return {}


class MemoryTransactionalSink:
    """An in-memory transactional sink recording commit/abort (reclaim) counts."""

    def __init__(self, *, fail_commit: bool = False, fail_abort: bool = False) -> None:
        self.pending = bytearray()
        self.published: bytes | None = None
        self.commit_count = 0
        self.abort_count = 0
        self.fail_commit = fail_commit
        self.fail_abort = fail_abort

    async def write(self, data: bytes) -> None:
        self.pending += data

    async def commit(self) -> None:
        if self.fail_commit:
            raise RuntimeError("commit failed")

        self.commit_count += 1
        self.published = bytes(self.pending)
        self.pending.clear()

    async def abort(self) -> None:
        self.abort_count += 1
        self.pending.clear()
        if self.fail_abort:
            raise RuntimeError("abort failed")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def test_commit_exactly_once_for_three_chunks() -> None:
    source = ChunkedByteStream([b"a", b"b", b"c"])
    sink = MemoryTransactionalSink()
    result = await transfer_to_sink(
        source, sink, expected_sha256=sha256_hex(b"abc"), expected_size=3
    )
    assert result == TransferResult(bytes_transferred=3, sha256=sha256_hex(b"abc"))
    assert sink.commit_count == 1
    assert sink.abort_count == 0
    assert sink.published == b"abc"
    assert source.close_count == 1


async def test_digest_mismatch_aborts_and_keeps_published_content() -> None:
    sink = MemoryTransactionalSink()
    sink.published = b"previous"
    source = ChunkedByteStream([b"a", b"b", b"c"])
    with pytest.raises(DigestMismatchError):
        await transfer_to_sink(source, sink, expected_sha256=sha256_hex(b"xyz"))

    assert sink.commit_count == 0
    assert sink.abort_count == 1
    assert sink.published == b"previous"
    assert source.close_count == 1


async def test_oversized_input_aborts_and_keeps_published_content() -> None:
    sink = MemoryTransactionalSink()
    sink.published = b"previous"
    source = ChunkedByteStream([b"ab", b"cd", b"ef"])
    with pytest.raises(SizeLimitExceededError):
        await transfer_to_sink(source, sink, max_bytes=3)

    assert sink.commit_count == 0
    assert sink.abort_count == 1
    assert sink.published == b"previous"
    assert source.close_count == 1


async def test_length_mismatch_aborts() -> None:
    sink = MemoryTransactionalSink()
    source = ChunkedByteStream([b"abc"])
    with pytest.raises(LengthMismatchError):
        await transfer_to_sink(source, sink, expected_size=4)

    assert sink.commit_count == 0
    assert sink.abort_count == 1
    assert sink.published is None


async def test_cancellation_while_waiting_for_second_chunk_aborts() -> None:
    gate = Event()
    entered_gate = Event()
    source = ChunkedByteStream(
        [b"a", b"b", b"c"], gate=gate, gate_before=2, entered_gate=entered_gate
    )
    sink = MemoryTransactionalSink()

    async with create_task_group() as tg:
        tg.start_soon(transfer_to_sink, source, sink)
        # Wait until the transfer is blocked waiting for the second chunk
        await entered_gate.wait()
        tg.cancel_scope.cancel()

    assert sink.commit_count == 0
    assert sink.abort_count == 1
    assert sink.published is None
    assert source.close_count == 1


async def test_commit_failure_is_not_reported_as_success() -> None:
    sink = MemoryTransactionalSink(fail_commit=True)
    source = ChunkedByteStream([b"abc"])
    with pytest.raises(RuntimeError, match="commit failed"):
        await transfer_to_sink(source, sink, expected_sha256=sha256_hex(b"abc"))

    assert sink.commit_count == 0
    assert sink.abort_count == 1
    assert sink.published is None


async def test_abort_failure_does_not_mask_original_error() -> None:
    sink = MemoryTransactionalSink(fail_abort=True)
    source = ChunkedByteStream([b"abc"])
    with pytest.raises(DigestMismatchError) as excinfo:
        await transfer_to_sink(source, sink, expected_sha256=sha256_hex(b"xyz"))

    assert sink.abort_count == 1
    notes = getattr(excinfo.value, "__notes__", [])
    assert any("abort() also failed" in note for note in notes)


async def test_close_source_false_retains_ownership() -> None:
    source = ChunkedByteStream([b"abc"])
    sink = MemoryTransactionalSink()
    await transfer_to_sink(source, sink, close_source=False)
    assert source.close_count == 0
    assert sink.commit_count == 1


async def test_expected_digest_accepts_raw_bytes() -> None:
    source = ChunkedByteStream([b"abc"])
    sink = MemoryTransactionalSink()
    result = await transfer_to_sink(
        source, sink, expected_sha256=hashlib.sha256(b"abc").digest()
    )
    assert result.bytes_transferred == 3
    assert sink.commit_count == 1


async def test_sink_protocol_runtime_check() -> None:
    assert isinstance(MemoryTransactionalSink(), TransactionalSink)
