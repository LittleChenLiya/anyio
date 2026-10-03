"""
Short demo: stream bytes to a transactional sink, printing commit/abort counts.

Run with::

    python examples/transfer_demo.py
"""

from __future__ import annotations

import hashlib
from contextlib import suppress

from anyio import EndOfStream, run
from anyio.abc import ByteReceiveStream, TransactionalByteSink
from anyio.streams.transfer import transfer_stream


class ListSource(ByteReceiveStream):
    """A trivial byte source that yields chunks from a list."""

    def __init__(self, *chunks: bytes) -> None:
        self._chunks = list(chunks)
        self._index = 0

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self._index >= len(self._chunks):
            raise EndOfStream

        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    async def aclose(self) -> None:
        pass


class MemorySink(TransactionalByteSink):
    """An in-memory transactional sink that counts commits and aborts."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.published = bytearray()
        self.commit_count = 0
        self.abort_count = 0

    async def write(self, item: bytes) -> None:
        self._buffer.extend(item)

    async def commit(self) -> None:
        self.published.extend(self._buffer)
        self._buffer.clear()
        self.commit_count += 1

    async def abort(self) -> None:
        self._buffer.clear()
        self.abort_count += 1


async def main() -> None:
    digest = hashlib.sha256(b"abc").hexdigest()

    # 1. Correct digest, within the size limit -- the sink is committed once.
    sink = MemorySink()
    await transfer_stream(
        ListSource(b"a", b"b", b"c"),
        sink,
        expected_digest=digest,
        max_bytes=100,
    )
    print(  # noqa: T201
        f"correct digest : commits={sink.commit_count} "
        f"aborts={sink.abort_count} published={bytes(sink.published)!r}"
    )

    # 2. Wrong digest -- the sink is aborted, nothing is published.
    sink = MemorySink()
    with suppress(ValueError):
        await transfer_stream(
            ListSource(b"a", b"b", b"c"),
            sink,
            expected_digest="0" * 64,
            max_bytes=100,
        )
    print(  # noqa: T201
        f"wrong digest   : commits={sink.commit_count} "
        f"aborts={sink.abort_count} published={bytes(sink.published)!r}"
    )

    # 3. Input exceeds the size limit -- the sink is aborted, nothing is published.
    sink = MemorySink()
    with suppress(ValueError):
        await transfer_stream(
            ListSource(b"a", b"b", b"c"),
            sink,
            expected_digest=digest,
            max_bytes=2,
        )
    print(  # noqa: T201
        f"over-length    : commits={sink.commit_count} "
        f"aborts={sink.abort_count} published={bytes(sink.published)!r}"
    )


run(main)
