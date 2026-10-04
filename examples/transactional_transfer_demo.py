"""
Short demo of ``anyio.streams.transactional.transfer_to_sink``.

Transfers the three chunks ``b"a"``, ``b"b"``, ``b"c"`` from an in-memory
source stream into an in-memory transactional sink twice: once with the
correct SHA-256 digest (committed) and once with a wrong digest (aborted /
reclaimed). Finally it prints the sink's commit and reclaim (abort) counts.

Run with:  python examples/transactional_transfer_demo.py
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Any

from anyio import EndOfStream, run
from anyio.streams.transactional import DigestMismatchError, transfer_to_sink


class DemoSourceStream:
    """A minimal byte receive stream serving a fixed list of chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if not self._chunks:
            raise EndOfStream

        return self._chunks.pop(0)

    async def aclose(self) -> None:
        pass

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return {}


class DemoTransactionalSink:
    """An in-memory transactional sink counting commits and reclaims (aborts)."""

    def __init__(self) -> None:
        self.pending = bytearray()
        self.published: bytes | None = None
        self.commit_count = 0
        self.reclaim_count = 0

    async def write(self, data: bytes) -> None:
        self.pending += data

    async def commit(self) -> None:
        self.commit_count += 1
        self.published = bytes(self.pending)
        self.pending.clear()

    async def abort(self) -> None:
        self.reclaim_count += 1
        self.pending.clear()


async def main() -> None:
    sink = DemoTransactionalSink()
    good_digest = hashlib.sha256(b"abc").hexdigest()

    # 1) Correct digest: the transfer is committed exactly once
    result = await transfer_to_sink(
        DemoSourceStream([b"a", b"b", b"c"]), sink, expected_sha256=good_digest
    )
    print(
        f"transfer 1: committed {result.bytes_transferred} bytes, "
        f"sha256={result.sha256}"
    )

    # 2) Wrong digest: the transfer is aborted (reclaimed), nothing is published
    try:
        await transfer_to_sink(
            DemoSourceStream([b"a", b"b", b"c"]),
            sink,
            expected_sha256=hashlib.sha256(b"xyz").hexdigest(),
        )
    except DigestMismatchError as exc:
        print(f"transfer 2: rejected ({exc})")

    print(f"published content: {sink.published!r}")
    print(f"commits: {sink.commit_count}, reclaims: {sink.reclaim_count}")


if __name__ == "__main__":
    run(main)
