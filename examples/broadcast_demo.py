"""
Repeatable terminal demonstration of the broadcast object stream.

Run::

    python examples/broadcast_demo.py            # asyncio
    python examples/broadcast_demo.py trio       # trio

The entire demo takes a fraction of a second, comfortably within 90 seconds,
and prints the actual sequence of items received by each of the two
capacity-1 subscriptions.

If AnyIO is not installed, run against the source tree with::

    PYTHONPATH=src python examples/broadcast_demo.py trio

"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from anyio import (
    CancelScope,
    EndOfStream,
    WouldBlock,
    create_broadcast_object_stream,
    create_task_group,
    run,
    sleep,
    wait_all_tasks_blocked,
)

if TYPE_CHECKING:
    from anyio.streams.broadcast import BroadcastObjectReceiveStream


async def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "asyncio"
    print(f"== broadcast object stream demo (backend: {backend}) ==")  # noqa: T201

    publisher = create_broadcast_object_stream[str]()
    sub_a = publisher.subscribe(1)
    sub_b = publisher.subscribe(1)
    seq_a: list[str] = []
    seq_b: list[str] = []

    def drain(
        subscription: BroadcastObjectReceiveStream[str], target: list[str]
    ) -> None:
        while True:
            try:
                target.append(subscription.receive_nowait())
            except (WouldBlock, EndOfStream):
                break

    # Step 1: two capacity-1 subscriptions both receive x.
    print("1. publish x -> both subscriptions receive x")  # noqa: T201
    await publisher.publish("x")
    drain(sub_a, seq_a)
    drain(sub_b, seq_b)
    print(f"   A={seq_a} B={seq_b}")  # noqa: T201

    # Step 2: refill and drain only A. B stays full, so publish y blocks.
    print("2. publish x, drain only A; publish y must block")  # noqa: T201
    await publisher.publish("x")
    drain(sub_a, seq_a)

    still_blocked = True

    async def publish_y() -> None:
        nonlocal still_blocked
        await publisher.publish("y")
        still_blocked = False

    with CancelScope() as cancel_scope:
        async with create_task_group() as tg:
            tg.start_soon(publish_y)
            await wait_all_tasks_blocked()
            await sleep(0.05)
            print(f"   publish(y) still blocked: {still_blocked}")  # noqa: T201

            # Step 3: cancel the blocked publish. Nobody may receive y.
            print("3. cancel blocked publish y -> neither gets y")  # noqa: T201
            cancel_scope.cancel()

    assert still_blocked is True
    drain(sub_a, seq_a)
    drain(sub_b, seq_b)
    print(f"   after cancellation: A={seq_a} B={seq_b}")  # noqa: T201
    assert "y" not in seq_a and "y" not in seq_b, (seq_a, seq_b)

    # Step 4: drain B and publish z; both subscriptions receive z.
    print("4. drain B, publish z -> both receive z")  # noqa: T201
    drain(sub_b, seq_b)
    await publisher.publish("z")
    drain(sub_a, seq_a)
    drain(sub_b, seq_b)
    print(f"   A={seq_a} B={seq_b}")  # noqa: T201

    # Step 5: queue one item, close the publisher, then verify drain + EOS.
    print("5. close publisher -> buffered items drained, then EndOfStream")  # noqa: T201
    await publisher.publish("final")
    publisher.close()
    drain(sub_a, seq_a)
    drain(sub_b, seq_b)

    for name, subscription in (("A", sub_a), ("B", sub_b)):
        try:
            subscription.receive_nowait()
        except EndOfStream:
            print(f"   subscription {name}: EndOfStream after drain")  # noqa: T201

    print(f"== final received sequences: A={seq_a} B={seq_b} ==")  # noqa: T201
    sub_a.close()
    sub_b.close()


if __name__ == "__main__":
    run(main, backend=sys.argv[1] if len(sys.argv) > 1 else "asyncio")
