"""
Terminal demonstration of the anyio broadcast stream.

A publisher broadcasts an incrementing counter to several subscriptions with
different buffer capacities and consumption speeds. A fourth subscriber joins
late and leaves early, demonstrating that subscriptions only receive items
published while they are alive. After the run (90 seconds by default), the
publishing end is closed and every subscriber drains its remaining buffered
items before seeing EndOfStream. Finally, the exact sequence of items each
subscription received is printed and checked for contiguity.

Usage::

    python examples/broadcast_demo.py                     # 90 second demo
    python examples/broadcast_demo.py --duration 10       # shorter run
    python examples/broadcast_demo.py --backend trio      # run on Trio
"""

from __future__ import annotations

import argparse

import anyio
from anyio import EndOfStream
from anyio.streams.broadcast import BroadcastReceiveStream, BroadcastStream


async def subscriber(
    name: str,
    stream: BroadcastReceiveStream[int],
    delay: float,
    results: dict[str, list[int]],
) -> None:
    received: list[int] = []
    try:
        while True:
            item = await stream.receive()
            received.append(item)
            print(f"  {name}: received {item} ({len(received)} so far)")
            await anyio.sleep(delay)
    except EndOfStream:
        print(f"  {name}: publishing end closed, drained {len(received)} items")
    finally:
        results[name] = received
        await stream.aclose()


async def late_subscriber(
    hub: BroadcastStream[int],
    join_at: float,
    leave_at: float,
    results: dict[str, list[int]],
) -> None:
    await anyio.sleep(join_at)
    stream = hub.subscribe(2)
    print("  late: subscribed (only receives items published from now on)")
    received: list[int] = []
    try:
        with anyio.move_on_after(leave_at - join_at):
            while True:
                item = await stream.receive()
                received.append(item)
                print(f"  late: received {item} ({len(received)} so far)")
                await anyio.sleep(0.2)
    finally:
        print("  late: unsubscribing")
        results["late"] = received
        await stream.aclose()


async def publisher(hub: BroadcastStream[int], duration: float, interval: float) -> int:
    item = 0
    started = anyio.current_time()
    while anyio.current_time() - started < duration:
        before = anyio.current_time()
        await hub.publish(item)
        blocked = anyio.current_time() - before
        if blocked > 0.05:
            print(f"published {item} (blocked {blocked:.2f}s on full subscriptions)")
        else:
            print(f"published {item}")

        item += 1
        await anyio.sleep(interval)

    return item


async def main(duration: float, interval: float) -> None:
    results: dict[str, list[int]] = {}
    async with BroadcastStream[int]() as hub:
        async with anyio.create_task_group() as tg:
            tg.start_soon(subscriber, "fast  ", hub.subscribe(4), 0.1, results)
            tg.start_soon(subscriber, "medium", hub.subscribe(2), 0.5, results)
            tg.start_soon(subscriber, "slow  ", hub.subscribe(1), 1.0, results)
            tg.start_soon(late_subscriber, hub, duration / 3, 2 * duration / 3, results)
            published = await publisher(hub, duration, interval)
            print("closing the broadcast stream; subscribers drain their buffers")
            hub.close()

    print(f"\n{published} items were published in total")
    ok = True
    for name, received in results.items():
        print(f"{name.strip():>6} received {len(received)} items: {received}")
        expected = list(range(received[0], received[0] + len(received)))
        if received != expected:
            print(f"ERROR: {name.strip()} missed items!")
            ok = False

    if ok:
        print("OK: every subscription received a contiguous sequence of items")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=90.0,
        help="how long to publish items, in seconds (default: 90)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="interval between published items, in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--backend",
        choices=["asyncio", "trio"],
        default="asyncio",
        help="async backend to run on (default: asyncio)",
    )
    args = parser.parse_args()
    anyio.run(main, args.duration, args.interval, backend=args.backend)
