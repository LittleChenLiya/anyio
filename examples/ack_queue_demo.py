"""
Demonstrate the acknowledgement queue (``anyio.create_ack_queue``).

Run either scenario repeatedly, on either backend::

    python examples/ack_queue_demo.py --scenario acknowledge
    python examples/ack_queue_demo.py --scenario redelivery
    python examples/ack_queue_demo.py --scenario redelivery --backend trio

"""

from __future__ import annotations

import argparse

from anyio import (
    CancelScope,
    create_ack_queue,
    create_task_group,
    run,
    sleep,
    wait_all_tasks_blocked,
)


async def scenario_acknowledge() -> None:
    """Three messages through a capacity-2 queue; the third send unblocks on ack."""
    send, receive = create_ack_queue[str](2)
    await send.send("one")
    await send.send("two")

    async with create_task_group() as tg:
        tg.start_soon(send.send, "three")
        await wait_all_tasks_blocked()
        print("two items delivered, third send is blocked (capacity is 2)")

        item, handle = await receive.receive()
        assert item == "one"
        print(f"processed and acknowledging: {item!r} (seq={handle.sequence})")
        handle.acknowledge()
        print("the third send can now complete, freeing the next slot")

        for expected in ("two", "three"):
            item, handle = await receive.receive()
            assert item == expected
            print(f"processed and acknowledging: {item!r} (seq={handle.sequence})")
            handle.acknowledge()

        await send.join()
        print("join() returned: every message was acknowledged")


async def scenario_redelivery() -> None:
    """A consumer cancelled after receipt causes an automatic redelivery."""
    send, receive = create_ack_queue[str](2)
    captured_sequence: list[int] = []
    consumer_scope = CancelScope()

    async def flaky_consumer() -> None:
        with consumer_scope:
            item, handle = await receive.receive()
            captured_sequence.append(handle.sequence)
            print(f"flaky consumer got {item!r} (seq={handle.sequence}); cancelling")
            await sleep(10)

    async with create_task_group() as tg:
        tg.start_soon(flaky_consumer)
        await send.send("important")
        await wait_all_tasks_blocked()
        consumer_scope.cancel()
        await wait_all_tasks_blocked()

        # Give the background reaper a moment to requeue the abandoned message.
        while not send.statistics().current_buffer_used:
            await sleep(0.02)

    item, handle = await receive.receive()
    print(
        f"message redelivered: {item!r} with the SAME seq={handle.sequence} "
        f"(original seq={captured_sequence[0]})"
    )
    assert handle.sequence == captured_sequence[0]
    handle.acknowledge()
    await send.join()
    print("join() returned after the redelivered message was acknowledged")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("acknowledge", "redelivery"),
        default="acknowledge",
    )
    parser.add_argument("--backend", choices=("asyncio", "trio"), default="asyncio")
    args = parser.parse_args()

    scenario = scenario_acknowledge if args.scenario == "acknowledge" else scenario_redelivery
    run(scenario, backend=args.backend)


if __name__ == "__main__":
    main()
