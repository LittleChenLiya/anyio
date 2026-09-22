"""A short demonstration of anyio.receive_any() with memory object streams."""

from __future__ import annotations

from anyio import (
    CancelScope,
    create_memory_object_stream,
    create_task_group,
    receive_any,
    run,
    wait_all_tasks_blocked,
)


async def main() -> None:
    # Three streams, two of them prefilled: only the first readable stream
    # (in the order given) loses exactly one item
    send1, recv1 = create_memory_object_stream[str](2)
    send2, recv2 = create_memory_object_stream[str](2)
    send3, recv3 = create_memory_object_stream[str](2)
    with send1, recv1, send2, recv2, send3, recv3:
        send2.send_nowait("alpha")
        send2.send_nowait("beta")
        send3.send_nowait("gamma")

        index, item = await receive_any([recv1, recv2, recv3])
        print(f"selected index: {index}, item: {item!r}")
        print("remaining items:", [recv2.receive_nowait(), recv3.receive_nowait()])

    # Cancellation while registered: all waiter registrations are revoked
    send4, recv4 = create_memory_object_stream[str](0)
    send5, recv5 = create_memory_object_stream[str](0)
    with send4, recv4, send5, recv5:
        with CancelScope() as scope:
            async with create_task_group() as tg:
                tg.start_soon(receive_any, [recv4, recv5])
                await wait_all_tasks_blocked()
                scope.cancel()

        stats = [recv4.statistics(), recv5.statistics()]
        print(
            "tasks_waiting_receive after cancel:",
            [stat.tasks_waiting_receive for stat in stats],
        )


if __name__ == "__main__":
    run(main)
