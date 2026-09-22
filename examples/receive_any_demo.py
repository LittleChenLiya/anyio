from __future__ import annotations

from anyio import (
    CancelScope,
    Event,
    create_memory_object_stream,
    create_task_group,
    receive_any,
    run,
    wait_all_tasks_blocked,
)


async def main() -> None:
    # 1) Three streams, two prefilled: the first ready stream in input order wins.
    send1, receive1 = create_memory_object_stream[str](1)
    send2, receive2 = create_memory_object_stream[str](1)
    send3, receive3 = create_memory_object_stream[str](1)
    send1.send_nowait("a")
    send2.send_nowait("b")
    index, item = await receive_any([receive1, receive2, receive3])
    print(f"selected index={index}, item={item!r}")  # noqa: T201
    print(  # noqa: T201
        "remaining untouched items: "
        f"stream2={receive2.receive_nowait()!r}, stream3 empty (no item was sent)"
    )

    # 2) Two zero-buffer streams: cancel a registered receive_any, then two senders
    # race. The losing sender stays blocked and an ordinary receive() gets its item.
    send_z1, recv_z1 = create_memory_object_stream[str](0)
    send_z2, recv_z2 = create_memory_object_stream[str](0)
    barrier = Event()

    async def sender(send_stream, value: str) -> None:
        await barrier.wait()
        await send_stream.send(value)

    with CancelScope() as scope:
        async with create_task_group() as tg:
            tg.start_soon(receive_any, [recv_z1, recv_z2])
            await wait_all_tasks_blocked()
            scope.cancel()

    results: list[tuple[int, str]] = []

    async def multi_receiver() -> None:
        results.append(await receive_any([recv_z1, recv_z2]))

    async with create_task_group() as tg:
        tg.start_soon(sender, send_z1, "x")
        tg.start_soon(sender, send_z2, "y")
        tg.start_soon(multi_receiver)
        await wait_all_tasks_blocked()
        barrier.set()
        await wait_all_tasks_blocked()
        winner = results[0][0]
        loser = recv_z1 if winner == 1 else recv_z2
        loser_item = "x" if winner == 1 else "y"
        assert await loser.receive() == loser_item

    stats = [
        recv_z1.statistics().tasks_waiting_receive
        + recv_z1.statistics().tasks_waiting_send,
        recv_z2.statistics().tasks_waiting_receive
        + recv_z2.statistics().tasks_waiting_send,
    ]
    print(f"winner index={winner}; ordinary receive got {loser_item!r}")  # noqa: T201
    print(f"waiters after cancellation/completion: {stats}")  # noqa: T201

    for stream in (
        send1,
        send2,
        send3,
        receive1,
        receive2,
        receive3,
        send_z1,
        send_z2,
        recv_z1,
        recv_z2,
    ):
        stream.close()


if __name__ == "__main__":
    run(main)
