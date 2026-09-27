#!/usr/bin/env python3
"""
Demonstration of AnyIO's acknowledgeable queue (``anyio.create_ack_queue``).

The demo is deterministic and can be run repeatedly without leaving tasks or
messages behind::

    python examples/ack_queue_demo.py            # asyncio backend
    python examples/ack_queue_demo.py trio       # trio backend

It shows two scenarios:

1. Normal acknowledgement with backpressure -- with a capacity of 2, the third
   sender can only make progress once the first message has been acknowledged.
2. Automatic redelivery on cancellation -- a consumer that is cancelled while
   holding an unacknowledged message has the same message put back at the head
   of the queue, where it is redelivered before newer messages.
"""

from __future__ import annotations

import sys

import anyio


def log(message: str) -> None:
    print(message, flush=True)  # noqa: T201


async def demo_backpressure() -> None:
    log("== backpressure and acknowledgement ==")
    send, receive = anyio.create_ack_queue[str](2)
    with send, receive:
        async with anyio.create_task_group() as tg:

            async def producer() -> None:
                for number in range(3):
                    message = f"msg-{number}"
                    await send.send(message)
                    log(f"producer: sent {message}")

            tg.start_soon(producer)
            await anyio.wait_all_tasks_blocked()

            first = await receive.receive()
            second = await receive.receive()
            log("consumer: received two messages; third send is blocked")
            log(f"blocked senders: {send.statistics().tasks_waiting_send}")

            first.ack.acknowledge()
            log("consumer: acknowledged msg-0; third send can now proceed")
            await anyio.wait_all_tasks_blocked()
            log(f"blocked senders now: {send.statistics().tasks_waiting_send}")

            third = await receive.receive()
            second.ack.acknowledge()
            third.ack.acknowledge()
            await send.join()
            log("join() returned: every message was processed")


async def demo_cancel_redelivery() -> None:
    log("== cancellation and redelivery ==")
    send, receive = anyio.create_ack_queue[str](2)
    message_taken = anyio.Event()
    try:
        async with anyio.create_task_group() as tg:

            async def flaky_consumer() -> None:
                # process() returns the message if the block is cancelled.
                async with receive.process() as delivery:
                    log(f"consumer A: took {delivery.item!r}, working...")
                    message_taken.set()
                    await anyio.sleep_forever()  # gets cancelled here
                    delivery.ack.acknowledge()

            await send.send("important")
            tg.start_soon(flaky_consumer)
            await message_taken.wait()

            # join() must not finish while a message is still unacknowledged.
            with anyio.move_on_after(0.05) as scope:
                await send.join()

            log("join() stayed blocked while unacked: " + str(scope.cancel_called))
            tg.cancel_scope.cancel()

        # After the cancelled consumer was torn down, a new consumer gets the
        # very same message back, ahead of any later sends.
        async with anyio.create_task_group() as tg:
            await send.send("after")
            redelivered = await receive.receive()
            log("consumer B: redelivered " + repr(redelivered.item))
            after = await receive.receive()
            log("consumer B: then received " + repr(after.item))
            redelivered.ack.acknowledge()
            after.ack.acknowledge()
            await send.join()
            log("join() returned after redelivery was acknowledged")
    finally:
        send.close()
        receive.close()


async def main() -> None:
    await demo_backpressure()
    log("")
    await demo_cancel_redelivery()


if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "asyncio"
    anyio.run(main, backend=backend)
