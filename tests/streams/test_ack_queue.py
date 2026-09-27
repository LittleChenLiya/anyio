from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING

import pytest

from anyio import (
    BrokenResourceError,
    CancelScope,
    ClosedResourceError,
    DuplicateAcknowledgeError,
    EndOfStream,
    WouldBlock,
    create_ack_queue,
    create_task_group,
    fail_after,
    move_on_after,
    sleep,
    sleep_forever,
    wait_all_tasks_blocked,
)

if TYPE_CHECKING:
    from anyio.streams.ack_queue import MessageAcknowledgeHandle

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


def test_invalid_max_buffer() -> None:
    pytest.raises(ValueError, create_ack_queue, 1.0).match(
        "max_buffer_size must be either an integer or math.inf"
    )


def test_max_buffer_too_small() -> None:
    pytest.raises(ValueError, create_ack_queue, 0).match(
        "max_buffer_size must be at least 1"
    )


async def test_acknowledge_and_join() -> None:
    send, receive = create_ack_queue[str](2)
    try:
        await send.send("hello")
        await send.send("anyio")
        assert send.statistics().current_buffer_used == 2

        item1, handle1 = await receive.receive()
        item2, handle2 = await receive.receive()
        assert item1 == "hello"
        assert item2 == "anyio"
        assert send.statistics().current_in_flight == 2

        handle1.acknowledge()
        handle2.acknowledge()
        await send.join()
    finally:
        send.close()
        receive.close()


async def test_clones_and_infinite_capacity() -> None:
    send, receive = create_ack_queue[str](math.inf)
    try:
        send_clone = send.clone()
        receive_clone = receive.clone()
        assert send.statistics().open_send_streams == 2
        assert receive.statistics().open_receive_streams == 2

        # Closing one clone of each end does not close the ends entirely.
        send_clone.close()
        receive_clone.close()
        assert send.statistics().open_send_streams == 1
        assert receive.statistics().open_receive_streams == 1

        for number in range(100):
            send.send_nowait(str(number))

        item, handle = receive.receive_nowait()
        assert item == "0"
        handle.acknowledge()

        send.close()
        for number in range(1, 100):
            item, handle = await receive.receive()
            assert item == str(number)
            handle.acknowledge()

        with pytest.raises(EndOfStream):
            await receive.receive()
    finally:
        receive.close()


async def test_clone_of_closed_stream_raises() -> None:
    send, receive = create_ack_queue[str]()
    send.close()
    receive.close()
    with pytest.raises(ClosedResourceError):
        send.clone()
    with pytest.raises(ClosedResourceError):
        receive.clone()


async def test_cancelled_consumer_redelivers_and_join_waits() -> None:
    """
    A consumer task that receives an item and is cancelled before acknowledging it
    leaves the message outstanding. The message is redelivered (with the same
    sequence number) ahead of any newer items, ``join()`` does not return early,
    double acknowledgement raises, and once redelivered and acknowledged, the
    queue drains.

    """
    send, receive = create_ack_queue[str](2)
    captured: list[MessageAcknowledgeHandle[str]] = []
    consumer_scope = CancelScope()
    try:

        async def get_item_then_block() -> None:
            with consumer_scope:
                _item, handle = await receive.receive()
                captured.append(handle)
                await sleep_forever()

        with fail_after(2):
            async with create_task_group() as tg:
                tg.start_soon(get_item_then_block)
                await send.send("one")
                await wait_all_tasks_blocked()
                assert len(captured) == 1
                consumer_scope.cancel()
                while not send.statistics().current_buffer_used:  # noqa: ASYNC110
                    await sleep(0.02)

        old_handle = captured[0]
        with pytest.raises(ValueError, match="already been abandoned"):
            old_handle.acknowledge()
        assert send.statistics().current_buffer_used == 1

        send.send_nowait("two")
        item, new_handle = await receive.receive()
        assert item == "one"
        assert new_handle.sequence == old_handle.sequence

        with move_on_after(0.1) as scope:
            await send.join()
        assert scope.cancel_called

        new_handle.acknowledge()
        item2, handle2 = await receive.receive()
        assert item2 == "two"
        handle2.acknowledge()
        await send.join()
    finally:
        send.close()
        receive.close()


async def test_explicit_requeue_is_redelivered_first() -> None:
    send, receive = create_ack_queue[str](2)
    try:
        send.send_nowait("one")
        send.send_nowait("two")

        item, handle = receive.receive_nowait()
        assert item == "one"
        first_sequence = handle.sequence
        handle.requeue()

        redelivered, redelivery_handle = receive.receive_nowait()
        assert redelivered == "one"
        assert redelivery_handle.sequence == first_sequence

        with pytest.raises(ValueError, match="already been requeued"):
            handle.acknowledge()

        redelivery_handle.acknowledge()
        _item2, handle2 = receive.receive_nowait()
        handle2.acknowledge()
        await send.join()
    finally:
        send.close()
        receive.close()


async def test_requeue_after_acknowledge_raises() -> None:
    send, receive = create_ack_queue[str]()
    try:
        send.send_nowait("one")
        _item, handle = receive.receive_nowait()
        handle.acknowledge()
        with pytest.raises(ValueError, match="already been acknowledged"):
            handle.requeue()
    finally:
        send.close()
        receive.close()


async def test_sender_blocked_until_receiver_closed() -> None:
    send, receive = create_ack_queue[str]()
    try:
        send.send_nowait("zero")
        with pytest.raises(ExceptionGroup) as exc_info:
            async with create_task_group() as tg:
                tg.start_soon(send.send, "one")
                await wait_all_tasks_blocked()
                assert send.statistics().tasks_waiting_send == 1
                receive.close()
        assert isinstance(exc_info.value.exceptions[0], BrokenResourceError)
    finally:
        send.close()


async def test_join_raises_when_receivers_closed_with_outstanding_items() -> None:
    send, receive = create_ack_queue[str]()
    try:
        send.send_nowait("one")
        _item, _handle = receive.receive_nowait()
        receive.close()
        with pytest.raises(BrokenResourceError, match="receive streams were closed"):
            await send.join()
    finally:
        send.close()


async def test_join_with_waiting_task_is_woken_when_receivers_closed() -> None:
    send, receive = create_ack_queue[str]()
    try:
        send.send_nowait("one")
        _item, _handle = receive.receive_nowait()
        with pytest.raises(ExceptionGroup) as exc_info:
            async with create_task_group() as tg:
                tg.start_soon(send.join)
                await wait_all_tasks_blocked()
                assert send.statistics().tasks_waiting_join == 1
                receive.close()
        assert isinstance(exc_info.value.exceptions[0], BrokenResourceError)
    finally:
        send.close()


async def test_no_tasks_left_waiting_after_all_ends_closed() -> None:
    send, receive = create_ack_queue[str]()
    try:
        send.send_nowait("one")
        _item, _in_flight_handle = receive.receive_nowait()
        with pytest.raises(ExceptionGroup):
            async with create_task_group() as tg:
                tg.start_soon(send.send, "two")
                tg.start_soon(receive.receive)
                tg.start_soon(send.join)
                await wait_all_tasks_blocked()
                assert send.statistics().tasks_waiting_send == 1
                assert send.statistics().tasks_waiting_receive == 1
                assert send.statistics().tasks_waiting_join == 1
                receive.close()

        assert send.statistics().tasks_waiting_send == 0
        assert send.statistics().tasks_waiting_receive == 0
        assert send.statistics().tasks_waiting_join == 0
    finally:
        # The in-flight item will be requeued by the janitor after the test task
        # ends; closing the send end stops the queue for good.
        send.close()


async def test_closed_send_end_allows_draining() -> None:
    send, receive = create_ack_queue[str](2)
    try:
        send.send_nowait("one")
        send.send_nowait("two")
        send.close()

        item1, handle1 = await receive.receive()
        item2, handle2 = await receive.receive()
        assert (item1, item2) == ("one", "two")
        handle1.acknowledge()
        handle2.acknowledge()
        with pytest.raises(EndOfStream):
            await receive.receive()
    finally:
        receive.close()


async def test_iteration_yields_items_and_handles() -> None:
    send, receive = create_ack_queue[str](2)
    try:
        send.send_nowait("one")
        send.send_nowait("two")
        received: list[tuple[str, int]] = []
        async for item, handle in receive:
            received.append((item, handle.sequence))
            handle.acknowledge()
            if len(received) == 2:
                break

        assert received == [("one", 0), ("two", 1)]
    finally:
        send.close()
        receive.close()


async def test_send_on_closed_stream_raises() -> None:
    send, receive = create_ack_queue[str]()
    send.close()
    receive.close()
    with pytest.raises(ClosedResourceError):
        send.send_nowait("one")
    with pytest.raises(ClosedResourceError):
        await send.send("one")
    with pytest.raises(ClosedResourceError):
        receive.receive_nowait()


async def test_handle_context_manager_requeues_unacknowledged_item() -> None:
    send, receive = create_ack_queue[str]()
    try:
        send.send_nowait("one")
        with pytest.raises(ValueError, match="boom"):
            _item, handle = receive.receive_nowait()
            with handle:
                raise ValueError("boom")

        redelivered, redelivery_handle = receive.receive_nowait()
        assert redelivered == "one"
        redelivery_handle.acknowledge()
        await send.join()
    finally:
        send.close()
        receive.close()


async def test_double_acknowledge_raises() -> None:
    send, receive = create_ack_queue[str]()
    try:
        send.send_nowait("hello")
        _item, handle = receive.receive_nowait()
        handle.acknowledge()
        with pytest.raises(
            DuplicateAcknowledgeError, match="already been acknowledged"
        ):
            handle.acknowledge()
        with pytest.raises(DuplicateAcknowledgeError):
            handle.ack()
    finally:
        send.close()
        receive.close()


async def test_capacity_counts_pending_items() -> None:
    """
    With a capacity of 2, the third send stays blocked until one of the
    delivered-but-unacknowledged items is acknowledged.

    """
    send, receive = create_ack_queue[str](2)
    try:
        send.send_nowait("one")
        send.send_nowait("two")
        with pytest.raises(WouldBlock):
            send.send_nowait("three")

        with fail_after(1):
            async with create_task_group() as tg:
                tg.start_soon(send.send, "three")
                await wait_all_tasks_blocked()
                assert send.statistics().tasks_waiting_send == 1

                _item, handle = await receive.receive()
                assert _item == "one"
                await wait_all_tasks_blocked()
                assert send.statistics().tasks_waiting_send == 1

                handle.acknowledge()

            item2, handle2 = receive.receive_nowait()
            item3, handle3 = receive.receive_nowait()
            assert item2 == "two"
            assert item3 == "three"
            handle2.acknowledge()
            handle3.acknowledge()
            await send.join()
    finally:
        send.close()
        receive.close()
