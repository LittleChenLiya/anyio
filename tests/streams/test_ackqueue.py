from __future__ import annotations

import gc
import math
import sys

import pytest

from anyio import (
    AlreadyAcknowledgedError,
    BrokenResourceError,
    CancelScope,
    ClosedResourceError,
    EndOfStream,
    Event,
    WouldBlock,
    create_ack_queue,
    create_task_group,
    sleep_forever,
    wait_all_tasks_blocked,
)
from anyio.streams.ackqueue import AckHandle, AckQueueReceiveStream, AckQueueSendStream

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


def test_invalid_max_buffer_type() -> None:
    pytest.raises(ValueError, create_ack_queue, 1.0).match(
        "max_buffer_size must be either an integer or math.inf"
    )


def test_zero_max_buffer() -> None:
    pytest.raises(ValueError, create_ack_queue, 0).match(
        "max_buffer_size must be at least 1"
    )


def test_negative_max_buffer() -> None:
    pytest.raises(ValueError, create_ack_queue, -1).match(
        "max_buffer_size must be at least 1"
    )


def test_infinite_max_buffer() -> None:
    send, receive = create_ack_queue[str](math.inf)
    send.send_nowait("hello")
    delivery = receive.receive_nowait()
    assert delivery.item == "hello"
    delivery.ack.acknowledge()
    send.close()
    receive.close()


async def test_send_receive_acknowledge() -> None:
    send, receive = create_ack_queue[str](2)
    with send, receive:
        await send.send("hello")
        delivery = await receive.receive()
        assert delivery.item == "hello"
        assert isinstance(delivery.ack, AckHandle)
        assert receive.statistics().in_flight == 1
        assert receive.statistics().total_outstanding == 1
        delivery.ack.acknowledge()
        assert receive.statistics().total_outstanding == 0
        await send.join()


async def test_nowait_primitives() -> None:
    send, receive = create_ack_queue[str](1)
    with send, receive:
        send.send_nowait("a")
        with pytest.raises(WouldBlock):
            send.send_nowait("b")

        delivery = receive.receive_nowait()
        assert delivery.item == "a"
        # The in-flight message keeps occupying the single capacity slot.
        with pytest.raises(WouldBlock):
            send.send_nowait("b")

        with pytest.raises(WouldBlock):
            receive.receive_nowait()

        delivery.ack.acknowledge()
        send.send_nowait("b")
        assert receive.receive_nowait().item == "b"


async def test_capacity_blocks_third_send_until_acknowledgement() -> None:
    """Acceptance: with capacity 2 the third send only completes after an ack."""
    send, receive = create_ack_queue[str](2)
    events: list[str] = []

    async def send_three() -> None:
        await send.send("first")
        await send.send("second")
        events.append("third:start")
        await send.send("third")
        events.append("third:done")

    with send, receive:
        async with create_task_group() as tg:
            tg.start_soon(send_three)
            await wait_all_tasks_blocked()
            assert events == ["third:start"]
            assert send.statistics().tasks_waiting_send == 1

            first = await receive.receive()
            second = await receive.receive()
            await wait_all_tasks_blocked()
            assert events == ["third:start"]
            assert send.statistics().tasks_waiting_send == 1

            # Acknowledging the first delivery frees the slot for "third".
            first.ack.acknowledge()
            await wait_all_tasks_blocked()
            assert events == ["third:start", "third:done"]

            third = await receive.receive()
            assert third.item == "third"
            second.ack.acknowledge()
            third.ack.acknowledge()
            await send.join()


async def test_duplicate_acknowledgement_fails() -> None:
    send, receive = create_ack_queue[str](1)
    with send, receive:
        await send.send("x")
        delivery = await receive.receive()
        delivery.ack.acknowledge()
        with pytest.raises(AlreadyAcknowledgedError):
            delivery.ack.acknowledge()

        with pytest.raises(AlreadyAcknowledgedError):
            delivery.ack.reject()


async def test_duplicate_reject_fails() -> None:
    send, receive = create_ack_queue[str](1)
    with send, receive:
        await send.send("x")
        delivery = await receive.receive()
        delivery.ack.reject()
        with pytest.raises(AlreadyAcknowledgedError):
            delivery.ack.reject()

        with pytest.raises(AlreadyAcknowledgedError):
            delivery.ack.acknowledge()


async def test_reject_redelivers_same_message_first() -> None:
    send, receive = create_ack_queue[str](2)
    with send, receive:
        await send.send("one")
        await send.send("two")
        first = await receive.receive()
        second = await receive.receive()
        assert first.item == "one"
        first.ack.reject()

        redelivered = await receive.receive()
        assert redelivered.item is first.item
        assert redelivered.ack.sequence != first.ack.sequence
        redelivered.ack.acknowledge()
        second.ack.acknowledge()
        await send.join()


async def test_reject_takes_priority_over_new_messages() -> None:
    send, receive = create_ack_queue[str](4)
    with send, receive:
        await send.send("old")
        old = await receive.receive()
        await send.send("new1")
        await send.send("new2")
        old.ack.reject()
        redelivered = receive.receive_nowait()
        assert redelivered.item == "old"
        assert redelivered.ack.sequence != old.ack.sequence
        new1 = receive.receive_nowait()
        assert new1.item == "new1"
        new2 = receive.receive_nowait()
        assert new2.item == "new2"
        with pytest.raises(WouldBlock):
            receive.receive_nowait()

        redelivered.ack.acknowledge()
        new1.ack.acknowledge()
        new2.ack.acknowledge()
        await send.join()


async def test_join_blocks_while_in_flight() -> None:
    send, receive = create_ack_queue[str](2)
    with send, receive:
        await send.send("a")
        delivery = await receive.receive()
        with CancelScope() as scope:
            scope.cancel()
            await send.join()

        assert scope.cancel_called
        delivery.ack.acknowledge()
        await send.join()


async def test_join_counts_buffered_and_in_flight() -> None:
    send, receive = create_ack_queue[str](4)
    with send, receive:
        await send.send("a")
        await send.send("b")
        first = await receive.receive()
        # One buffered (b), one in flight (a): join must stay blocked.
        with CancelScope() as scope:
            scope.cancel()
            await send.join()

        assert scope.cancel_called
        first.ack.reject()
        # Both messages are buffered again; join is still blocked.
        with CancelScope() as scope:
            scope.cancel()
            await send.join()

        assert scope.cancel_called
        redelivered = await receive.receive()
        redelivered.ack.acknowledge()
        second = await receive.receive()
        second.ack.acknowledge()
        await send.join()


async def test_process_acknowledged_on_success() -> None:
    send, receive = create_ack_queue[str](2)
    with send, receive:
        await send.send("ok")
        async with receive.process() as delivery:
            assert delivery.item == "ok"
            delivery.ack.acknowledge()

        await send.join()


async def test_process_returns_message_on_cancellation() -> None:
    """Acceptance: cancel while taken-but-unacked, then verify redelivery."""
    send, receive = create_ack_queue[str](2)
    taken = Event()
    handles: list[AckHandle[str]] = []

    async def consumer() -> None:
        async with receive.process() as delivery:
            handles.append(delivery.ack)
            taken.set()
            await sleep_forever()
            delivery.ack.acknowledge()

    with send, receive:
        await send.send("payload")
        async with create_task_group() as tg:
            tg.start_soon(consumer)
            await taken.wait()
            await wait_all_tasks_blocked()

            # join() must not return while the message is in flight.
            with CancelScope() as scope:
                scope.cancel()
                await send.join()

            assert scope.cancel_called
            tg.cancel_scope.cancel()

        # The same message comes back, with a fresh handle, ahead of new sends.
        await send.send("later")
        redelivery = await receive.receive()
        assert redelivery.item == "payload"
        assert redelivery.ack.sequence != handles[0].sequence

        # The old handle can no longer be used.
        with pytest.raises(AlreadyAcknowledgedError):
            handles[0].acknowledge()

        later = await receive.receive()
        assert later.item == "later"
        redelivery.ack.acknowledge()
        later.ack.acknowledge()
        await send.join()


async def test_process_returns_message_on_exception() -> None:
    send, receive = create_ack_queue[str](2)
    with send, receive:
        await send.send("boom")
        with pytest.raises(RuntimeError):
            async with receive.process():
                raise RuntimeError("processing failed")

        redelivery = await receive.receive()
        assert redelivery.item == "boom"
        redelivery.ack.acknowledge()
        await send.join()


async def test_process_explicit_reject_is_not_returned_twice() -> None:
    send, receive = create_ack_queue[str](2)
    with send, receive:
        await send.send("m")
        with pytest.raises(RuntimeError):
            async with receive.process() as delivery:
                delivery.ack.reject()
                raise RuntimeError("boom")

        first = await receive.receive()
        assert first.item == "m"
        with pytest.raises(WouldBlock):
            receive.receive_nowait()

        first.ack.acknowledge()
        await send.join()


async def test_abandoned_handle_is_returned() -> None:
    """A handle dropped without being settled returns the message on GC."""
    send, receive = create_ack_queue[str](2)
    with send, receive:
        await send.send("z")
        delivery = receive.receive_nowait()
        sequence = delivery.ack.sequence
        del delivery
        gc.collect()

        redelivery = await receive.receive()
        assert redelivery.item == "z"
        assert redelivery.ack.sequence != sequence
        redelivery.ack.acknowledge()
        await send.join()


async def test_drain_after_send_closed() -> None:
    send, receive = create_ack_queue[str](2)
    await send.send("a")
    await send.send("b")
    await send.aclose()
    delivered: list[str] = []
    while True:
        try:
            delivery = await receive.receive()
        except EndOfStream:
            break

        delivered.append(delivery.item)
        delivery.ack.acknowledge()

    assert delivered == ["a", "b"]
    await send.join()
    await receive.aclose()


async def test_receive_when_send_closed_and_empty() -> None:
    send, receive = create_ack_queue[str](2)
    await send.aclose()
    with pytest.raises(EndOfStream):
        await receive.receive()

    with pytest.raises(EndOfStream):
        receive.receive_nowait()

    await receive.aclose()
    with pytest.raises(ClosedResourceError):
        receive.receive_nowait()


async def test_send_after_receive_closed() -> None:
    send, receive = create_ack_queue[str](2)
    await receive.aclose()
    with pytest.raises(BrokenResourceError):
        send.send_nowait("x")

    with pytest.raises(BrokenResourceError):
        await send.send("x")

    send.close()


async def test_join_after_receive_closed() -> None:
    send, receive = create_ack_queue[str](2)
    await receive.aclose()
    with pytest.raises(BrokenResourceError):
        await send.join()

    send.close()


async def test_close_receive_wakes_sender() -> None:
    send, receive = create_ack_queue[str](1)
    send.send_nowait("a")
    with pytest.raises(ExceptionGroup) as exc_info:
        async with create_task_group() as tg:
            tg.start_soon(send.send, "b")
            await wait_all_tasks_blocked()
            await receive.aclose()

    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], BrokenResourceError)
    send.close()


async def test_close_receive_wakes_receiver() -> None:
    send, receive = create_ack_queue[str](1)
    with pytest.raises(ExceptionGroup) as exc_info:
        async with create_task_group() as tg:
            tg.start_soon(receive.receive)
            await wait_all_tasks_blocked()
            await receive.aclose()

    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], BrokenResourceError)
    send.close()


async def test_close_receive_wakes_join() -> None:
    send, receive = create_ack_queue[str](1)
    await send.send("a")
    with pytest.raises(ExceptionGroup) as exc_info:
        async with create_task_group() as tg:
            tg.start_soon(send.join)
            await wait_all_tasks_blocked()
            await receive.aclose()

    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], BrokenResourceError)
    send.close()


async def test_no_waiters_left_after_all_ends_closed() -> None:
    """Acceptance: closing all ends leaves no blocked send/receive/join tasks."""
    send, receive = create_ack_queue[str](1)
    send.send_nowait("a")
    with pytest.raises(ExceptionGroup) as exc_info:
        async with create_task_group() as tg:
            tg.start_soon(send.send, "b")
            tg.start_soon(receive.receive)
            tg.start_soon(send.join)
            await wait_all_tasks_blocked()
            await receive.aclose()

    assert len(exc_info.value.exceptions) == 2
    assert all(
        isinstance(exc, BrokenResourceError) for exc in exc_info.value.exceptions
    )
    statistics = send.statistics()
    assert statistics.tasks_waiting_send == 0
    assert statistics.tasks_waiting_receive == 0
    assert statistics.tasks_waiting_join == 0
    send.close()


async def test_cancel_during_blocking_receive_redelivers() -> None:
    """Cancelling receive() right when a message arrives does not lose it."""
    send, receive = create_ack_queue[str](1)

    async def scoped_receiver(cancel_scope: CancelScope) -> None:
        with cancel_scope:
            await receive.receive()

    with send, receive:
        scope = CancelScope()
        async with create_task_group() as tg:
            tg.start_soon(scoped_receiver, scope)
            await wait_all_tasks_blocked()
            scope.cancel()
            send.send_nowait("hello")
            await wait_all_tasks_blocked()

        delivery = receive.receive_nowait()
        assert delivery.item == "hello"
        delivery.ack.acknowledge()


async def test_clone() -> None:
    send1, receive1 = create_ack_queue[str](2)
    send2 = send1.clone()
    receive2 = receive1.clone()
    await send1.aclose()
    await receive1.aclose()
    send2.send_nowait("hello")
    delivery = receive2.receive_nowait()
    assert delivery.item == "hello"
    delivery.ack.acknowledge()
    send2.close()
    receive2.close()


async def test_clone_closed() -> None:
    send, receive = create_ack_queue[str](1)
    await send.aclose()
    await receive.aclose()
    pytest.raises(ClosedResourceError, send.clone)
    pytest.raises(ClosedResourceError, receive.clone)


async def test_statistics() -> None:
    send, receive = create_ack_queue[None](2)
    for stream in (send, receive):
        statistics = stream.statistics()
        assert statistics.max_buffer_size == 2
        assert statistics.current_buffer_used == 0
        assert statistics.in_flight == 0
        assert statistics.total_outstanding == 0
        assert statistics.open_send_streams == 1
        assert statistics.open_receive_streams == 1
        assert statistics.tasks_waiting_send == 0
        assert statistics.tasks_waiting_receive == 0
        assert statistics.tasks_waiting_join == 0

    send.send_nowait(None)
    delivery = receive.receive_nowait()
    statistics = send.statistics()
    assert statistics.in_flight == 1
    assert statistics.total_outstanding == 1
    delivery.ack.acknowledge()
    assert send.statistics().total_outstanding == 0
    send.close()
    receive.close()


async def test_sync_close() -> None:
    send, receive = create_ack_queue[None](1)
    with send, receive:
        pass

    with pytest.raises(ClosedResourceError):
        send.send_nowait(None)

    with pytest.raises(ClosedResourceError):
        receive.receive_nowait()


async def test_type_variance() -> None:
    """This test only exists to be checked by the static type checker."""
    send, receive = create_ack_queue[float](2)
    receive1: AckQueueReceiveStream[complex] = receive  # noqa: F841
    send1: AckQueueSendStream[int] = send  # noqa: F841
    send.close()
    receive.close()


async def test_not_closed_warning() -> None:
    send, unused_receive = create_ack_queue[int](2)
    with pytest.warns(
        ResourceWarning, match="Unclosed <AckQueueSendStream at [0-9a-f]+>"
    ):
        del send
        gc.collect()

    unused_send, receive = create_ack_queue[int](2)
    with pytest.warns(
        ResourceWarning, match="Unclosed <AckQueueReceiveStream at [0-9a-f]+>"
    ):
        del receive
        gc.collect()

    unused_receive.close()
    unused_send.close()
