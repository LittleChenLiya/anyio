from __future__ import annotations

import gc
import sys
from typing import TYPE_CHECKING, NoReturn

import pytest

from anyio import (
    BrokenResourceError,
    CancelScope,
    ClosedResourceError,
    EndOfStream,
    Event,
    WouldBlock,
    create_memory_object_stream,
    create_task_group,
    fail_after,
    receive_any,
    wait_all_tasks_blocked,
)
from anyio.streams.memory import (
    MemoryObjectReceiveStream,
    MemoryObjectSendStream,
    _MemoryObjectItemReceiver,
)

from ..conftest import asyncio_params

if TYPE_CHECKING:
    from anyio.abc import ObjectReceiveStream, ObjectSendStream, TaskStatus

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


def test_invalid_max_buffer() -> None:
    pytest.raises(ValueError, create_memory_object_stream, 1.0).match(
        "max_buffer_size must be either an integer or math.inf"
    )


def test_negative_max_buffer() -> None:
    pytest.raises(ValueError, create_memory_object_stream, -1).match(
        "max_buffer_size cannot be negative"
    )


async def test_receive_then_send() -> None:
    async def receiver() -> None:
        received_objects.append(await receive.receive())
        received_objects.append(await receive.receive())

    send, receive = create_memory_object_stream[str](0)
    received_objects: list[str] = []
    async with create_task_group() as tg:
        tg.start_soon(receiver)
        await wait_all_tasks_blocked()
        await send.send("hello")
        await send.send("anyio")

    assert received_objects == ["hello", "anyio"]

    send.close()
    receive.close()


async def test_receive_then_send_nowait() -> None:
    async def receiver() -> None:
        received_objects.append(await receive.receive())

    send, receive = create_memory_object_stream[str](0)
    received_objects: list[str] = []
    async with create_task_group() as tg:
        tg.start_soon(receiver)
        tg.start_soon(receiver)
        await wait_all_tasks_blocked()
        send.send_nowait("hello")
        send.send_nowait("anyio")

    assert sorted(received_objects, reverse=True) == ["hello", "anyio"]

    send.close()
    receive.close()


async def test_send_then_receive_nowait() -> None:
    send, receive = create_memory_object_stream[str](0)
    async with create_task_group() as tg:
        tg.start_soon(send.send, "hello")
        await wait_all_tasks_blocked()
        assert receive.receive_nowait() == "hello"

    send.close()
    receive.close()


async def test_send_is_unblocked_after_receive_nowait() -> None:
    send, receive = create_memory_object_stream[str](1)
    send.send_nowait("hello")

    with fail_after(1):
        async with create_task_group() as tg:
            tg.start_soon(send.send, "anyio")
            await wait_all_tasks_blocked()
            assert receive.receive_nowait() == "hello"

    assert receive.receive_nowait() == "anyio"

    send.close()
    receive.close()


async def test_send_nowait_then_receive_nowait() -> None:
    send, receive = create_memory_object_stream[str](2)
    send.send_nowait("hello")
    send.send_nowait("anyio")
    assert receive.receive_nowait() == "hello"
    assert receive.receive_nowait() == "anyio"

    send.close()
    receive.close()


async def test_iterate() -> None:
    async def receiver() -> None:
        received_objects.extend([item async for item in receive])

    send, receive = create_memory_object_stream[str]()
    received_objects: list[str] = []
    async with create_task_group() as tg:
        tg.start_soon(receiver)
        await send.send("hello")
        await send.send("anyio")
        await send.aclose()

    assert received_objects == ["hello", "anyio"]

    send.close()
    receive.close()


async def test_receive_send_closed_send_stream() -> None:
    send, receive = create_memory_object_stream[None]()
    await send.aclose()
    with pytest.raises(EndOfStream):
        receive.receive_nowait()

    with pytest.raises(ClosedResourceError):
        await send.send(None)

    receive.close()


async def test_receive_send_closed_receive_stream() -> None:
    send, receive = create_memory_object_stream[None]()
    await receive.aclose()
    with pytest.raises(ClosedResourceError):
        receive.receive_nowait()

    with pytest.raises(BrokenResourceError):
        await send.send(None)

    send.close()


async def test_cancel_receive() -> None:
    send, receive = create_memory_object_stream[str]()
    async with create_task_group() as tg:
        tg.start_soon(receive.receive)
        await wait_all_tasks_blocked()
        tg.cancel_scope.cancel()

    with pytest.raises(WouldBlock):
        send.send_nowait("hello")

    send.close()
    receive.close()


async def test_cancel_send() -> None:
    send, receive = create_memory_object_stream[str]()
    async with create_task_group() as tg:
        tg.start_soon(send.send, "hello")
        await wait_all_tasks_blocked()
        tg.cancel_scope.cancel()

    with pytest.raises(WouldBlock):
        receive.receive_nowait()

    send.close()
    receive.close()


async def test_clone() -> None:
    send1, receive1 = create_memory_object_stream[str](1)
    send2 = send1.clone()
    receive2 = receive1.clone()
    await send1.aclose()
    await receive1.aclose()
    send2.send_nowait("hello")
    assert receive2.receive_nowait() == "hello"

    send1.close()
    receive1.close()
    send2.close()
    receive2.close()


async def test_clone_closed() -> None:
    send, receive = create_memory_object_stream[NoReturn](1)
    await send.aclose()
    await receive.aclose()
    pytest.raises(ClosedResourceError, send.clone)
    pytest.raises(ClosedResourceError, receive.clone)


async def test_close_send_while_receiving() -> None:
    send, receive = create_memory_object_stream[NoReturn](1)
    with pytest.raises(ExceptionGroup) as exc:
        async with create_task_group() as tg:
            tg.start_soon(receive.receive)
            await wait_all_tasks_blocked()
            await send.aclose()

    assert len(exc.value.exceptions) == 1
    assert isinstance(exc.value.exceptions[0], EndOfStream)

    send.close()
    receive.close()


async def test_close_receive_while_sending() -> None:
    # We send None here as a regression test for #731
    send, receive = create_memory_object_stream[None](0)
    with pytest.raises(ExceptionGroup) as exc:
        async with create_task_group() as tg:
            tg.start_soon(send.send, None)
            await wait_all_tasks_blocked()
            await receive.aclose()

    assert len(exc.value.exceptions) == 1
    assert isinstance(exc.value.exceptions[0], BrokenResourceError)

    send.close()
    receive.close()


async def test_receive_after_send_closed() -> None:
    send, receive = create_memory_object_stream[str](1)
    await send.send("hello")
    await send.aclose()
    assert await receive.receive() == "hello"

    send.close()
    receive.close()


async def test_receive_when_cancelled() -> None:
    """
    Test that calling receive() in a cancelled scope prevents it from going through with
    the operation.

    """
    send, receive = create_memory_object_stream[str]()
    async with create_task_group() as tg:
        tg.start_soon(send.send, "hello")
        await wait_all_tasks_blocked()
        tg.start_soon(send.send, "world")
        await wait_all_tasks_blocked()

        with CancelScope() as scope:
            scope.cancel()
            await receive.receive()

        assert await receive.receive() == "hello"
        assert await receive.receive() == "world"

    send.close()
    receive.close()


async def test_send_when_cancelled() -> None:
    """
    Test that calling send() in a cancelled scope prevents it from going through with
    the operation.

    """

    async def receiver() -> None:
        received.append(await receive.receive())

    received: list[str] = []
    send, receive = create_memory_object_stream[str]()
    async with create_task_group() as tg:
        tg.start_soon(receiver)
        with CancelScope() as scope:
            scope.cancel()
            await send.send("hello")

        await send.send("world")

    assert received == ["world"]

    send.close()
    receive.close()


async def test_cancel_during_receive() -> None:
    """
    Test that cancelling a pending receive() operation does not cause an item in the
    stream to be lost.

    """

    async def scoped_receiver(task_status: TaskStatus[CancelScope]) -> None:
        with CancelScope() as cancel_scope:
            task_status.started(cancel_scope)
            received.append(await receive.receive())

        assert cancel_scope.cancel_called

    received: list[str] = []
    send, receive = create_memory_object_stream[str]()
    with send, receive:
        async with create_task_group() as tg:
            receiver_scope = await tg.start(scoped_receiver)
            await wait_all_tasks_blocked()
            send.send_nowait("hello")
            receiver_scope.cancel()

    assert received == ["hello"]


async def test_cancel_during_receive_buffered() -> None:
    """
    Test that sending an item to a memory object stream when the receiver that is next
    in line has been cancelled will not result in the item being lost.
    """

    async def scoped_receiver(
        receive: MemoryObjectReceiveStream[str], task_status: TaskStatus[CancelScope]
    ) -> None:
        with CancelScope() as cancel_scope:
            task_status.started(cancel_scope)
            await receive.receive()

    send, receive = create_memory_object_stream[str](1)
    with send, receive:
        async with create_task_group() as tg:
            cancel_scope = await tg.start(scoped_receiver, receive)
            await wait_all_tasks_blocked()
            cancel_scope.cancel()
            send.send_nowait("item")

        # Since the item was not sent to the cancelled task, it should be available here
        assert receive.receive_nowait() == "item"


async def test_close_receive_after_send() -> None:
    async def send() -> None:
        async with send_stream:
            await send_stream.send("test")

    async def receive() -> None:
        async with receive_stream:
            assert await receive_stream.receive() == "test"

    send_stream, receive_stream = create_memory_object_stream[str]()
    async with create_task_group() as tg:
        tg.start_soon(send)
        tg.start_soon(receive)

    send_stream.close()
    receive_stream.close()


async def test_statistics() -> None:
    send_stream, receive_stream = create_memory_object_stream[None](1)
    streams: list[MemoryObjectReceiveStream[None] | MemoryObjectSendStream[None]] = [
        send_stream,
        receive_stream,
    ]
    for stream in streams:
        statistics = stream.statistics()
        assert statistics.max_buffer_size == 1
        assert statistics.current_buffer_used == 0
        assert statistics.open_send_streams == 1
        assert statistics.open_receive_streams == 1
        assert statistics.tasks_waiting_send == 0
        assert statistics.tasks_waiting_receive == 0

    for stream in streams:
        async with create_task_group() as tg:
            # Test tasks_waiting_send
            send_stream.send_nowait(None)
            assert stream.statistics().current_buffer_used == 1
            tg.start_soon(send_stream.send, None)
            await wait_all_tasks_blocked()
            assert stream.statistics().current_buffer_used == 1
            assert stream.statistics().tasks_waiting_send == 1
            receive_stream.receive_nowait()
            assert stream.statistics().current_buffer_used == 1
            assert stream.statistics().tasks_waiting_send == 0
            receive_stream.receive_nowait()
            assert stream.statistics().current_buffer_used == 0

            # Test tasks_waiting_receive
            tg.start_soon(receive_stream.receive)
            await wait_all_tasks_blocked()
            assert stream.statistics().tasks_waiting_receive == 1
            send_stream.send_nowait(None)
            assert stream.statistics().tasks_waiting_receive == 0

        async with create_task_group() as tg:
            # Test tasks_waiting_send
            send_stream.send_nowait(None)
            assert stream.statistics().tasks_waiting_send == 0
            for _ in range(3):
                tg.start_soon(send_stream.send, None)

            await wait_all_tasks_blocked()
            assert stream.statistics().tasks_waiting_send == 3
            for i in range(2, -1, -1):
                receive_stream.receive_nowait()
                assert stream.statistics().tasks_waiting_send == i

            receive_stream.receive_nowait()

        assert stream.statistics().current_buffer_used == 0
        assert stream.statistics().tasks_waiting_send == 0
        assert stream.statistics().tasks_waiting_receive == 0

    send_stream.close()
    receive_stream.close()


async def test_sync_close() -> None:
    send_stream, receive_stream = create_memory_object_stream[None](1)
    with send_stream, receive_stream:
        pass

    with pytest.raises(ClosedResourceError):
        send_stream.send_nowait(None)

    with pytest.raises(ClosedResourceError):
        receive_stream.receive_nowait()


async def test_type_variance() -> None:
    """
    This test does not do anything at run time, but since the test suite is also checked
    with a static type checker, it ensures that the memory object stream
    co/contravariance works as intended. If it doesn't, one or both of the following
    reassignments will trip the type checker.

    """
    send, receive = create_memory_object_stream[float]()
    receive1: MemoryObjectReceiveStream[complex] = receive  # noqa: F841
    receive2: ObjectReceiveStream[complex] = receive  # noqa: F841
    send1: MemoryObjectSendStream[int] = send  # noqa: F841
    send2: ObjectSendStream[int] = send  # noqa: F841

    send.close()
    receive.close()


async def test_deprecated_item_type_parameter() -> None:
    with pytest.warns(DeprecationWarning, match="item_type argument has been "):
        send, receive = create_memory_object_stream(item_type=int)  # type: ignore[var-annotated]

        send.close()
        receive.close()


async def test_not_closed_warning() -> None:
    send, receive = create_memory_object_stream[int]()

    with pytest.warns(
        ResourceWarning, match="Unclosed <MemoryObjectSendStream at [0-9a-f]+>"
    ):
        del send
        gc.collect()

    with pytest.warns(
        ResourceWarning, match="Unclosed <MemoryObjectReceiveStream at [0-9a-f]+>"
    ):
        del receive
        gc.collect()


@pytest.mark.parametrize("anyio_backend", asyncio_params)
async def test_send_to_natively_cancelled_receiver() -> None:
    """
    Test that if a task waiting on receive.receive() is cancelled and then another
    task sends an item, said item is not delivered to the task with a pending
    cancellation, but rather to the next one in line.

    """
    from asyncio import CancelledError, create_task

    send, receive = create_memory_object_stream[str](1)
    with send, receive:
        receive_task = create_task(receive.receive())
        await wait_all_tasks_blocked()  # ensure that the task is waiting to receive
        receive_task.cancel()
        send.send_nowait("hello")
        with pytest.raises(CancelledError):
            await receive_task

        assert receive.receive_nowait() == "hello"


async def test_memory_object_item_receiver_repr() -> None:
    """
    Test the repr of `MemoryObjectItemReceiver`.
    Since when `item` is not set, the default dataclass repr raises an AttributeError.
    """
    receiver = _MemoryObjectItemReceiver[str]()

    assert str(receiver) is not None
    receiver_repr = repr(receiver)
    assert "item=None" in receiver_repr

    assert str(receiver) is not None
    receiver.item = "test_item"
    receiver_repr = repr(receiver)
    assert "item='test_item'" in receiver_repr


async def test_receive_any_no_streams() -> None:
    with pytest.raises(ValueError, match="at least one receive stream"):
        await receive_any([])


async def test_receive_any_invalid_type() -> None:
    with pytest.raises(TypeError, match="MemoryObjectReceiveStream"):
        await receive_any([None])  # type: ignore[arg-type]


async def test_receive_any_closed_stream() -> None:
    send, receive = create_memory_object_stream[str]()
    with send, receive:
        receive.close()
        with pytest.raises(ClosedResourceError):
            await receive_any([receive])


async def test_receive_any_duplicate_state() -> None:
    send, receive = create_memory_object_stream[str]()
    with send, receive, receive.clone() as clone:
        with pytest.raises(ValueError, match="same underlying stream state"):
            await receive_any([receive, clone])


async def test_receive_any_first_readable_stream() -> None:
    send1, receive1 = create_memory_object_stream[str](2)
    send2, receive2 = create_memory_object_stream[str](2)
    send3, receive3 = create_memory_object_stream[str](2)
    with send1, receive1, send2, receive2, send3, receive3:
        send2.send_nowait("foo")
        send2.send_nowait("bar")
        send3.send_nowait("baz")

        index, item = await receive_any([receive1, receive2, receive3])
        assert (index, item) == (1, "foo")
        # The other streams must have been left untouched
        assert receive1.statistics().current_buffer_used == 0
        assert receive2.receive_nowait() == "bar"
        assert receive3.receive_nowait() == "baz"


async def test_receive_any_concurrent_senders() -> None:
    send1, receive1 = create_memory_object_stream[str](0)
    send2, receive2 = create_memory_object_stream[str](0)
    senders_ready = Event()
    senders_started = 0

    async def sender(send: MemoryObjectSendStream[str], item: str) -> None:
        nonlocal senders_started
        senders_started += 1
        if senders_started == 2:
            senders_ready.set()

        await send.send(item)

    with send1, receive1, send2, receive2:
        async with create_task_group() as tg:
            tg.start_soon(sender, send1, "one")
            tg.start_soon(sender, send2, "two")
            await senders_ready.wait()
            await wait_all_tasks_blocked()

            index, item = await receive_any([receive1, receive2])
            assert (index, item) == (0, "one")
            # The unselected sender must still be waiting, its item intact
            assert receive2.statistics().tasks_waiting_send == 1
            assert await receive2.receive() == "two"


async def test_receive_any_simultaneous_delivery() -> None:
    send1, receive1 = create_memory_object_stream[str](0)
    send2, receive2 = create_memory_object_stream[str](0)
    results: list[tuple[int, str]] = []

    async def receiver() -> None:
        results.append(await receive_any([receive1, receive2]))

    with send1, receive1, send2, receive2:
        async with create_task_group() as tg:
            tg.start_soon(receiver)
            await wait_all_tasks_blocked()
            # Deliver items on both streams before the waiting task runs again
            send1.send_nowait("a")
            send2.send_nowait("b")

        assert results == [(0, "a")]
        # The item that lost the race must have been reclaimed by its stream
        assert receive2.receive_nowait() == "b"
        assert receive1.statistics().tasks_waiting_receive == 0
        assert receive2.statistics().tasks_waiting_receive == 0


async def test_receive_any_cancelled() -> None:
    send1, receive1 = create_memory_object_stream[str](0)
    send2, receive2 = create_memory_object_stream[str](0)
    with send1, receive1, send2, receive2:
        with CancelScope() as scope:
            async with create_task_group() as tg:
                tg.start_soon(receive_any, [receive1, receive2])
                await wait_all_tasks_blocked()
                assert receive1.statistics().tasks_waiting_receive == 1
                assert receive2.statistics().tasks_waiting_receive == 1
                scope.cancel()

        assert receive1.statistics().tasks_waiting_receive == 0
        assert receive2.statistics().tasks_waiting_receive == 0


async def test_receive_any_respects_waiting_receivers() -> None:
    send1, receive1 = create_memory_object_stream[str](0)
    send2, receive2 = create_memory_object_stream[str](0)
    received: list[str] = []

    async def normal_receiver() -> None:
        received.append(await receive1.receive())

    with send1, receive1, send2, receive2:
        async with create_task_group() as tg:
            tg.start_soon(normal_receiver)
            await wait_all_tasks_blocked()
            with CancelScope() as scope:
                async with create_task_group() as tg2:
                    tg2.start_soon(receive_any, [receive1, receive2])
                    await wait_all_tasks_blocked()
                    # Must go to the task already waiting on receive()
                    send1.send_nowait("hello")
                    scope.cancel()

            await wait_all_tasks_blocked()
            assert received == ["hello"]
            assert receive1.statistics().tasks_waiting_receive == 0
            assert receive2.statistics().tasks_waiting_receive == 0


async def test_receive_any_partial_eof() -> None:
    send1, receive1 = create_memory_object_stream[str](0)
    send2, receive2 = create_memory_object_stream[str](0)
    results: list[tuple[int, str]] = []

    async def receiver() -> None:
        results.append(await receive_any([receive1, receive2]))

    with send1, receive1, send2, receive2:
        send1.close()
        async with create_task_group() as tg:
            tg.start_soon(receiver)
            await wait_all_tasks_blocked()
            assert receive2.statistics().tasks_waiting_receive == 1
            await send2.send("hello")

        assert results == [(1, "hello")]
        assert receive1.statistics().tasks_waiting_receive == 0
        assert receive2.statistics().tasks_waiting_receive == 0


async def test_receive_any_all_eof() -> None:
    send1, receive1 = create_memory_object_stream[str](1)
    send2, receive2 = create_memory_object_stream[str](1)
    with send1, receive1, send2, receive2:
        send1.close()
        send2.close()
        with pytest.raises(EndOfStream):
            await receive_any([receive1, receive2])


async def test_receive_any_all_eof_while_waiting() -> None:
    send1, receive1 = create_memory_object_stream[str](0)
    send2, receive2 = create_memory_object_stream[str](0)

    async def receiver() -> None:
        with pytest.raises(EndOfStream):
            await receive_any([receive1, receive2])

    with send1, receive1, send2, receive2:
        async with create_task_group() as tg:
            tg.start_soon(receiver)
            await wait_all_tasks_blocked()
            send1.close()
            await wait_all_tasks_blocked()
            # The remaining stream is still being waited on
            assert receive2.statistics().tasks_waiting_receive == 1
            send2.close()

        assert receive1.statistics().tasks_waiting_receive == 0
        assert receive2.statistics().tasks_waiting_receive == 0


async def test_receive_any_closed_clone() -> None:
    send1, receive1 = create_memory_object_stream[str](1)
    send2, receive2 = create_memory_object_stream[str](1)
    with send1, receive1, send2, receive2, receive1.clone() as clone:
        clone.close()
        send2.send_nowait("hello")
        index, item = await receive_any([receive1, receive2])
        assert (index, item) == (1, "hello")
