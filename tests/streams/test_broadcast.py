from __future__ import annotations

import pytest

from anyio import (
    CancelScope,
    ClosedResourceError,
    EndOfStream,
    Event,
    WouldBlock,
    create_task_group,
    wait_all_tasks_blocked,
)
from anyio.streams.broadcast import (
    BroadcastReceiveStream,
    BroadcastStream,
    BroadcastStreamStatistics,
)

pytestmark = pytest.mark.anyio


class _Barrier:
    """A one-shot barrier that releases all parties at the same time."""

    def __init__(self, parties: int) -> None:
        self._parties = parties
        self._count = 0
        self._event = Event()

    async def wait(self) -> None:
        self._count += 1
        if self._count == self._parties:
            self._event.set()

        await self._event.wait()


def test_invalid_max_buffer_size() -> None:
    with BroadcastStream[str]() as hub:
        pytest.raises(ValueError, hub.subscribe, 0).match("must be at least 1")
        pytest.raises(ValueError, hub.subscribe, -2).match("must be at least 1")
        with pytest.raises(TypeError, match="must be an integer"):
            hub.subscribe(1.5)  # type: ignore[arg-type]


async def test_publish_reaches_all_subscriptions() -> None:
    with BroadcastStream[str]() as hub:
        sub1 = hub.subscribe(1)
        sub2 = hub.subscribe(1)
        await hub.publish("x")
        assert sub1.receive_nowait() == "x"
        assert sub2.receive_nowait() == "x"
        sub1.close()
        sub2.close()


async def test_publish_blocks_until_all_subscriptions_have_room() -> None:
    """
    Two subscriptions with capacity 1 both receive an item. After clearing only
    one of them, the next publish must block, and cancelling it must guarantee
    that neither subscription receives the item.
    """
    with BroadcastStream[str]() as hub:
        sub1 = hub.subscribe(1)
        sub2 = hub.subscribe(1)
        await hub.publish("x")

        # Clear only the first subscription
        assert sub1.receive_nowait() == "x"

        # Publishing must now block because the second buffer is still full
        publish_scope: CancelScope | None = None
        async with create_task_group() as tg:

            async def publisher() -> None:
                nonlocal publish_scope
                with CancelScope() as scope:
                    publish_scope = scope
                    await hub.publish("y")

            tg.start_soon(publisher)
            await wait_all_tasks_blocked()
            assert hub.statistics().tasks_waiting_publish == 1
            assert publish_scope is not None
            publish_scope.cancel()

        # Clear the second subscription; it still holds "x", not "y"
        assert sub2.receive_nowait() == "x"

        # Neither subscription may have received the cancelled item
        pytest.raises(WouldBlock, sub1.receive_nowait)
        pytest.raises(WouldBlock, sub2.receive_nowait)

        # The stream must still be usable afterwards
        await hub.publish("z")
        assert sub1.receive_nowait() == "z"
        assert sub2.receive_nowait() == "z"
        sub1.close()
        sub2.close()


async def test_cancel_racing_with_freed_capacity() -> None:
    """
    Cancelling a publish at the same moment a subscription frees buffer capacity
    must still guarantee that no subscription receives the item.
    """
    with BroadcastStream[str]() as hub:
        sub1 = hub.subscribe(1)
        sub2 = hub.subscribe(1)
        await hub.publish("x")

        barrier = _Barrier(2)
        publish_scope: CancelScope | None = None
        async with create_task_group() as tg:

            async def publisher() -> None:
                nonlocal publish_scope
                with CancelScope() as scope:
                    publish_scope = scope
                    await hub.publish("y")

            async def freer() -> None:
                await barrier.wait()
                # Free one slot, waking the publisher at the same moment it is
                # cancelled
                assert sub1.receive_nowait() == "x"

            async def canceller() -> None:
                await barrier.wait()
                assert publish_scope is not None
                publish_scope.cancel()

            tg.start_soon(publisher)
            await wait_all_tasks_blocked()
            assert hub.statistics().tasks_waiting_publish == 1
            tg.start_soon(freer)
            tg.start_soon(canceller)

        # The cancelled publish must not have reached any subscription
        assert sub2.receive_nowait() == "x"
        pytest.raises(WouldBlock, sub1.receive_nowait)
        pytest.raises(WouldBlock, sub2.receive_nowait)
        sub1.close()
        sub2.close()


async def test_publish_with_no_subscriptions() -> None:
    """Publishing with zero live subscriptions succeeds immediately as a no-op."""
    with BroadcastStream[str]() as hub:
        await hub.publish("x")
        hub.publish_nowait("y")

        # A subscription created afterwards does not receive earlier items
        sub = hub.subscribe(1)
        pytest.raises(WouldBlock, sub.receive_nowait)
        await hub.publish("z")
        assert sub.receive_nowait() == "z"

        # Closing the last subscription brings the count back to zero
        sub.close()
        assert hub.statistics().open_subscriptions == 0
        await hub.publish("discarded")


async def test_closing_subscription_wakes_publisher() -> None:
    with BroadcastStream[str]() as hub:
        sub = hub.subscribe(1)
        await hub.publish("x")

        async with create_task_group() as tg:
            tg.start_soon(hub.publish, "y")
            await wait_all_tasks_blocked()
            assert hub.statistics().tasks_waiting_publish == 1

            # Closing the subscription must wake the publisher, whose publish now
            # succeeds as a no-op since there are no live subscriptions left
            sub.close()

        assert hub.statistics().tasks_waiting_publish == 0
        pytest.raises(ClosedResourceError, sub.receive_nowait)


async def test_close_lets_receivers_drain_buffer() -> None:
    hub = BroadcastStream[int]()
    sub1 = hub.subscribe(3)
    sub2 = hub.subscribe(2)
    await hub.publish(1)
    await hub.publish(2)
    hub.close()
    hub.close()  # repeated close is a no-op

    # Publishing and subscribing after close must fail
    pytest.raises(ClosedResourceError, hub.publish_nowait, 3)
    with pytest.raises(ClosedResourceError):
        await hub.publish(3)

    pytest.raises(ClosedResourceError, hub.subscribe, 1)

    # Receivers drain the queued items first, then get EndOfStream
    assert sub1.receive_nowait() == 1
    assert await sub2.receive() == 1
    assert sub1.receive_nowait() == 2
    assert await sub2.receive() == 2
    with pytest.raises(EndOfStream):
        await sub1.receive()

    with pytest.raises(EndOfStream):
        await sub2.receive()

    sub1.close()
    sub2.close()
    sub2.close()  # repeated close is a no-op


async def test_close_wakes_waiting_publisher() -> None:
    hub = BroadcastStream[str]()
    sub = hub.subscribe(1)
    await hub.publish("x")
    async with create_task_group() as tg:

        async def publisher() -> None:
            with pytest.raises(ClosedResourceError):
                await hub.publish("y")

        tg.start_soon(publisher)
        await wait_all_tasks_blocked()
        assert hub.statistics().tasks_waiting_publish == 1
        hub.close()

    sub.close()


async def test_close_wakes_waiting_receiver() -> None:
    hub = BroadcastStream[str]()
    sub = hub.subscribe(1)
    async with create_task_group() as tg:

        async def receiver() -> None:
            with pytest.raises(EndOfStream):
                await sub.receive()

        tg.start_soon(receiver)
        await wait_all_tasks_blocked()
        assert hub.statistics().tasks_waiting_receive == 1
        hub.close()

    sub.close()


async def test_clone_creates_new_independent_subscription() -> None:
    with BroadcastStream[str]() as hub:
        sub = hub.subscribe(2)
        await hub.publish("before")
        clone = sub.clone()
        assert isinstance(clone, BroadcastReceiveStream)

        # The clone is a new subscription: it does not see items published before
        # it was created
        pytest.raises(WouldBlock, clone.receive_nowait)

        # Both the original and the clone receive subsequently published items
        await hub.publish("after")
        assert sub.receive_nowait() == "before"
        assert sub.receive_nowait() == "after"
        assert clone.receive_nowait() == "after"
        pytest.raises(WouldBlock, clone.receive_nowait)

        # Closing the original subscription does not affect the clone
        sub.close()
        pytest.raises(ClosedResourceError, sub.clone)
        await hub.publish("last")
        assert clone.receive_nowait() == "last"
        clone.close()


async def test_clone_after_hub_close() -> None:
    hub = BroadcastStream[str]()
    sub = hub.subscribe(1)
    hub.close()
    pytest.raises(ClosedResourceError, sub.clone)
    sub.close()


async def test_publish_nowait() -> None:
    with BroadcastStream[str]() as hub:
        sub1 = hub.subscribe(1)
        sub2 = hub.subscribe(2)
        hub.publish_nowait("x")
        pytest.raises(WouldBlock, hub.publish_nowait, "y")  # sub1 is full
        assert sub1.receive_nowait() == "x"
        hub.publish_nowait("y")
        assert sub1.receive_nowait() == "y"
        assert sub2.receive_nowait() == "x"
        assert sub2.receive_nowait() == "y"
        sub1.close()
        sub2.close()


async def test_fifo_order_per_subscription() -> None:
    with BroadcastStream[int]() as hub:
        sub = hub.subscribe(3)
        for item in range(3):
            await hub.publish(item)

        assert [sub.receive_nowait() for _ in range(3)] == [0, 1, 2]
        sub.close()


async def test_receive_after_close() -> None:
    with BroadcastStream[str]() as hub:
        sub = hub.subscribe(1)
        sub.close()
        sub.close()  # repeated close is a no-op
        pytest.raises(ClosedResourceError, sub.receive_nowait)
        with pytest.raises(ClosedResourceError):
            await sub.receive()


async def test_statistics() -> None:
    with BroadcastStream[str]() as hub:
        assert hub.statistics() == BroadcastStreamStatistics(0, 0, 0)
        sub1 = hub.subscribe(1)
        sub2 = hub.subscribe(1)
        assert hub.statistics().open_subscriptions == 2

        await hub.publish("x")  # fills both buffers
        async with create_task_group() as tg:
            tg.start_soon(hub.publish, "y")
            await wait_all_tasks_blocked()
            assert hub.statistics().tasks_waiting_publish == 1
            assert sub1.receive_nowait() == "x"  # not enough: sub2 is still full
            assert sub2.receive_nowait() == "x"  # now the publisher can proceed

        assert hub.statistics().tasks_waiting_publish == 0
        assert sub1.receive_nowait() == "y"
        assert sub2.receive_nowait() == "y"

        async with create_task_group() as tg:
            tg.start_soon(sub1.receive)
            await wait_all_tasks_blocked()
            assert hub.statistics().tasks_waiting_receive == 1
            tg.cancel_scope.cancel()

        assert hub.statistics().tasks_waiting_receive == 0
        sub1.close()
        assert hub.statistics().open_subscriptions == 1
        sub2.close()


async def test_async_iteration() -> None:
    hub = BroadcastStream[int]()
    sub = hub.subscribe(2)
    received: list[int] = []
    async with create_task_group() as tg:

        async def consumer() -> None:
            received.extend([item async for item in sub])

        tg.start_soon(consumer)
        await wait_all_tasks_blocked()
        for item in range(3):
            await hub.publish(item)

        hub.close()

    assert received == [0, 1, 2]
    sub.close()


async def test_context_managers() -> None:
    with BroadcastStream[int]() as hub:
        with hub.subscribe(1) as sub:
            await hub.publish(1)
            assert sub.receive_nowait() == 1

    pytest.raises(ClosedResourceError, hub.publish_nowait, 2)
    pytest.raises(ClosedResourceError, sub.receive_nowait)


async def test_async_context_managers() -> None:
    async with BroadcastStream[int]() as hub:
        async with hub.subscribe(1) as sub:
            await hub.publish(1)
            assert sub.receive_nowait() == 1

    pytest.raises(ClosedResourceError, hub.publish_nowait, 2)
    pytest.raises(ClosedResourceError, sub.receive_nowait)
