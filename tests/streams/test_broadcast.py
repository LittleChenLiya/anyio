from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from anyio import (
    BrokenResourceError,
    CancelScope,
    ClosedResourceError,
    EndOfStream,
    Event,
    WouldBlock,
    create_broadcast_object_stream,
    create_task_group,
    fail_after,
    wait_all_tasks_blocked,
)

if TYPE_CHECKING:
    from anyio.abc import ObjectReceiveStream, ObjectSendStream
    from anyio.streams.broadcast import (
        BroadcastObjectReceiveStream,
    )

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


class Barrier:
    """One-shot barrier used to release several tasks at the same time."""

    def __init__(self, parties: int) -> None:
        self._parties = parties
        self._waiting = 0
        self._event = Event()

    async def wait(self) -> None:
        self._waiting += 1
        if self._waiting == self._parties:
            self._event.set()

        await self._event.wait()


def drain_nowait(stream: BroadcastObjectReceiveStream[str]) -> list[str]:
    items: list[str] = []
    while True:
        try:
            items.append(stream.receive_nowait())
        except WouldBlock:
            return items


def test_invalid_capacity() -> None:
    publisher = create_broadcast_object_stream[str]()
    for capacity in (0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            publisher.subscribe(capacity)  # type: ignore[arg-type]

    publisher.close()


async def test_broadcast_to_all_subscriptions() -> None:
    publisher = create_broadcast_object_stream[str]()
    first = publisher.subscribe(1)
    second = publisher.subscribe(2)
    await publisher.publish("x")

    assert first.receive_nowait() == "x"
    assert second.receive_nowait() == "x"

    # A subscription only receives items published after it was created.
    await publisher.publish("w")
    assert first.receive_nowait() == "w"
    third = publisher.subscribe(1)
    await publisher.publish("y")
    assert second.receive_nowait() == "w"
    assert second.receive_nowait() == "y"
    assert first.receive_nowait() == "y"
    assert third.receive_nowait() == "y"
    with pytest.raises(WouldBlock):
        first.receive_nowait()
    with pytest.raises(WouldBlock):
        third.receive_nowait()

    publisher.close()
    first.close()
    second.close()
    third.close()


async def test_publish_blocks_until_every_subscription_has_room() -> None:
    publisher = create_broadcast_object_stream[str]()
    first = publisher.subscribe(1)
    second = publisher.subscribe(2)
    await publisher.publish("a")

    # second still has room, but first is full, so publishing blocks.
    async with create_task_group() as tg:
        tg.start_soon(publisher.publish, "b")
        await wait_all_tasks_blocked()
        assert first.receive_nowait() == "a"
        with fail_after(1):
            assert await second.receive() == "a"
            assert await second.receive() == "b"

        assert first.receive_nowait() == "b"

    publisher.close()
    first.close()
    second.close()


async def test_cancel_blocked_publish_is_atomic() -> None:
    """The headline acceptance scenario: cancel a blocked publish mid-broadcast."""
    publisher = create_broadcast_object_stream[str]()
    first = publisher.subscribe(1)
    second = publisher.subscribe(1)
    await publisher.publish("x")

    # Only one of the two buffers is drained.
    assert first.receive_nowait() == "x"

    with CancelScope() as scope:
        async with create_task_group() as tg:
            tg.start_soon(publisher.publish, "y")
            await wait_all_tasks_blocked()
            scope.cancel()

    # Drain the other subscription. Neither side may have received y.
    assert second.receive_nowait() == "x"
    assert drain_nowait(first) == []
    assert drain_nowait(second) == []

    # The publisher is still usable after the cancelled broadcast.
    await publisher.publish("z")
    assert first.receive_nowait() == "z"
    assert second.receive_nowait() == "z"

    publisher.close()
    first.close()
    second.close()


async def test_publish_nowait_atomicity() -> None:
    publisher = create_broadcast_object_stream[str]()
    first = publisher.subscribe(1)
    second = publisher.subscribe(1)
    publisher.publish_nowait("x")
    with pytest.raises(WouldBlock):
        publisher.publish_nowait("y")

    assert drain_nowait(first) == ["x"]
    assert drain_nowait(second) == ["x"]

    publisher.close()
    first.close()
    second.close()


async def test_closing_one_subscription_wakes_publisher() -> None:
    """A publisher blocked on a full subscription is woken when it closes."""
    publisher = create_broadcast_object_stream[str]()
    first = publisher.subscribe(1)
    second = publisher.subscribe(1)
    await publisher.publish("x")

    async with create_task_group() as tg:
        tg.start_soon(publisher.publish, "y")
        await wait_all_tasks_blocked()

        # Closing the full first subscription wakes the publisher, but the
        # still-full second subscription makes it keep waiting.
        first.close()
        await wait_all_tasks_blocked()

        # Freeing the last live subscription completes the broadcast.
        assert await second.receive() == "x"
        assert await second.receive() == "y"

    assert drain_nowait(second) == []

    publisher.close()
    second.close()


async def test_closing_last_subscription_breaks_publisher() -> None:
    publisher = create_broadcast_object_stream[str]()
    first = publisher.subscribe(1)
    second = publisher.subscribe(1)
    await publisher.publish("x")

    with pytest.raises(ExceptionGroup) as exc_info:
        async with create_task_group() as tg:
            tg.start_soon(publisher.publish, "y")
            await wait_all_tasks_blocked()
            first.close()
            await wait_all_tasks_blocked()
            second.close()

    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], BrokenResourceError)

    publisher.close()


async def test_closing_publisher_wakes_waiting_publisher() -> None:
    publisher = create_broadcast_object_stream[str]()
    first = publisher.subscribe(1)
    second = publisher.subscribe(1)
    await publisher.publish("x")
    clone = publisher.clone()

    with pytest.raises(ExceptionGroup) as exc_info:
        async with create_task_group() as tg:
            tg.start_soon(publisher.publish, "y")
            await wait_all_tasks_blocked()
            publisher.close()
            clone.close()

    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], ClosedResourceError)

    first.close()
    second.close()


async def test_subscribers_drain_before_end_of_stream() -> None:
    publisher = create_broadcast_object_stream[str]()
    first = publisher.subscribe(2)
    second = publisher.subscribe(2)
    await publisher.publish("a")
    await publisher.publish("b")

    publisher.close()
    with pytest.raises(ClosedResourceError):
        publisher.subscribe(1)

    assert [item async for item in first] == ["a", "b"]
    assert [item async for item in second] == ["a", "b"]

    first.close()
    second.close()


async def test_end_of_stream_after_draining_full_subscription() -> None:
    publisher = create_broadcast_object_stream[str]()
    subscription = publisher.subscribe(1)
    await publisher.publish("a")
    publisher.close()

    assert await subscription.receive() == "a"
    with pytest.raises(EndOfStream):
        await subscription.receive()
    with pytest.raises(EndOfStream):
        subscription.receive_nowait()

    subscription.close()


async def test_zero_subscribers() -> None:
    publisher = create_broadcast_object_stream[str]()
    with pytest.raises(BrokenResourceError):
        publisher.publish_nowait("x")
    with pytest.raises(BrokenResourceError):
        await publisher.publish("x")

    first = publisher.subscribe(1)
    first.close()
    with pytest.raises(BrokenResourceError):
        await publisher.publish("x")

    publisher.close()


async def test_repeated_close_is_idempotent() -> None:
    publisher = create_broadcast_object_stream[str]()
    first = publisher.subscribe(1)
    publisher.close()
    publisher.close()
    await publisher.aclose()
    first.close()
    first.close()
    await first.aclose()


async def test_close_subscription_while_receiving() -> None:
    publisher = create_broadcast_object_stream[str]()
    first = publisher.subscribe(1)

    with pytest.raises(ExceptionGroup) as exc_info:
        async with create_task_group() as tg:
            tg.start_soon(first.receive)
            await wait_all_tasks_blocked()
            first.close()

    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], ClosedResourceError)

    publisher.close()


async def test_receive_clone_shares_subscription() -> None:
    publisher = create_broadcast_object_stream[str]()
    subscription = publisher.subscribe(2)
    clone = subscription.clone()
    other = publisher.subscribe(2)
    await publisher.publish("x")

    # Publish another item so both readers of the shared subscription get one.
    await publisher.publish("w")

    # The clone competes for the same buffer; it is not a new subscription.
    received: list[str] = []

    async def reader(stream: BroadcastObjectReceiveStream[str]) -> None:
        received.append(await stream.receive())

    async with create_task_group() as tg:
        tg.start_soon(reader, subscription)
        await wait_all_tasks_blocked()
        tg.start_soon(reader, clone)
        await wait_all_tasks_blocked()

    assert sorted(received) == ["w", "x"]
    assert other.receive_nowait() == "x"
    assert other.receive_nowait() == "w"
    with pytest.raises(WouldBlock):
        subscription.receive_nowait()

    # Closing one clone keeps the shared subscription alive.
    clone.close()
    await publisher.publish("y")
    assert subscription.receive_nowait() == "y"

    publisher.close()
    subscription.close()
    other.close()


async def test_clone_closed_publisher() -> None:
    publisher = create_broadcast_object_stream[str]()
    await publisher.aclose()
    with pytest.raises(ClosedResourceError):
        publisher.clone()


@pytest.mark.parametrize("iteration", range(10))
async def test_cancel_races_with_free_slots(iteration: int) -> None:
    """
    Cancellation and both buffers freeing up at the same instant must never
    produce a half-finished broadcast: y ends up in both subscriptions or in
    neither of them.

    """
    publisher = create_broadcast_object_stream[str]()
    first = publisher.subscribe(1)
    second = publisher.subscribe(1)
    await publisher.publish("x")

    barrier = Barrier(3)
    delivered: list[bool] = []

    async def blocked_publisher(task_status: object) -> None:
        with CancelScope() as scope:
            task_status.started(scope)  # type: ignore[attr-defined]
            await publisher.publish("y")
            delivered.append(True)

    async def cancel_party(scope: CancelScope) -> None:
        await barrier.wait()
        scope.cancel()

    async def drain_party(stream: BroadcastObjectReceiveStream[str]) -> None:
        await barrier.wait()
        assert stream.receive_nowait() == "x"

    async with create_task_group() as tg:
        scope = await tg.start(blocked_publisher)
        tg.start_soon(cancel_party, scope)
        tg.start_soon(drain_party, first)
        tg.start_soon(drain_party, second)

    first_items = drain_nowait(first)
    second_items = drain_nowait(second)
    assert ("y" in first_items) is ("y" in second_items)
    assert delivered == ([True] if "y" in first_items else [])

    publisher.close()
    first.close()
    second.close()


async def test_type_variance() -> None:
    publisher = create_broadcast_object_stream[int]()
    send_stream: ObjectSendStream[int] = publisher
    subscription = publisher.subscribe(1)
    subscription_stream: ObjectReceiveStream[int] = subscription

    assert send_stream is publisher
    assert subscription_stream is subscription
    publisher.close()
    subscription.close()
