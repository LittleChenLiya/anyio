from __future__ import annotations

__all__ = (
    "BroadcastObjectReceiveStream",
    "BroadcastObjectSendStream",
)

import warnings
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from types import TracebackType
from typing import Generic, TypeVar

from .._core._exceptions import (
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    WouldBlock,
)
from .._core._synchronization import Event
from ..abc import ObjectReceiveStream, ObjectSendStream
from ..lowlevel import checkpoint_if_cancelled

T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


@dataclass(eq=False)
class _BroadcastSubscription(Generic[T_co]):
    capacity: int
    buffer: deque[T_co] = field(init=False, default_factory=deque)
    open_receive_channels: int = field(init=False, default=0)
    waiting_receivers: OrderedDict[Event, None] = field(
        init=False, default_factory=OrderedDict
    )


@dataclass(eq=False)
class _BroadcastState(Generic[T_co]):
    subscriptions: OrderedDict[_BroadcastSubscription[T_co], None] = field(
        init=False, default_factory=OrderedDict
    )
    open_send_channels: int = field(init=False, default=0)
    waiting_publishers: OrderedDict[Event, None] = field(
        init=False, default_factory=OrderedDict
    )

    @property
    def send_closed(self) -> bool:
        return self.open_send_channels == 0

    def _wake_publishers(self) -> None:
        for event in self.waiting_publishers:
            event.set()


@dataclass(eq=False)
class BroadcastObjectReceiveStream(ObjectReceiveStream[T_co], Generic[T_co]):
    """
    A subscription on a :class:`~anyio.BroadcastObjectSendStream`.

    Every clone of a subscription shares its buffer and capacity, so items
    delivered to the subscription are distributed between the clones (as with a
    memory object stream). Cloning a subscription never creates a new
    subscription; use :meth:`subscribe` to add another subscriber.

    """

    _state: _BroadcastState[T_co]
    _subscription: _BroadcastSubscription[T_co]
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._subscription.open_receive_channels += 1

    def receive_nowait(self) -> T_co:
        """
        Receive the next published item without waiting.

        Only items published after the subscription was created can be received.

        :return: the received item
        :raises ~anyio.ClosedResourceError: if this subscription has been closed
        :raises ~anyio.EndOfStream: if the buffer is empty and all clones of the
            broadcast send stream have been closed
        :raises ~anyio.WouldBlock: if there are no items in the buffer and the
            send stream is still open

        """
        if self._closed:
            raise ClosedResourceError

        if self._subscription.buffer:
            item = self._subscription.buffer.popleft()
            # Freeing a slot may unblock one or more waiting publishers.
            self._state._wake_publishers()
            return item
        elif self._state.send_closed:
            raise EndOfStream

        raise WouldBlock

    async def receive(self) -> T_co:
        await checkpoint_if_cancelled()
        while True:
            try:
                return self.receive_nowait()
            except WouldBlock:
                receive_event = Event()
                self._subscription.waiting_receivers[receive_event] = None
                try:
                    await receive_event.wait()
                finally:
                    self._subscription.waiting_receivers.pop(receive_event, None)

    def clone(self) -> BroadcastObjectReceiveStream[T_co]:
        """
        Create a clone of this receive stream.

        A clone shares the subscription (including its buffer and capacity) with
        the original; it does not create a new subscription. Each clone can be
        closed separately, and the subscription is only removed from the
        broadcast once all of its clones have been closed.

        :return: the cloned stream

        """
        if self._closed:
            raise ClosedResourceError

        return BroadcastObjectReceiveStream(
            _state=self._state, _subscription=self._subscription
        )

    def close(self) -> None:
        """
        Close the stream.

        This works the exact same way as :meth:`aclose`, but is provided as a
        special case for the benefit of synchronous callbacks.

        """
        if not self._closed:
            self._closed = True
            self._subscription.open_receive_channels -= 1
            if self._subscription.open_receive_channels == 0:
                # Remove the subscription and wake any publishers waiting on it.
                self._state.subscriptions.pop(self._subscription, None)
                receive_events = list(self._subscription.waiting_receivers)
                for event in receive_events:
                    event.set()

                self._state._wake_publishers()

    async def aclose(self) -> None:
        self.close()

    def __enter__(self) -> BroadcastObjectReceiveStream[T_co]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        if not self._closed:
            warnings.warn(
                f"Unclosed <{self.__class__.__name__} at {id(self):x}>",
                ResourceWarning,
                stacklevel=1,
                source=self,
            )


@dataclass(eq=False)
class BroadcastObjectSendStream(ObjectSendStream[T_contra], Generic[T_contra]):
    """
    A send stream that broadcasts every published item to every live
    subscription.

    Unlike memory object streams, items are not distributed between receivers:
    each subscription receives its own copy of every item published while it is
    alive. Items published before a subscription was created are not delivered.

    """

    _state: _BroadcastState[T_contra]
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._state.open_send_channels += 1

    def subscribe(self, capacity: int) -> BroadcastObjectReceiveStream[T_contra]:
        """
        Create a new subscription.

        The subscription will only receive items published after it was created.
        It has its own bounded buffer that can hold ``capacity`` items.

        :param capacity: maximum number of published items that can be buffered
            for this subscription (must be a positive integer)
        :return: a receive stream for the new subscription
        :raises ~anyio.ClosedResourceError: if this broadcast send stream has
            been closed
        :raises ValueError: if capacity is not a positive integer

        """
        if self._closed:
            raise ClosedResourceError
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")

        subscription = _BroadcastSubscription[T_contra](capacity)
        self._state.subscriptions[subscription] = None
        return BroadcastObjectReceiveStream(
            _state=self._state, _subscription=subscription
        )

    def publish_nowait(self, item: T_contra) -> None:
        """
        Publish an item immediately if every live subscription has room for it.

        Delivery is all-or-nothing: the item is appended to every live
        subscription buffer in a single indivisible step, or not at all.

        :param item: the item to publish
        :raises ~anyio.ClosedResourceError: if this broadcast send stream has
            been closed
        :raises ~anyio.BrokenResourceError: if there are no live subscriptions
        :raises ~anyio.WouldBlock: if at least one live subscription has a full
            buffer

        """
        if self._closed:
            raise ClosedResourceError
        if not self._state.subscriptions:
            raise BrokenResourceError
        if any(
            len(subscription.buffer) >= subscription.capacity
            for subscription in self._state.subscriptions
        ):
            raise WouldBlock

        # Linearization point of a successful publish: the item is appended to
        # every live subscription buffer below without any suspension point, so
        # the broadcast either happens to all of them or to none of them.
        for subscription in self._state.subscriptions:
            subscription.buffer.append(item)
            receive_events = list(subscription.waiting_receivers)
            for event in receive_events:
                event.set()

    async def publish(self, item: T_contra) -> None:
        """
        Broadcast an item to every live subscription.

        If any live subscription buffer is full, this method blocks until every
        live subscription has room, the broadcast is cancelled, a subscription
        is closed or this send stream is closed.

        Delivery is atomic: cancellation while waiting guarantees that no
        subscription receives the item. The linearization point of a successful
        publish is the synchronous section that re-checks the free space of
        every live subscription and appends the item to every subscription
        buffer without any suspension point in between.

        :param item: the item to publish
        :raises ~anyio.ClosedResourceError: if this broadcast send stream has
            been closed
        :raises ~anyio.BrokenResourceError: if there are no live subscriptions

        """
        await checkpoint_if_cancelled()
        while True:
            # Each iteration uses a fresh event: an edge-triggered Event that
            # was set by a wake-up we could not act on (e.g. another
            # subscription was still full) cannot be waited on again.
            publish_event = Event()
            try:
                self.publish_nowait(item)
                return
            except WouldBlock:
                self._state.waiting_publishers[publish_event] = None
                try:
                    await publish_event.wait()
                except BaseException:
                    self._state.waiting_publishers.pop(publish_event, None)
                    raise

                self._state.waiting_publishers.pop(publish_event, None)

                if self._closed:
                    raise ClosedResourceError from None
                if not self._state.subscriptions:
                    raise BrokenResourceError from None

    # Aliases making the broadcaster usable wherever an ObjectSendStream is
    # expected. Cancelling one of these coroutines cancels the whole broadcast.
    send = publish

    def clone(self) -> BroadcastObjectSendStream[T_contra]:
        """
        Create a clone of this broadcast send stream.

        Each clone can be closed separately. Only once all clones have been
        closed will the subscribers be notified of the end of the stream (after
        draining their buffers).

        :return: the cloned stream

        """
        if self._closed:
            raise ClosedResourceError

        return BroadcastObjectSendStream(_state=self._state)

    def close(self) -> None:
        """
        Close the stream.

        This works the exact same way as :meth:`aclose`, but is provided as a
        special case for the benefit of synchronous callbacks. Once all clones
        have been closed, subscribers receive
        :class:`~anyio.EndOfStream` after draining their already enqueued
        items.

        """
        if not self._closed:
            self._closed = True
            self._state.open_send_channels -= 1
            if self._state.open_send_channels == 0:
                publish_events = list(self._state.waiting_publishers)
                self._state.waiting_publishers.clear()
                for event in publish_events:
                    event.set()

                for subscription in self._state.subscriptions:
                    receive_events = list(subscription.waiting_receivers)
                    for event in receive_events:
                        event.set()

    async def aclose(self) -> None:
        self.close()

    def __enter__(self) -> BroadcastObjectSendStream[T_contra]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        if not self._closed:
            warnings.warn(
                f"Unclosed <{self.__class__.__name__} at {id(self):x}>",
                ResourceWarning,
                stacklevel=1,
                source=self,
            )
