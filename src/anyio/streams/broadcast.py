from __future__ import annotations

__all__ = (
    "BroadcastReceiveStream",
    "BroadcastStream",
    "BroadcastStreamStatistics",
)

import warnings
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from types import TracebackType
from typing import Generic, NamedTuple, TypeVar

from .._core._exceptions import ClosedResourceError, EndOfStream, WouldBlock
from .._core._synchronization import Event
from ..abc import ObjectReceiveStream
from ..lowlevel import checkpoint

T_Item = TypeVar("T_Item")
T_co = TypeVar("T_co", covariant=True)


class BroadcastStreamStatistics(NamedTuple):
    open_subscriptions: int  #: number of unclosed subscriptions
    #: number of tasks blocked on :meth:`BroadcastStream.publish`
    tasks_waiting_publish: int
    #: number of tasks blocked on :meth:`BroadcastReceiveStream.receive`
    tasks_waiting_receive: int


@dataclass(eq=False)
class _BroadcastSubscription(Generic[T_Item]):
    max_buffer_size: int
    buffer: deque[T_Item] = field(init=False, default_factory=deque)
    waiting_receivers: OrderedDict[Event, None] = field(
        init=False, default_factory=OrderedDict
    )


@dataclass(eq=False)
class _BroadcastStreamState(Generic[T_Item]):
    subscriptions: list[_BroadcastSubscription[T_Item]] = field(
        init=False, default_factory=list
    )
    closed: bool = field(init=False, default=False)
    waiting_publishers: OrderedDict[Event, None] = field(
        init=False, default_factory=OrderedDict
    )

    def statistics(self) -> BroadcastStreamStatistics:
        return BroadcastStreamStatistics(
            len(self.subscriptions),
            len(self.waiting_publishers),
            sum(len(sub.waiting_receivers) for sub in self.subscriptions),
        )


def _wake_publishers(state: _BroadcastStreamState[T_Item]) -> None:
    for event in state.waiting_publishers:
        event.set()

    state.waiting_publishers.clear()


def _wake_receivers(subscription: _BroadcastSubscription[T_Item]) -> None:
    for event in subscription.waiting_receivers:
        event.set()

    subscription.waiting_receivers.clear()


class BroadcastStream(Generic[T_Item]):
    """
    A broadcast hub that delivers every published item to every live subscription.

    Unlike :func:`~anyio.create_memory_object_stream`, where each item is received
    by exactly one consumer, a broadcast stream fans out every published item so
    that each subscription created via :meth:`subscribe` receives its own copy of
    every item published after the subscription was created.

    Each subscription has its own independent buffer with a positive integer
    capacity. Publishing first checks that **all** live subscriptions have room for
    the item, and only then delivers it to all of them in a single synchronous step
    (the linearization point of a successful publish). If any subscription is full,
    the publisher waits; if the publish is cancelled while waiting, the item is
    guaranteed to not have been delivered to any subscription (there is no partial
    broadcast).

    Publishing when there are no live subscriptions succeeds immediately and the
    item is discarded, since subscriptions only receive items published after they
    were created.
    """

    def __init__(self) -> None:
        self._state: _BroadcastStreamState[T_Item] = _BroadcastStreamState()

    def subscribe(self, max_buffer_size: int = 1) -> BroadcastReceiveStream[T_Item]:
        """
        Create a new subscription to this broadcast stream.

        The returned receive stream will receive its own copy of every item
        published after this call, up to ``max_buffer_size`` unconsumed items.

        :param max_buffer_size: the maximum number of items buffered for this
            subscription before publishers start blocking (must be a positive
            integer)
        :raises ~anyio.ClosedResourceError: if the broadcast stream has been closed
        :raises TypeError: if ``max_buffer_size`` is not an integer
        :raises ValueError: if ``max_buffer_size`` is less than 1
        :return: the receive stream of the new subscription

        """
        if self._state.closed:
            raise ClosedResourceError
        if not isinstance(max_buffer_size, int):
            raise TypeError("max_buffer_size must be an integer")
        if max_buffer_size < 1:
            raise ValueError("max_buffer_size must be at least 1")

        subscription = _BroadcastSubscription[T_Item](max_buffer_size)
        self._state.subscriptions.append(subscription)
        return BroadcastReceiveStream(self._state, subscription)

    def _has_capacity(self) -> bool:
        return all(
            len(subscription.buffer) < subscription.max_buffer_size
            for subscription in self._state.subscriptions
        )

    def _deliver(self, item: T_Item) -> None:
        # This is the linearization point of a successful publish: the item is
        # appended to every live subscription's buffer in a single synchronous
        # step, so no task can observe a partial broadcast.
        for subscription in self._state.subscriptions:
            subscription.buffer.append(item)
            _wake_receivers(subscription)

    def publish_nowait(self, item: T_Item) -> None:
        """
        Publish an item immediately if it can be done without waiting.

        :param item: the item to publish
        :raises ~anyio.ClosedResourceError: if the broadcast stream has been closed
        :raises ~anyio.WouldBlock: if any live subscription's buffer is full

        """
        if self._state.closed:
            raise ClosedResourceError
        if not self._has_capacity():
            raise WouldBlock

        self._deliver(item)

    async def publish(self, item: T_Item) -> None:
        """
        Publish an item to all live subscriptions.

        If any live subscription's buffer is full, this method waits until every
        live subscription has room, then delivers the item to all of them
        atomically. The linearization point of a successful publish is the single
        synchronous step in which the item is appended to every live subscription's
        buffer; no task can observe the item in some buffers but not others.

        If the publish is cancelled while waiting, the item is guaranteed to not
        have been delivered to any subscription.

        :param item: the item to publish
        :raises ~anyio.ClosedResourceError: if the broadcast stream has been closed
            (either before the call or while waiting)

        """
        await checkpoint()
        while True:
            if self._state.closed:
                raise ClosedResourceError
            if self._has_capacity():
                self._deliver(item)
                return

            event = Event()
            self._state.waiting_publishers[event] = None
            try:
                await event.wait()
            except BaseException:
                self._state.waiting_publishers.pop(event, None)
                raise

    def close(self) -> None:
        """
        Close the broadcast stream.

        No new subscriptions can be created and no more items can be published.
        Tasks blocked on :meth:`publish` are woken with
        :exc:`~anyio.ClosedResourceError`. Each subscription keeps its buffered
        items, so receivers first drain what was already published to them and only
        then get :exc:`~anyio.EndOfStream`.

        This works the exact same way as :meth:`aclose`, but is provided as a
        special case for the benefit of synchronous callbacks.

        """
        if not self._state.closed:
            self._state.closed = True
            _wake_publishers(self._state)
            for subscription in self._state.subscriptions:
                _wake_receivers(subscription)

    async def aclose(self) -> None:
        self.close()

    def statistics(self) -> BroadcastStreamStatistics:
        """
        Return statistics about the current state of this stream.

        .. versionadded:: 4.16
        """
        return self._state.statistics()

    def __enter__(self) -> BroadcastStream[T_Item]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> BroadcastStream[T_Item]:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        if not self._state.closed:
            warnings.warn(
                f"Unclosed <{self.__class__.__name__} at {id(self):x}>",
                ResourceWarning,
                stacklevel=1,
                source=self,
            )


@dataclass(eq=False)
class BroadcastReceiveStream(ObjectReceiveStream[T_co], Generic[T_co]):
    """
    The receiving end of a single subscription to a :class:`BroadcastStream`.

    Each subscription has its own independent buffer, so a slow consumer only
    applies backpressure to publishers through its own buffer capacity and never
    causes other subscriptions to miss items.
    """

    _state: _BroadcastStreamState[T_co]
    _subscription: _BroadcastSubscription[T_co]
    _closed: bool = field(init=False, default=False)

    def receive_nowait(self) -> T_co:
        """
        Receive the next item if it can be done without waiting.

        :return: the received item
        :raises ~anyio.ClosedResourceError: if this receive stream has been closed
        :raises ~anyio.EndOfStream: if the buffer is empty and the broadcast stream
            has been closed from the publishing end
        :raises ~anyio.WouldBlock: if there are no items in the buffer

        """
        if self._closed:
            raise ClosedResourceError

        if self._subscription.buffer:
            item = self._subscription.buffer.popleft()
            _wake_publishers(self._state)
            return item

        if self._state.closed:
            raise EndOfStream

        raise WouldBlock

    async def receive(self) -> T_co:
        await checkpoint()
        while True:
            try:
                return self.receive_nowait()
            except WouldBlock:
                event = Event()
                self._subscription.waiting_receivers[event] = None
                try:
                    await event.wait()
                except BaseException:
                    self._subscription.waiting_receivers.pop(event, None)
                    raise

    def clone(self) -> BroadcastReceiveStream[T_co]:
        """
        Create a clone of this receive stream.

        The clone is a **new, independent subscription** with the same buffer
        capacity, not a shared view of this subscription. It does not receive any
        items already buffered for (or received by) this subscription; it only
        receives items published after the clone was created.

        :return: the cloned stream
        :raises ~anyio.ClosedResourceError: if this receive stream or the broadcast
            stream has been closed

        """
        if self._closed or self._state.closed:
            raise ClosedResourceError

        subscription = _BroadcastSubscription[T_co](self._subscription.max_buffer_size)
        self._state.subscriptions.append(subscription)
        return BroadcastReceiveStream(self._state, subscription)

    def close(self) -> None:
        """
        Close the subscription.

        The subscription is removed from the broadcast stream, so publishers no
        longer wait for it and it receives no further items. Any items still
        buffered for this subscription are discarded. Tasks blocked on
        :meth:`BroadcastStream.publish` are woken so they can re-evaluate the
        remaining subscriptions.

        This works the exact same way as :meth:`aclose`, but is provided as a
        special case for the benefit of synchronous callbacks.

        """
        if not self._closed:
            self._closed = True
            if self._subscription in self._state.subscriptions:
                self._state.subscriptions.remove(self._subscription)

            _wake_receivers(self._subscription)
            _wake_publishers(self._state)

    async def aclose(self) -> None:
        self.close()

    def __enter__(self) -> BroadcastReceiveStream[T_co]:
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
