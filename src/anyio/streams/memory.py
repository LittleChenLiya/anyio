from __future__ import annotations

__all__ = (
    "MemoryObjectReceiveStream",
    "MemoryObjectSendStream",
    "MemoryObjectStreamStatistics",
    "receive_any",
)

import warnings
from collections import OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Generic, NamedTuple, TypeVar

from .._core._exceptions import (
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    WouldBlock,
)
from .._core._synchronization import Event
from .._core._testing import TaskInfo, get_current_task
from ..abc import ObjectReceiveStream, ObjectSendStream
from ..lowlevel import checkpoint

T_Item = TypeVar("T_Item")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


class MemoryObjectStreamStatistics(NamedTuple):
    current_buffer_used: int  #: number of items stored in the buffer
    #: maximum number of items that can be stored on this stream (or :data:`math.inf`)
    max_buffer_size: float
    open_send_streams: int  #: number of unclosed clones of the send stream
    open_receive_streams: int  #: number of unclosed clones of the receive stream
    #: number of tasks blocked on :meth:`MemoryObjectSendStream.send`
    tasks_waiting_send: int
    #: number of tasks blocked on :meth:`MemoryObjectReceiveStream.receive`
    tasks_waiting_receive: int


@dataclass(eq=False)
class _MemoryObjectItemReceiver(Generic[T_Item]):
    task_info: TaskInfo = field(init=False, default_factory=get_current_task)
    item: T_Item = field(init=False)

    def try_deliver(self, item: T_Item) -> bool:
        if self.task_info.has_pending_cancellation():
            return False

        self.item = item
        return True

    def __repr__(self) -> str:
        # When item is not defined, we get following error with default __repr__:
        # AttributeError: 'MemoryObjectItemReceiver' object has no attribute 'item'
        item = getattr(self, "item", None)
        return f"{self.__class__.__name__}(task_info={self.task_info}, item={item!r})"


@dataclass(eq=False)
class _MultiStreamItemReceiver(Generic[T_Item]):
    """The shared winning-receiver state of a :func:`receive_any` wait."""

    task_info: TaskInfo = field(init=False, default_factory=get_current_task)
    event: Event = field(init=False, default_factory=Event)
    index: int = field(init=False, default=-1)
    item: T_Item = field(init=False)

    def claim(self, index: int, item: T_Item) -> bool:
        """
        Atomically claim this receiver for an item delivered from ``index``.

        :return: ``True`` if this receiver won the item, ``False`` if another stream
            already won it or the waiting task has a pending cancellation

        """
        if self.index >= 0 or self.task_info.has_pending_cancellation():
            return False

        self.index = index
        self.item = item
        return True


@dataclass(eq=False)
class _MultiStreamRegistration(Generic[T_Item]):
    """A per-stream-state registration of a :func:`receive_any` wait."""

    receiver: _MultiStreamItemReceiver[T_Item]
    index: int
    task_info: TaskInfo = field(init=False, default_factory=get_current_task)

    def try_deliver(self, item: T_Item) -> bool:
        if self.task_info.has_pending_cancellation():
            return False

        return self.receiver.claim(self.index, item)


@dataclass(eq=False)
class _MemoryObjectStreamState(Generic[T_Item]):
    max_buffer_size: float = field()
    buffer: deque[T_Item] = field(init=False, default_factory=deque)
    open_send_channels: int = field(init=False, default=0)
    open_receive_channels: int = field(init=False, default=0)
    waiting_receivers: OrderedDict[
        Event,
        _MemoryObjectItemReceiver[T_Item] | _MultiStreamRegistration[T_Item],
    ] = field(init=False, default_factory=OrderedDict)
    waiting_senders: OrderedDict[Event, T_Item] = field(
        init=False, default_factory=OrderedDict
    )

    def statistics(self) -> MemoryObjectStreamStatistics:
        return MemoryObjectStreamStatistics(
            len(self.buffer),
            self.max_buffer_size,
            self.open_send_channels,
            self.open_receive_channels,
            len(self.waiting_senders),
            len(self.waiting_receivers),
        )


@dataclass(eq=False)
class MemoryObjectReceiveStream(ObjectReceiveStream[T_co], Generic[T_co]):
    _state: _MemoryObjectStreamState[T_co]
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._state.open_receive_channels += 1

    def receive_nowait(self) -> T_co:
        """
        Receive the next item if it can be done without waiting.

        :return: the received item
        :raises ~anyio.ClosedResourceError: if this send stream has been closed
        :raises ~anyio.EndOfStream: if the buffer is empty and this stream has been
            closed from the sending end
        :raises ~anyio.WouldBlock: if there are no items in the buffer and no tasks
            waiting to send

        """
        if self._closed:
            raise ClosedResourceError

        if self._state.waiting_senders:
            # Get the item from the next sender
            send_event, item = self._state.waiting_senders.popitem(last=False)
            self._state.buffer.append(item)
            send_event.set()

        if self._state.buffer:
            return self._state.buffer.popleft()
        elif not self._state.open_send_channels:
            raise EndOfStream

        raise WouldBlock

    async def receive(self) -> T_co:
        await checkpoint()
        try:
            return self.receive_nowait()
        except WouldBlock:
            # Add ourselves in the queue
            receive_event = Event()
            receiver = _MemoryObjectItemReceiver[T_co]()
            self._state.waiting_receivers[receive_event] = receiver

            try:
                await receive_event.wait()
            finally:
                self._state.waiting_receivers.pop(receive_event, None)

            try:
                return receiver.item
            except AttributeError:
                raise EndOfStream from None

    def clone(self) -> MemoryObjectReceiveStream[T_co]:
        """
        Create a clone of this receive stream.

        Each clone can be closed separately. Only when all clones have been closed will
        the receiving end of the memory stream be considered closed by the sending ends.

        :return: the cloned stream

        """
        if self._closed:
            raise ClosedResourceError

        return MemoryObjectReceiveStream(_state=self._state)

    def close(self) -> None:
        """
        Close the stream.

        This works the exact same way as :meth:`aclose`, but is provided as a special
        case for the benefit of synchronous callbacks.

        """
        if not self._closed:
            self._closed = True
            self._state.open_receive_channels -= 1
            if self._state.open_receive_channels == 0:
                send_events = list(self._state.waiting_senders.keys())
                for event in send_events:
                    event.set()

    async def aclose(self) -> None:
        self.close()

    def statistics(self) -> MemoryObjectStreamStatistics:
        """
        Return statistics about the current state of this stream.

        .. versionadded:: 3.0
        """
        return self._state.statistics()

    def __enter__(self) -> MemoryObjectReceiveStream[T_co]:
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


async def receive_any(
    receivers: Sequence[MemoryObjectReceiveStream[T_Item]],
) -> tuple[int, T_Item]:
    """
    Receive one item from any of the given memory object receive streams.

    The call blocks until at least one of the streams has an item available, and then
    receives a single item from the first such stream (in the order given in
    ``receivers``). The other streams are left untouched: no items are removed from
    them, and no items are speculatively received and put back. Streams that have
    reached the end of data are skipped; the call only raises
    :exc:`~anyio.EndOfStream` once every stream has reached the end of data.

    For each underlying memory object stream, the resulting wait is queued after any
    tasks that are already waiting on :meth:`MemoryObjectReceiveStream.receive` of the
    same stream, so existing waiters cannot be overtaken. Clones of the same receive
    stream share the same state and may therefore not be passed more than once.

    :param receivers: the receive streams to wait on
    :return: a tuple of (the index of the stream the item was received from, the item)
    :raises ValueError: if ``receivers`` is empty or contains clones of the same
        receive stream
    :raises TypeError: if any member of ``receivers`` is not a
        :class:`MemoryObjectReceiveStream`
    :raises ~anyio.ClosedResourceError: if any member of ``receivers`` has been closed

    """
    if not receivers:
        raise ValueError("the list of receive streams must not be empty")

    states: list[_MemoryObjectStreamState[T_Item]] = []
    for stream in receivers:
        if not isinstance(stream, MemoryObjectReceiveStream):
            raise TypeError(
                "all receive streams must be MemoryObjectReceiveStream instances; "
                f"got {type(stream).__name__!r}"
            )
        elif stream._state in states:
            raise ValueError(
                "receivers sharing the same underlying memory object stream (such as "
                "clones of the same receive stream) cannot be passed more than once"
            )

        states.append(stream._state)

    if any(stream._closed for stream in receivers):
        raise ClosedResourceError

    await checkpoint()
    while True:
        # First, try to receive immediately, honoring the order of the input streams.
        all_closed = True
        eof_streams: list[int] = []
        for index, stream in enumerate(receivers):
            if stream._closed:
                continue

            all_closed = False
            try:
                return index, stream.receive_nowait()
            except WouldBlock:
                pass
            except EndOfStream:
                eof_streams.append(index)

        if all_closed:
            raise ClosedResourceError
        elif len(eof_streams) == sum(not stream._closed for stream in receivers):
            raise EndOfStream

        # Register a single wait on each stream that is still open for sending.
        wait = _MultiStreamItemReceiver[T_Item]()
        registrations: list[tuple[Event, _MemoryObjectStreamState[T_Item]]] = []
        for index, stream in enumerate(receivers):
            if stream._closed:
                continue

            state = stream._state
            if state.open_send_channels:
                registration = _MultiStreamRegistration[T_Item](
                    receiver=wait, index=index
                )
                state.waiting_receivers[wait.event] = registration
                registrations.append((wait.event, state))

        try:
            await wait.event.wait()
        finally:
            for event, state in registrations:
                state.waiting_receivers.pop(event, None)

        if wait.index >= 0:
            return wait.index, wait.item


@dataclass(eq=False)
class MemoryObjectSendStream(ObjectSendStream[T_contra], Generic[T_contra]):
    _state: _MemoryObjectStreamState[T_contra]
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._state.open_send_channels += 1

    def send_nowait(self, item: T_contra) -> None:
        """
        Send an item immediately if it can be done without waiting.

        :param item: the item to send
        :raises ~anyio.ClosedResourceError: if this send stream has been closed
        :raises ~anyio.BrokenResourceError: if the stream has been closed from the
            receiving end
        :raises ~anyio.WouldBlock: if the buffer is full and there are no tasks waiting
            to receive

        """
        if self._closed:
            raise ClosedResourceError
        if not self._state.open_receive_channels:
            raise BrokenResourceError

        while self._state.waiting_receivers:
            receive_event, receiver = self._state.waiting_receivers.popitem(last=False)
            if receiver.try_deliver(item):
                receive_event.set()
                return

        if len(self._state.buffer) < self._state.max_buffer_size:
            self._state.buffer.append(item)
        else:
            raise WouldBlock

    async def send(self, item: T_contra) -> None:
        """
        Send an item to the stream.

        If the buffer is full, this method blocks until there is again room in the
        buffer or the item can be sent directly to a receiver.

        :param item: the item to send
        :raises ~anyio.ClosedResourceError: if this send stream has been closed
        :raises ~anyio.BrokenResourceError: if the stream has been closed from the
            receiving end

        """
        await checkpoint()
        try:
            self.send_nowait(item)
        except WouldBlock:
            # Wait until there's someone on the receiving end
            send_event = Event()
            self._state.waiting_senders[send_event] = item
            try:
                await send_event.wait()
            except BaseException:
                self._state.waiting_senders.pop(send_event, None)
                raise

            if send_event in self._state.waiting_senders:
                del self._state.waiting_senders[send_event]
                raise BrokenResourceError from None

    def clone(self) -> MemoryObjectSendStream[T_contra]:
        """
        Create a clone of this send stream.

        Each clone can be closed separately. Only when all clones have been closed will
        the sending end of the memory stream be considered closed by the receiving ends.

        :return: the cloned stream

        """
        if self._closed:
            raise ClosedResourceError

        return MemoryObjectSendStream(_state=self._state)

    def close(self) -> None:
        """
        Close the stream.

        This works the exact same way as :meth:`aclose`, but is provided as a special
        case for the benefit of synchronous callbacks.

        """
        if not self._closed:
            self._closed = True
            self._state.open_send_channels -= 1
            if self._state.open_send_channels == 0:
                receive_events = list(self._state.waiting_receivers.keys())
                self._state.waiting_receivers.clear()
                for event in receive_events:
                    event.set()

    async def aclose(self) -> None:
        self.close()

    def statistics(self) -> MemoryObjectStreamStatistics:
        """
        Return statistics about the current state of this stream.

        .. versionadded:: 3.0
        """
        return self._state.statistics()

    def __enter__(self) -> MemoryObjectSendStream[T_contra]:
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
