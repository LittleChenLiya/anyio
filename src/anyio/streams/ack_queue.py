from __future__ import annotations

__all__ = (
    "AckQueueReceiveStream",
    "AckQueueSendStream",
    "AckQueueStatistics",
    "MessageAcknowledgeHandle",
)

import warnings
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from types import TracebackType
from typing import Generic, Literal, NamedTuple, TypeVar

from .._core._eventloop import current_async_library, current_time
from .._core._exceptions import (
    BrokenResourceError,
    ClosedResourceError,
    DuplicateAcknowledgeError,
    EndOfStream,
    WouldBlock,
)
from .._core._synchronization import Event
from .._core._tasks import CancelScope
from .._core._testing import TaskInfo, get_current_task, get_running_tasks
from ..abc import ObjectReceiveStream, ObjectSendStream
from ..lowlevel import checkpoint

T_Item = TypeVar("T_Item")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

_HandleStatus = Literal["delivered", "acknowledged", "requeued", "abandoned"]


@dataclass(frozen=True)
class _Envelope(Generic[T_Item]):
    sequence: int
    item: T_Item


@dataclass(eq=False)
class _WaitingReceiver(Generic[T_Item]):
    task_info: TaskInfo = field(init=False, default_factory=get_current_task)
    envelope: _Envelope[T_Item] | None = field(init=False, default=None)
    handle: MessageAcknowledgeHandle[T_Item] | None = field(init=False, default=None)


@dataclass(eq=False)
class _AckQueueState(Generic[T_Item]):
    max_buffer_size: float = field()
    #: Items waiting to be delivered. Requeued items are placed at the front.
    buffer: deque[_Envelope[T_Item]] = field(init=False, default_factory=deque)
    #: Handles for delivered but not yet acknowledged items, keyed by sequence
    pending: OrderedDict[int, MessageAcknowledgeHandle[T_Item]] = field(
        init=False, default_factory=OrderedDict
    )
    waiting_receivers: OrderedDict[Event, _WaitingReceiver[T_Item]] = field(
        init=False, default_factory=OrderedDict
    )
    waiting_senders: OrderedDict[Event, T_Item] = field(
        init=False, default_factory=OrderedDict
    )
    waiting_joiners: OrderedDict[Event, None] = field(
        init=False, default_factory=OrderedDict
    )
    open_send_channels: int = field(init=False, default=0)
    open_receive_channels: int = field(init=False, default=0)
    _sequence_counter: int = field(init=False, default=0)
    _janitor_started: bool = field(init=False, default=False)
    _janitor_stop: bool = field(init=False, default=False)
    _janitor_event: Event = field(init=False, default_factory=Event)
    _janitor_interval: float = field(default=0.05)

    @property
    def outstanding(self) -> int:
        """Number of items either queued or delivered but unacknowledged."""
        return len(self.buffer) + len(self.pending)

    def _has_capacity(self) -> bool:
        return self.outstanding < self.max_buffer_size

    def _new_envelope(self, item: T_Item) -> _Envelope[T_Item]:
        envelope = _Envelope(self._sequence_counter, item)
        self._sequence_counter += 1
        return envelope

    def _pop_live_receiver(
        self,
    ) -> tuple[Event, _WaitingReceiver[T_Item]] | None:
        while self.waiting_receivers:
            event, receiver = self.waiting_receivers.popitem(last=False)
            if not receiver.task_info.has_pending_cancellation():
                return event, receiver

        return None

    def _admit(self, item: T_Item) -> Event | None:
        """
        Admit a new item, handing it directly to a waiting receiver if possible.

        :return: the event of a receiver that the item was handed off to, or
            ``None`` if the item was buffered

        """
        envelope = self._new_envelope(item)
        receiver = self._pop_live_receiver()
        if receiver is not None:
            receive_event, waiting_receiver = receiver
            handle = MessageAcknowledgeHandle(
                self, envelope, waiting_receiver.task_info.id
            )
            self.pending[envelope.sequence] = handle
            waiting_receiver.envelope = envelope
            waiting_receiver.handle = handle
            self._ensure_janitor()
            return receive_event

        self.buffer.append(envelope)
        return None

    def _pump(self) -> None:
        """Admit as many waiting senders as the remaining capacity allows."""
        while (
            self.open_receive_channels and self.waiting_senders and self._has_capacity()
        ):
            send_event, item = self.waiting_senders.popitem(last=False)
            receive_event = self._admit(item)
            if receive_event is not None:
                receive_event.set()
            else:
                send_event.set()

    def _wake_parked_receivers(self) -> None:
        for event in tuple(self.waiting_receivers):
            event.set()

    def _wake_joiners(self) -> None:
        for event in tuple(self.waiting_joiners):
            event.set()

    def reap_abandoned(self) -> None:
        """
        Requeue items whose receiving task has ended without acknowledging them.

        Such handles can be left behind when a consumer task is cancelled after
        receiving an item but before acknowledging it.

        """
        try:
            running_tasks = get_running_tasks()
        except RuntimeError:
            # No running event loop, so there are no abandoned async tasks to reap
            return

        live_task_ids = {task.id for task in running_tasks}
        stale = [
            (sequence, handle)
            for sequence, handle in self.pending.items()
            if handle.owner_task_id not in live_task_ids
        ]
        for sequence, handle in stale:
            del self.pending[sequence]
            handle._status = "abandoned"

        # Requeue in the original delivery order, ahead of any newer buffered items.
        for _sequence, handle in reversed(stale):
            self.buffer.appendleft(_Envelope(handle.sequence, handle.item))

        if stale:
            self._wake_parked_receivers()
            self._pump()

    def _ensure_janitor(self) -> None:
        """
        Start the background task that reaps items abandoned by cancelled consumers.

        The janitor periodically checks whether the tasks owning delivered items are
        still running, so redelivery and waking up blocked receivers/senders does
        not depend on someone else calling into the queue.

        """
        if self._janitor_started:
            return

        self._janitor_started = True
        if current_async_library() == "asyncio":
            import asyncio

            asyncio.ensure_future(self._run_janitor())
        else:
            import trio.lowlevel

            trio.lowlevel.spawn_system_task(self._run_janitor)

    async def _run_janitor(self) -> None:
        while not self._janitor_stop:
            # Wait before the first check too: the receiving task may not have been
            # scheduled yet (in particular with asyncio's eager task factory), and
            # immediately checking would misclassify it as abandoned.
            with CancelScope(deadline=current_time() + self._janitor_interval):
                await self._janitor_event.wait()
            if self.open_send_channels == 0 and self.open_receive_channels == 0:
                self._stop_janitor()
                return

            if self.pending:
                self.reap_abandoned()
                if self.pending and not self.open_receive_channels:
                    # Nothing can acknowledge the outstanding items anymore;
                    # waiting tasks have already been woken through close().
                    self._stop_janitor()

    def _stop_janitor(self) -> None:
        self._janitor_stop = True
        self._janitor_event.set()

    def acknowledge(self, handle: MessageAcknowledgeHandle[T_Item]) -> None:
        if handle._status == "acknowledged":
            raise DuplicateAcknowledgeError(
                f"message with sequence number {handle.sequence} has already been "
                "acknowledged"
            )
        if handle._status != "delivered":
            raise ValueError(
                f"message with sequence number {handle.sequence} has already been "
                f"{handle._status}"
            )
        if self.pending.get(handle.sequence) is not handle:
            raise ValueError(
                f"message with sequence number {handle.sequence} is no longer held "
                "by this handle"
            )

        handle._status = "acknowledged"
        del self.pending[handle.sequence]
        if self.outstanding == 0:
            self._wake_joiners()

        self._pump()

    def requeue(self, handle: MessageAcknowledgeHandle[T_Item]) -> None:
        if handle._status == "acknowledged":
            raise ValueError(
                f"message with sequence number {handle.sequence} has already been "
                "acknowledged and cannot be requeued"
            )
        if handle._status != "delivered":
            raise ValueError(
                f"message with sequence number {handle.sequence} has already been "
                f"{handle._status}"
            )
        if self.pending.get(handle.sequence) is not handle:
            raise ValueError(
                f"message with sequence number {handle.sequence} is no longer held "
                "by this handle"
            )

        handle._status = "requeued"
        del self.pending[handle.sequence]
        self.buffer.appendleft(_Envelope(handle.sequence, handle.item))
        self._wake_parked_receivers()

    def deliver_from_buffer(
        self,
    ) -> tuple[_Envelope[T_Item], MessageAcknowledgeHandle[T_Item]] | None:
        if not self.buffer:
            return None

        envelope = self.buffer.popleft()
        task_info = get_current_task()
        handle = MessageAcknowledgeHandle(self, envelope, task_info.id)
        self.pending[envelope.sequence] = handle
        self._ensure_janitor()
        return envelope, handle


class MessageAcknowledgeHandle(Generic[T_Item]):
    """
    A handle for acknowledging or returning a message received from an
    acknowledgement queue.

    The handle is bound to the task that received the message. If that task ends
    without acknowledging the message (e.g. it is cancelled), the message is
    automatically requeued and keeps its original identity (sequence number).

    The handle can also be used as a context manager that requeues the message when
    the ``with`` block is exited without acknowledging it::

        item, handle = await receive_stream.receive()
        with handle:
            process(item)
            handle.acknowledge()

    """

    def __init__(
        self,
        state: _AckQueueState[T_Item],
        envelope: _Envelope[T_Item],
        owner_task_id: int,
    ) -> None:
        self._state = state
        self._envelope = envelope
        self.owner_task_id = owner_task_id
        self._status = "delivered"

    @property
    def item(self) -> T_Item:
        """The received message."""
        return self._envelope.item

    @property
    def sequence(self) -> int:
        """
        Sequence number of the message.

        The number stays the same across redelivery attempts, so a redelivered
        message is still the same message.

        """
        return self._envelope.sequence

    @property
    def acknowledged(self) -> bool:
        """``True`` if the message has been acknowledged."""
        return self._status == "acknowledged"

    def acknowledge(self) -> None:
        """
        Mark the message as successfully processed.

        After this call the message is gone from the queue and can never be
        redelivered.

        :raises ~anyio.DuplicateAcknowledgeError: if the message has already been
            acknowledged
        :raises ValueError: if the message has already been requeued (explicitly or
            because the receiving task was cancelled)

        """
        self._state.acknowledge(self)

    # Shorthand for ``acknowledge()``.
    ack = acknowledge

    def requeue(self) -> None:
        """
        Return the message to the queue without acknowledging it.

        The message is placed ahead of any other buffered messages and will be the
        next one delivered. It keeps its original sequence number.

        :raises ValueError: if the message has already been acknowledged or requeued

        """
        self._state.requeue(self)

    def __enter__(self) -> MessageAcknowledgeHandle[T_Item]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._status == "delivered":
            self.requeue()


class AckQueueStatistics(NamedTuple):
    current_buffer_used: int  #: number of items stored in the buffer
    #: number of items delivered but not yet acknowledged
    current_in_flight: int
    #: maximum number of outstanding items (or :data:`math.inf`)
    max_buffer_size: float
    open_send_streams: int  #: number of unclosed clones of the send streams
    open_receive_streams: int  #: number of unclosed clones of the receive streams
    #: number of tasks blocked on :meth:`AckQueueSendStream.send`
    tasks_waiting_send: int
    #: number of tasks blocked on :meth:`AckQueueReceiveStream.receive`
    tasks_waiting_receive: int
    #: number of tasks blocked on :meth:`AckQueueSendStream.join`
    tasks_waiting_join: int


@dataclass(eq=False)
class AckQueueReceiveStream(
    ObjectReceiveStream[tuple[T_co, MessageAcknowledgeHandle[T_co]]],
    Generic[T_co],
):
    """
    The receive end of an acknowledgement queue.

    Each call to :meth:`receive` returns an ``(item, handle)`` pair. Iterating over
    the stream yields the same pairs.

    Items abandoned by a cancelled receiving task, or explicitly returned with
    :meth:`~MessageAcknowledgeHandle.requeue`, are redelivered before any newer
    buffered items.
    """

    _state: _AckQueueState[T_co]
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._state.open_receive_channels += 1

    def receive_nowait(
        self,
    ) -> tuple[T_co, MessageAcknowledgeHandle[T_co]]:
        """
        Receive the next item if it can be done without waiting.

        Requeued items are delivered before any newer items.

        :return: a tuple of the received item and a
            :class:`MessageAcknowledgeHandle`
        :raises ~anyio.ClosedResourceError: if this receive stream has been closed
        :raises ~anyio.EndOfStream: if the buffer is empty and all send streams have
            been closed
        :raises ~anyio.WouldBlock: if there are no items available

        """
        if self._closed:
            raise ClosedResourceError

        self._state.reap_abandoned()
        delivered = self._state.deliver_from_buffer()
        if delivered is not None:
            envelope, handle = delivered
            return envelope.item, handle
        elif not self._state.open_send_channels:
            raise EndOfStream

        raise WouldBlock

    async def receive(
        self,
    ) -> tuple[T_co, MessageAcknowledgeHandle[T_co]]:
        """
        Receive the next item and its acknowledgement handle.

        :return: a tuple of the received item and a
            :class:`MessageAcknowledgeHandle`
        :raises ~anyio.ClosedResourceError: if this receive stream has been closed
        :raises ~anyio.EndOfStream: if all send streams have been closed and the
            queue is empty

        """
        await checkpoint()
        while True:
            try:
                return self.receive_nowait()
            except WouldBlock:
                receive_event = Event()
                receiver = _WaitingReceiver[T_co]()
                self._state.waiting_receivers[receive_event] = receiver

                # Guard against the race where an item became available between
                # receive_nowait() and registering ourselves as waiting.
                self._state._pump()
                if receiver.envelope is None:
                    try:
                        await receive_event.wait()
                    except BaseException:
                        self._state.waiting_receivers.pop(receive_event, None)
                        raise

                self._state.waiting_receivers.pop(receive_event, None)
                if receiver.envelope is not None and receiver.handle is not None:
                    return receiver.envelope.item, receiver.handle

                # Woken without a direct handoff (e.g. after a requeue); loop back
                # to receive from the buffer.

    def clone(self) -> AckQueueReceiveStream[T_co]:
        """Create a clone of this receive stream."""
        if self._closed:
            raise ClosedResourceError

        return AckQueueReceiveStream(_state=self._state)

    def close(self) -> None:
        """
        Close this clone of the receive stream.

        When the last clone is closed, all blocked senders and joiners are woken
        with :exc:`~anyio.BrokenResourceError`.

        """
        if not self._closed:
            self._closed = True
            self._state.open_receive_channels -= 1
            if self._state.open_receive_channels == 0:
                for event in tuple(self._state.waiting_senders):
                    event.set()

                self._state._wake_joiners()
                if self._state.open_send_channels == 0:
                    self._state._stop_janitor()

    async def aclose(self) -> None:
        self.close()

    def statistics(self) -> AckQueueStatistics:
        """Return statistics about the current state of the queue."""
        return AckQueueStatistics(
            current_buffer_used=len(self._state.buffer),
            current_in_flight=len(self._state.pending),
            max_buffer_size=self._state.max_buffer_size,
            open_send_streams=self._state.open_send_channels,
            open_receive_streams=self._state.open_receive_channels,
            tasks_waiting_send=len(self._state.waiting_senders),
            tasks_waiting_receive=len(self._state.waiting_receivers),
            tasks_waiting_join=len(self._state.waiting_joiners),
        )

    def __enter__(self) -> AckQueueReceiveStream[T_co]:
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
class AckQueueSendStream(
    ObjectSendStream[T_contra],
    Generic[T_contra],
):
    """
    The send end of an acknowledgement queue.

    :meth:`send` blocks when the total number of queued and unacknowledged items
    reaches the queue capacity. :meth:`join` blocks until all items that entered the
    queue have been acknowledged.
    """

    _state: _AckQueueState[T_contra]
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._state.open_send_channels += 1

    def send_nowait(self, item: T_contra) -> None:
        """
        Send an item immediately if it can be done without waiting.

        :param item: the item to send
        :raises ~anyio.ClosedResourceError: if this send stream has been closed
        :raises ~anyio.BrokenResourceError: if all receive streams have been closed
        :raises ~anyio.WouldBlock: if the queue is full

        """
        if self._closed:
            raise ClosedResourceError
        if not self._state.open_receive_channels:
            raise BrokenResourceError

        self._state.reap_abandoned()
        if self._state._has_capacity():
            receive_event = self._state._admit(item)
            if receive_event is not None:
                receive_event.set()
        else:
            raise WouldBlock

    async def send(self, item: T_contra) -> None:
        """
        Send an item to the queue.

        If the queue is full, this method blocks until a previously delivered item
        is acknowledged (freeing a slot), or all receive streams are closed.

        :param item: the item to send
        :raises ~anyio.ClosedResourceError: if this send stream has been closed
        :raises ~anyio.BrokenResourceError: if all receive streams have been closed

        """
        await checkpoint()
        try:
            self.send_nowait(item)
        except WouldBlock:
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

    async def join(self) -> None:
        """
        Block until all queued and delivered-but-unacknowledged items have been
        acknowledged.

        Requeued items (explicitly returned or abandoned by a cancelled consumer)
        still count as outstanding until acknowledged after redelivery.

        :raises ~anyio.BrokenResourceError: if the last receive stream is closed
            while items remain unacknowledged

        """
        await checkpoint()
        while True:
            self._state.reap_abandoned()
            if self._state.outstanding == 0:
                return
            if not self._state.open_receive_channels:
                raise BrokenResourceError(
                    "all receive streams were closed before the outstanding items "
                    "were acknowledged"
                )

            join_event = Event()
            self._state.waiting_joiners[join_event] = None
            try:
                await join_event.wait()
            except BaseException:
                self._state.waiting_joiners.pop(join_event, None)
                raise

            self._state.waiting_joiners.pop(join_event, None)

    def clone(self) -> AckQueueSendStream[T_contra]:
        """Create a clone of this send stream."""
        if self._closed:
            raise ClosedResourceError

        return AckQueueSendStream(_state=self._state)

    def close(self) -> None:
        """
        Close this clone of the send stream.

        When the last clone is closed, blocked receivers get
        :exc:`~anyio.EndOfStream` once the queue has been drained.

        """
        if not self._closed:
            self._closed = True
            self._state.open_send_channels -= 1
            if self._state.open_send_channels == 0:
                receive_events = list(self._state.waiting_receivers.keys())
                self._state.waiting_receivers.clear()
                for event in receive_events:
                    event.set()
                if self._state.open_receive_channels == 0:
                    self._state._stop_janitor()

    async def aclose(self) -> None:
        self.close()

    def statistics(self) -> AckQueueStatistics:
        """Return statistics about the current state of the queue."""
        return AckQueueStatistics(
            current_buffer_used=len(self._state.buffer),
            current_in_flight=len(self._state.pending),
            max_buffer_size=self._state.max_buffer_size,
            open_send_streams=self._state.open_send_channels,
            open_receive_streams=self._state.open_receive_channels,
            tasks_waiting_send=len(self._state.waiting_senders),
            tasks_waiting_receive=len(self._state.waiting_receivers),
            tasks_waiting_join=len(self._state.waiting_joiners),
        )

    def __enter__(self) -> AckQueueSendStream[T_contra]:
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
