from __future__ import annotations

__all__ = (
    "AckHandle",
    "AckQueueDelivery",
    "AckQueueReceiveStream",
    "AckQueueSendStream",
    "AckQueueStatistics",
    "MessageProcess",
)

import warnings
import weakref
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar

from .._core._exceptions import (
    AlreadyAcknowledgedError,
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    WouldBlock,
)
from .._core._synchronization import Event
from .._core._testing import TaskInfo, get_current_task
from ..lowlevel import checkpoint

if TYPE_CHECKING:
    from types import TracebackType

T_Item = TypeVar("T_Item")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


class AckQueueStatistics(NamedTuple):
    current_buffer_used: int  #: number of messages waiting to be delivered
    #: number of messages handed out but not yet acknowledged
    in_flight: int
    #: number of messages taking up capacity (buffered + in flight)
    total_outstanding: int
    #: maximum number of messages that can take up capacity (or math.inf)
    max_buffer_size: float
    open_send_streams: int  #: number of unclosed clones of the send stream
    open_receive_streams: int  #: number of unclosed clones of the receive stream
    #: number of tasks blocked on AckQueueSendStream.send()
    tasks_waiting_send: int
    #: number of tasks blocked on AckQueueReceiveStream.receive()
    tasks_waiting_receive: int
    #: number of tasks blocked on AckQueueSendStream.join()
    tasks_waiting_join: int


@dataclass(eq=False)
class _AckQueueItemReceiver(Generic[T_Item]):
    task_info: TaskInfo = field(init=False, default_factory=get_current_task)
    item: tuple[T_Item, int] | None = field(init=False, default=None)

    def __repr__(self) -> str:
        return f"_AckQueueItemReceiver(task_info={self.task_info}, item={self.item!r})"


@dataclass(eq=False)
class _AckQueueState(Generic[T_Item]):
    max_buffer_size: float = field()
    #: Messages waiting to be delivered for the first time, in FIFO order.
    buffer: deque[T_Item] = field(init=False, default_factory=deque)
    #: Messages that were returned and must be redelivered before new messages.
    redelivery_buffer: deque[T_Item] = field(init=False, default_factory=deque)
    #: Sequence numbers of messages currently handed out to consumers.
    in_flight: set[int] = field(init=False, default_factory=set)
    #: Maps in-flight sequence numbers to the delivered messages.
    in_flight_items: dict[int, T_Item] = field(init=False, default_factory=dict)
    next_sequence: int = field(init=False, default=0)
    open_send_channels: int = field(init=False, default=0)
    open_receive_channels: int = field(init=False, default=0)
    broken: bool = field(init=False, default=False)
    waiting_receivers: OrderedDict[Event, _AckQueueItemReceiver[T_Item]] = field(
        init=False, default_factory=OrderedDict
    )
    waiting_senders: OrderedDict[Event, T_Item] = field(
        init=False, default_factory=OrderedDict
    )
    join_waiters: set[Event] = field(init=False, default_factory=set)

    @property
    def current_load(self) -> int:
        """Number of messages taking up capacity: buffered + in flight."""
        return len(self.buffer) + len(self.redelivery_buffer) + len(self.in_flight)

    def _register(self, item: T_Item) -> int:
        sequence = self.next_sequence
        self.next_sequence += 1
        self.in_flight.add(sequence)
        self.in_flight_items[sequence] = item
        return sequence

    def _pump(self) -> None:
        """Hand buffered/returned messages or eligible senders to waiting receivers."""
        while True:
            receiver_event = None
            receiver: _AckQueueItemReceiver[T_Item] | None = None
            while self.waiting_receivers:
                event, candidate = self.waiting_receivers.popitem(last=False)
                if not candidate.task_info.has_pending_cancellation():
                    receiver_event = event
                    receiver = candidate
                    break

            if receiver is None or receiver_event is None:
                break

            if self.redelivery_buffer:
                item = self.redelivery_buffer.popleft()
            elif self.buffer:
                item = self.buffer.popleft()
            elif self.waiting_senders and self.current_load < self.max_buffer_size:
                send_event, item = self.waiting_senders.popitem(last=False)
                send_event.set()
            else:
                # No message available; restore the receiver at the head.
                self.waiting_receivers[receiver_event] = receiver
                self.waiting_receivers.move_to_end(receiver_event, last=False)
                break

            sequence = self._register(item)
            receiver.item = (item, sequence)
            receiver_event.set()

        # Admit waiting senders into any remaining free buffer capacity.
        while self.waiting_senders and self.current_load < self.max_buffer_size:
            send_event, item = self.waiting_senders.popitem(last=False)
            self.buffer.append(item)
            send_event.set()

        self._wake_join_if_idle()

        # If the send end is closed and everything has been drained, any receivers
        # still waiting get end-of-stream.
        if not self.open_send_channels:
            for event in list(self.waiting_receivers.keys()):
                event.set()

    def _return_in_flight(self, sequence: int) -> None:
        """Put an in-flight message back at the head; redelivered before new ones."""
        item = self.in_flight_items.pop(sequence)
        self.in_flight.remove(sequence)
        self.redelivery_buffer.appendleft(item)
        self._pump()

    def _wake_join_if_idle(self) -> None:
        if self.current_load == 0:
            for event in list(self.join_waiters):
                event.set()

            self.join_waiters.clear()

    def statistics(self) -> AckQueueStatistics:
        return AckQueueStatistics(
            current_buffer_used=len(self.buffer) + len(self.redelivery_buffer),
            in_flight=len(self.in_flight),
            total_outstanding=self.current_load,
            max_buffer_size=self.max_buffer_size,
            open_send_streams=self.open_send_channels,
            open_receive_streams=self.open_receive_channels,
            tasks_waiting_send=len(self.waiting_senders),
            tasks_waiting_receive=len(self.waiting_receivers),
            tasks_waiting_join=len(self.join_waiters),
        )


def _abandoned_handle_finalizer(
    state_ref: weakref.ref[_AckQueueState[T_Item]], sequence: int
) -> None:
    # Running from garbage collection must never raise; do nothing if the queue
    # state is gone or being torn down.
    try:
        state = state_ref()
        if state is not None and sequence in state.in_flight:
            state._return_in_flight(sequence)
    except Exception:
        # Garbage collection must never propagate errors from the finalizer.
        return


def _make_handle(state: _AckQueueState[T_Item], sequence: int) -> AckHandle[T_Item]:
    handle = AckHandle(state, sequence)
    _attach_finalizer(handle, state, sequence)
    return handle


def _attach_finalizer(
    handle: AckHandle[T_Item], state: _AckQueueState[T_Item], sequence: int
) -> None:
    handle._finalizer = weakref.finalize(
        handle,
        _abandoned_handle_finalizer,
        weakref.ref(state),
        sequence,
    )


class AckHandle(Generic[T_Item]):
    """
    A handle for acknowledging or rejecting a delivered message.

    The handle is considered settled once acknowledge() (or reject()) has been
    called. Settling it a second time raises AlreadyAcknowledgedError.

    If a handle is never settled and the last reference to it is dropped (which
    happens when a consumer task is cancelled while processing the message), the
    message is automatically put back at the head of the queue and redelivered
    before any new messages. The handle is considered settled afterwards, so calling
    acknowledge() on it then raises AlreadyAcknowledgedError.
    """

    __slots__ = ("__weakref__", "_finalizer", "_sequence", "_state_ref")

    def __init__(self, state: _AckQueueState[T_Item], sequence: int) -> None:
        self._state_ref: weakref.ref[_AckQueueState[T_Item]] = weakref.ref(state)
        self._sequence = sequence
        self._finalizer: weakref.finalize | None = None

    @property
    def sequence(self) -> int:
        """
        The sequence number of this delivery.

        Every delivery of the same message (including redeliveries after a reject or
        abandonment) gets a new sequence number, while the message itself stays the
        same object.
        """
        return self._sequence

    def _settle(self, requeue: bool) -> None:
        state = self._state_ref()
        if state is None or self._sequence not in state.in_flight:
            raise AlreadyAcknowledgedError(
                "this delivery has already been acknowledged or rejected"
            )

        if self._finalizer is not None:
            self._finalizer.detach()

        if requeue:
            state._return_in_flight(self._sequence)
        else:
            state.in_flight_items.pop(self._sequence)
            state.in_flight.remove(self._sequence)
            state._pump()

    def acknowledge(self) -> None:
        """
        Mark the message as successfully processed.

        The message is removed from the queue and its capacity slot is released,
        potentially unblocking a waiting send() or join().

        :raises AlreadyAcknowledgedError: if the delivery has already been
            acknowledged, rejected, or automatically returned after the handle was
            abandoned
        """
        self._settle(requeue=False)

    def reject(self) -> None:
        """
        Return the message to the head of the queue to be redelivered.

        The returned message takes priority over all messages that have not yet been
        delivered. It keeps occupying its capacity slot, so waiting senders are not
        unblocked by this call.

        :raises AlreadyAcknowledgedError: if the delivery has already been
            acknowledged, rejected, or automatically returned after the handle was
            abandoned
        """
        self._settle(requeue=True)

    def __repr__(self) -> str:
        state = self._state_ref()
        status = (
            "pending"
            if state is not None and self._sequence in state.in_flight
            else "settled"
        )
        return f"AckHandle(sequence={self._sequence}, status={status!r})"


class AckQueueDelivery(NamedTuple, Generic[T_co]):
    """A received message together with its AckHandle."""

    item: T_co
    ack: AckHandle[T_co]


class MessageProcess(Generic[T_co]):
    """
    Async context manager returned by :meth:`AckQueueReceiveStream.process`.

    Entering the context receives the next message; ``delivery`` then holds the
    received :class:`AckQueueDelivery` (the message and its :class:`AckHandle`).

    If the body of the ``async with`` block is cancelled (or raises any other
    exception) and the message has not been settled yet, the message is
    deterministically put back at the head of the queue for redelivery. Acknowledge
    the delivery explicitly for successful processing::

        async with receive_stream.process() as delivery:
            await handle(delivery.item)
            delivery.ack.acknowledge()

    You can also call :meth:`AckHandle.reject` inside the block to return the
    message early. Settling the delivery more than once always raises
    :class:`AlreadyAcknowledgedError`.
    """

    __slots__ = ("_stream", "delivery")

    def __init__(self, stream: AckQueueReceiveStream[T_co]) -> None:
        self._stream = stream
        self.delivery: AckQueueDelivery[T_co]

    async def __aenter__(self) -> AckQueueDelivery[T_co]:
        self.delivery = await self._stream.receive()
        return self.delivery

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        handle = self.delivery.ack
        state = handle._state_ref()
        if state is not None and handle._sequence in state.in_flight:
            # Cancelled or failed before acknowledging: return the message so it
            # is redelivered. Detach the garbage-collection finalizer first so
            # that it cannot return the same delivery a second time.
            if handle._finalizer is not None:
                handle._finalizer.detach()

            state._return_in_flight(handle._sequence)

    def __repr__(self) -> str:
        return f"MessageProcess(delivery={self.delivery!r})"


@dataclass(eq=False)
class AckQueueReceiveStream(Generic[T_co]):
    _state: _AckQueueState[T_co]
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._state.open_receive_channels += 1

    def receive_nowait(self) -> AckQueueDelivery[T_co]:
        """
        Receive the next message if it can be done without waiting.

        Returned messages (via AckHandle.reject() or abandoned handles) are
        delivered before new messages.

        :return: the received message and a handle for acknowledging it
        :raises ~anyio.ClosedResourceError: if this receive stream has been closed
        :raises ~anyio.EndOfStream: if no messages are left and the send end is closed
        :raises ~anyio.WouldBlock: if no message is available yet
        """
        if self._closed:
            raise ClosedResourceError

        if self._state.redelivery_buffer:
            item = self._state.redelivery_buffer.popleft()
        elif (
            self._state.waiting_senders
            and self._state.current_load < self._state.max_buffer_size
        ):
            send_event, item = self._state.waiting_senders.popitem(last=False)
            send_event.set()
        elif self._state.buffer:
            item = self._state.buffer.popleft()
        elif not self._state.open_send_channels:
            raise EndOfStream
        else:
            raise WouldBlock

        sequence = self._state._register(item)
        return AckQueueDelivery(item, _make_handle(self._state, sequence))

    async def receive(self) -> AckQueueDelivery[T_co]:
        """
        Receive the next message together with an AckHandle.

        The caller must call acknowledge() on the handle after the message has been
        successfully processed. If the receiving task is cancelled before
        acknowledging (or the handle is otherwise discarded without being settled),
        the message is automatically returned to the head of the queue and
        redelivered before any new messages.

        :raises ~anyio.ClosedResourceError: if this receive stream has been closed
        :raises ~anyio.EndOfStream: if no messages are left and the send end is closed
        :raises ~anyio.BrokenResourceError: if the last receive stream was closed
            while this call was waiting
        """
        await checkpoint()
        try:
            return self.receive_nowait()
        except WouldBlock:
            receive_event = Event()
            receiver = _AckQueueItemReceiver[T_co]()
            self._state.waiting_receivers[receive_event] = receiver

            try:
                await receive_event.wait()
            except BaseException:
                self._state.waiting_receivers.pop(receive_event, None)
                if receiver.item is not None:
                    # The message was handed over just as the cancellation hit.
                    _, sequence = receiver.item
                    self._state._return_in_flight(sequence)

                raise
            else:
                self._state.waiting_receivers.pop(receive_event, None)

            if receiver.item is None:
                # Either this task had a pending cancellation, or the last
                # receive stream was closed while we were waiting. Let the
                # cancellation propagate if there is one; otherwise report the
                # queue as broken.
                if receiver.task_info.has_pending_cancellation():
                    # A checkpoint raises the pending cancellation immediately.
                    await checkpoint()

                raise BrokenResourceError from None

            item, sequence = receiver.item
            handle = AckHandle(self._state, sequence)
            _attach_finalizer(handle, self._state, sequence)
            return AckQueueDelivery(item, handle)

    def process(self) -> MessageProcess[T_co]:
        """
        Receive a message within an ``async with`` block, with automatic return.

        Entering the context manager waits for the next message, exactly like
        :meth:`receive`. If the block is cancelled or exits with an exception
        before the message is acknowledged or rejected, the same message is put
        back at the head of the queue and redelivered before any new messages.

        On success the message must be acknowledged explicitly::

            async with receive.process() as delivery:
                await work_on(delivery.item)
                delivery.ack.acknowledge()

        :return: an async context manager yielding an :class:`AckQueueDelivery`
        """
        return MessageProcess(self)

    def clone(self) -> AckQueueReceiveStream[T_co]:
        """
        Create a clone of this receive stream.

        Each clone can be closed separately. Only when all clones have been closed
        will the receiving end of the queue be considered closed.

        :return: the cloned stream
        """
        if self._closed:
            raise ClosedResourceError

        return AckQueueReceiveStream(_state=self._state)

    def close(self) -> None:
        """
        Close the stream.

        This works the exact same way as aclose(), but is provided as a special case
        for the benefit of synchronous callbacks.
        """
        if not self._closed:
            self._closed = True
            self._state.open_receive_channels -= 1
            if self._state.open_receive_channels == 0:
                self._state.broken = True

                # Senders and joiners can no longer make progress.
                for event in list(self._state.waiting_senders.keys()):
                    event.set()

                for event in list(self._state.join_waiters):
                    event.set()

                # Any in-flight messages will never be acknowledged now; drop them
                # so that tasks are not stuck waiting on dead handles.
                self._state.in_flight.clear()
                self._state.in_flight_items.clear()
                self._state._wake_join_if_idle()

                for event in list(self._state.waiting_receivers.keys()):
                    event.set()

    async def aclose(self) -> None:
        self.close()

    def statistics(self) -> AckQueueStatistics:
        """Return statistics about the current state of the queue."""
        return self._state.statistics()

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
class AckQueueSendStream(Generic[T_contra]):
    _state: _AckQueueState[T_contra]
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._state.open_send_channels += 1

    def send_nowait(self, item: T_contra) -> None:
        """
        Send a message immediately if it can be done without waiting.

        The message counts against the queue capacity until it is acknowledged,
        including while it is being processed by a consumer.

        :param item: the message to send
        :raises ~anyio.ClosedResourceError: if this send stream has been closed
        :raises ~anyio.BrokenResourceError: if the queue has been closed from the
            receiving end
        :raises ~anyio.WouldBlock: if the queue is full
        """
        if self._closed:
            raise ClosedResourceError

        if not self._state.open_receive_channels:
            raise BrokenResourceError

        while self._state.waiting_receivers:
            event, receiver = self._state.waiting_receivers.popitem(last=False)
            if not receiver.task_info.has_pending_cancellation():
                sequence = self._state._register(item)
                receiver.item = (item, sequence)
                event.set()
                return

        if self._state.current_load < self._state.max_buffer_size:
            self._state.buffer.append(item)
        else:
            raise WouldBlock

    async def send(self, item: T_contra) -> None:
        """
        Send a message to the queue.

        The message counts against the queue capacity until a consumer acknowledges
        it. If the queue is full (all capacity is taken up by queued or in-flight
        messages), this method blocks until a message is acknowledged.

        :param item: the message to send
        :raises ~anyio.ClosedResourceError: if this send stream has been closed
        :raises ~anyio.BrokenResourceError: if the queue has been closed from the
            receiving end
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

            self._state.waiting_senders.pop(send_event, None)
            if (
                self._state.broken
                or not self._state.open_receive_channels
                or self._closed
            ):
                raise BrokenResourceError from None

    async def join(self) -> None:
        """
        Wait until there are no unacknowledged messages left.

        Both messages still waiting in the queue and messages that have been handed
        out but not yet acknowledged are counted. Returned messages keep the count
        above zero until they are redelivered and acknowledged, so a waiting
        join() does not return while a message is being redelivered.

        :raises ~anyio.BrokenResourceError: if the last receive stream has been
            closed while this call was waiting
        """
        await checkpoint()
        if self._state.broken or not self._state.open_receive_channels:
            raise BrokenResourceError

        if self._state.current_load == 0:
            return

        join_event = Event()
        self._state.join_waiters.add(join_event)
        try:
            await join_event.wait()
        except BaseException:
            self._state.join_waiters.discard(join_event)
            raise

        self._state.join_waiters.discard(join_event)
        if self._state.broken or not self._state.open_receive_channels:
            raise BrokenResourceError

    def clone(self) -> AckQueueSendStream[T_contra]:
        """
        Create a clone of this send stream.

        Each clone can be closed separately. Only when all clones have been closed
        will the sending end of the queue be considered closed.

        :return: the cloned stream
        """
        if self._closed:
            raise ClosedResourceError

        return AckQueueSendStream(_state=self._state)

    def close(self) -> None:
        """
        Close the stream.

        This works the exact same way as aclose(), but is provided as a special case
        for the benefit of synchronous callbacks.

        Closing the last send stream does not discard any messages; consumers can
        still receive and acknowledge everything that was sent before the close.
        """
        if not self._closed:
            self._closed = True
            self._state.open_send_channels -= 1
            if self._state.open_send_channels == 0:
                self._state._pump()

    async def aclose(self) -> None:
        self.close()

    def statistics(self) -> AckQueueStatistics:
        """Return statistics about the current state of the queue."""
        return self._state.statistics()

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
