from __future__ import annotations

import math
from typing import TypeVar

from ..streams.ackqueue import AckQueueReceiveStream, AckQueueSendStream, _AckQueueState

T_Item = TypeVar("T_Item")


class create_ack_queue(
    tuple[AckQueueSendStream[T_Item], AckQueueReceiveStream[T_Item]],
):
    """
    Create a queue of messages that consumers must explicitly acknowledge.

    The queue's item type can be annotated like
    ``create_ack_queue[T_Item](max_buffer_size)``.

    Unlike :func:`~anyio.create_memory_object_stream`, receiving from the queue
    returns an :class:`~anyio.streams.ackqueue.AckQueueDelivery` containing the
    message and an :class:`~anyio.streams.ackqueue.AckHandle`. Messages keep
    taking up capacity until the consumer calls
    :meth:`~anyio.streams.ackqueue.AckHandle.acknowledge`. If the consumer task
    is cancelled (or the handle is discarded, or
    :meth:`~anyio.streams.ackqueue.AckHandle.reject` is called), the message is
    put back at the head of the queue and redelivered before any new messages.

    :param max_buffer_size: number of messages (queued plus in flight) allowed
        before :meth:`~anyio.streams.ackqueue.AckQueueSendStream.send` starts
        blocking; must be a positive integer or :data:`math.inf`
    :return: a tuple of (send stream, receive stream)
    """

    def __new__(  # type: ignore[misc]
        cls, max_buffer_size: float
    ) -> tuple[AckQueueSendStream[T_Item], AckQueueReceiveStream[T_Item]]:
        if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
            raise ValueError("max_buffer_size must be either an integer or math.inf")

        if max_buffer_size < 1:
            raise ValueError("max_buffer_size must be at least 1")

        state = _AckQueueState[T_Item](max_buffer_size)
        return (AckQueueSendStream(state), AckQueueReceiveStream(state))
