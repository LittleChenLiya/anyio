from __future__ import annotations

import math
from typing import TypeVar

from ..streams.ack_queue import (
    AckQueueReceiveStream,
    AckQueueSendStream,
    _AckQueueState,
)

T_Item = TypeVar("T_Item")


class create_ack_queue(
    tuple[AckQueueSendStream[T_Item], AckQueueReceiveStream[T_Item]],
):
    """
    Create an acknowledgement queue.

    Unlike :func:`create_memory_object_stream`, receiving an item returns an
    ``(item, handle)`` pair. The consumer must call
    :meth:`~anyio.streams.ack_queue.MessageAcknowledgeHandle.acknowledge` on the
    handle once the item has been successfully processed. The sending end provides
    :meth:`~anyio.streams.ack_queue.AckQueueSendStream.join` for waiting until all
    outstanding items have been acknowledged.

    Items that are not acknowledged before the receiving task ends (for example
    because it was cancelled) are automatically requeued, as are items explicitly
    returned with :meth:`~anyio.streams.ack_queue.MessageAcknowledgeHandle.requeue`.
    Requeued items keep their original identity and are delivered before any newer
    items.

    The queue's item type can be annotated like
    :func:`create_ack_queue[T_Item]`.

    :param max_buffer_size: maximum number of outstanding items, counting both
        queued items and items delivered but not yet acknowledged; the queue
        rejects zero because a delivered item always counts against the capacity
    :return: a tuple of (send stream, receive stream)

    """

    def __new__(  # type: ignore[misc]
        cls, max_buffer_size: float = 1
    ) -> tuple[AckQueueSendStream[T_Item], AckQueueReceiveStream[T_Item]]:
        if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
            raise ValueError("max_buffer_size must be either an integer or math.inf")
        if max_buffer_size < 1:
            raise ValueError("max_buffer_size must be at least 1")

        state = _AckQueueState[T_Item](max_buffer_size)
        return (AckQueueSendStream(state), AckQueueReceiveStream(state))
