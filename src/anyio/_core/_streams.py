from __future__ import annotations

import math
from typing import TypeVar
from warnings import warn

from ..streams.broadcast import (
    BroadcastObjectSendStream,
    _BroadcastState,
)
from ..streams.memory import (
    MemoryObjectReceiveStream,
    MemoryObjectSendStream,
    _MemoryObjectStreamState,
)

T_Item = TypeVar("T_Item")


class create_memory_object_stream(
    tuple[MemoryObjectSendStream[T_Item], MemoryObjectReceiveStream[T_Item]],
):
    """
    Create a memory object stream.

    The stream's item type can be annotated like
    :func:`create_memory_object_stream[T_Item]`.

    :param max_buffer_size: number of items held in the buffer until ``send()`` starts
        blocking
    :param item_type: old way of marking the streams with the right generic type for
        static typing (does nothing on AnyIO 4)

        .. deprecated:: 4.0
          Use ``create_memory_object_stream[YourItemType](...)`` instead.
    :return: a tuple of (send stream, receive stream)

    """

    def __new__(  # type: ignore[misc]
        cls, max_buffer_size: float = 0, item_type: object = None
    ) -> tuple[MemoryObjectSendStream[T_Item], MemoryObjectReceiveStream[T_Item]]:
        if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
            raise ValueError("max_buffer_size must be either an integer or math.inf")
        if max_buffer_size < 0:
            raise ValueError("max_buffer_size cannot be negative")
        if item_type is not None:
            warn(
                "The item_type argument has been deprecated in AnyIO 4.0. "
                "Use create_memory_object_stream[YourItemType](...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        state = _MemoryObjectStreamState[T_Item](max_buffer_size)
        return (MemoryObjectSendStream(state), MemoryObjectReceiveStream(state))


class create_broadcast_object_stream(BroadcastObjectSendStream[T_Item]):
    """
    Create a broadcast object stream.

    Unlike :func:`create_memory_object_stream`, only a send side is returned.
    Subscribers are added dynamically with
    :meth:`~anyio.streams.broadcast.BroadcastObjectSendStream.subscribe`, each
    receiving its own independent, bounded buffer. Every item published on the
    send stream is delivered to every live subscription (not distributed
    between them).

    The stream's item type can be annotated like
    ``create_broadcast_object_stream[T_Item]``.

    :return: the broadcast send stream

    """

    def __new__(cls) -> BroadcastObjectSendStream[T_Item]:  # type: ignore[misc]
        state = _BroadcastState[T_Item]()
        return BroadcastObjectSendStream(state)
