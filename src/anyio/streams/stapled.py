from __future__ import annotations

import sys

__all__ = (
    "HalfCloseByteStream",
    "MultiListener",
    "StapledByteStream",
    "StapledObjectStream",
)

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from .. import CancelScope, ClosedResourceError
from ..abc import (
    ByteReceiveStream,
    ByteSendStream,
    ByteStream,
    Listener,
    ObjectReceiveStream,
    ObjectSendStream,
    ObjectStream,
    TaskGroup,
)
from ..lowlevel import checkpoint

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup

T_Item = TypeVar("T_Item")
T_Stream = TypeVar("T_Stream")


@dataclass(eq=False)
class StapledByteStream(ByteStream):
    """
    Combines two byte streams into a single, bidirectional byte stream.

    Extra attributes will be provided from both streams, with the receive stream
    providing the values in case of a conflict.

    :param ByteSendStream send_stream: the sending byte stream
    :param ByteReceiveStream receive_stream: the receiving byte stream
    """

    send_stream: ByteSendStream
    receive_stream: ByteReceiveStream

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")

        return await self.receive_stream.receive(max_bytes)

    async def send(self, item: bytes) -> None:
        await self.send_stream.send(item)

    async def send_eof(self) -> None:
        await self.send_stream.aclose()

    async def aclose(self) -> None:
        await self.send_stream.aclose()
        await self.receive_stream.aclose()

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return {
            **self.send_stream.extra_attributes,
            **self.receive_stream.extra_attributes,
        }


@dataclass(eq=False)
class HalfCloseByteStream(StapledByteStream):
    """
    A :class:`~.abc.ByteStream` adapter that combines a
    :class:`~.abc.ByteSendStream` and a :class:`~.abc.ByteReceiveStream` into a
    single bidirectional byte stream with explicit half-close semantics.

    Unlike :class:`StapledByteStream`, this adapter keeps explicit track of the
    closed state of both directions:

    * :meth:`send_eof` only closes the send stream; the receive stream keeps
      working until :meth:`aclose` is called.
    * :meth:`aclose` closes both streams, but always attempts to close the
      receive stream even if closing the send stream raised an exception. If
      both closes raise, the exceptions are raised together as an exception
      group.
    * Neither :meth:`send_eof` nor :meth:`aclose` will call the underlying
      streams' ``aclose()`` more than once.
    * If :meth:`aclose` is cancelled while closing the streams, the cleanup of
      both streams is still completed before the cancellation is propagated to
      the caller.

    :param ByteSendStream send_stream: the sending byte stream
    :param ByteReceiveStream receive_stream: the receiving byte stream
    """

    _send_closed: bool = field(init=False, default=False)
    _receive_closed: bool = field(init=False, default=False)

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")

        if self._receive_closed:
            raise ClosedResourceError

        return await self.receive_stream.receive(max_bytes)

    async def send(self, item: bytes) -> None:
        if self._send_closed or self._receive_closed:
            raise ClosedResourceError

        await self.send_stream.send(item)

    async def send_eof(self) -> None:
        if self._send_closed:
            return

        self._send_closed = True
        await self.send_stream.aclose()

    async def aclose(self) -> None:
        if self._receive_closed:
            return

        errors: list[BaseException] = []
        with CancelScope(shield=True):
            if not self._send_closed:
                self._send_closed = True
                try:
                    await self.send_stream.aclose()
                except BaseException as exc:
                    errors.append(exc)

            if not self._receive_closed:
                self._receive_closed = True
                try:
                    await self.receive_stream.aclose()
                except BaseException as exc:
                    errors.append(exc)

        if errors:
            if len(errors) == 1:
                raise errors[0]

            raise BaseExceptionGroup(
                "Multiple exceptions occurred while closing the byte stream",
                errors,
            )

        # If the enclosing scope was cancelled while we were closing, propagate the
        # cancellation now that the cleanup has finished
        await checkpoint()

    @property
    def send_closed(self) -> bool:
        """``True`` if the send side of the stream has been closed."""
        return self._send_closed

    @property
    def receive_closed(self) -> bool:
        """``True`` if the receive side of the stream has been closed."""
        return self._receive_closed


@dataclass(eq=False)
class StapledObjectStream(ObjectStream[T_Item], Generic[T_Item]):
    """
    Combines two object streams into a single, bidirectional object stream.

    Extra attributes will be provided from both streams, with the receive stream
    providing the values in case of a conflict.

    :param ObjectSendStream send_stream: the sending object stream
    :param ObjectReceiveStream receive_stream: the receiving object stream
    """

    send_stream: ObjectSendStream[T_Item]
    receive_stream: ObjectReceiveStream[T_Item]

    async def receive(self) -> T_Item:
        return await self.receive_stream.receive()

    async def send(self, item: T_Item) -> None:
        await self.send_stream.send(item)

    def send_nowait(self, item: T_Item) -> None:
        try:
            send_nowait = self.send_stream.send_nowait  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise NotImplementedError(
                f"'send_nowait' method not implemented in {type(self.send_stream)}"
            ) from exc

        send_nowait(item)

    async def send_eof(self) -> None:
        await self.send_stream.aclose()

    async def aclose(self) -> None:
        await self.send_stream.aclose()
        await self.receive_stream.aclose()

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return {
            **self.send_stream.extra_attributes,
            **self.receive_stream.extra_attributes,
        }


@dataclass(eq=False)
class MultiListener(Listener[T_Stream], Generic[T_Stream]):
    """
    Combines multiple listeners into one, serving connections from all of them at once.

    Any MultiListeners in the given collection of listeners will have their listeners
    moved into this one.

    Extra attributes are provided from each listener, with each successive listener
    overriding any conflicting attributes from the previous one.

    :param listeners: listeners to serve
    :type listeners: Sequence[Listener[T_Stream]]
    """

    listeners: Sequence[Listener[T_Stream]]

    def __post_init__(self) -> None:
        listeners: list[Listener[T_Stream]] = []
        for listener in self.listeners:
            if isinstance(listener, MultiListener):
                listeners.extend(listener.listeners)
                del listener.listeners[:]  # type: ignore[attr-defined]
            else:
                listeners.append(listener)

        self.listeners = listeners

    async def serve(
        self, handler: Callable[[T_Stream], Any], task_group: TaskGroup | None = None
    ) -> None:
        from .. import create_task_group

        async with create_task_group() as tg:
            for listener in self.listeners:
                tg.start_soon(listener.serve, handler, task_group)

    async def aclose(self) -> None:
        for listener in self.listeners:
            await listener.aclose()

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        attributes: dict = {}
        for listener in self.listeners:
            attributes.update(listener.extra_attributes)

        return attributes
