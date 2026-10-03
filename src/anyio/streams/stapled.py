from __future__ import annotations

__all__ = (
    "HalfCloseableStapledByteStream",
    "MultiListener",
    "StapledByteStream",
    "StapledObjectStream",
)

import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from .. import CancelScope, ClosedResourceError, get_cancelled_exc_class
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

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

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
class HalfCloseableStapledByteStream(StapledByteStream):
    """
    Combines two byte streams into a single bidirectional byte stream with
    explicit half-close support.

    Unlike :class:`StapledByteStream`, this adapter keeps track of the close
    state of each direction separately:

    * :meth:`send_eof` closes only the send side, leaving the receive side open
      so that any remaining incoming data can still be read.
    * :meth:`aclose` attempts to close both sides, even if closing the first one
      fails. It is idempotent: the underlying streams are closed at most once
      each, no matter how many times it is called.
    * If :meth:`aclose` is cancelled, the remaining cleanup is finished in a
      shielded scope before the cancellation is re-raised to the caller.

    Extra attributes will be provided from both streams, with the receive stream
    providing the values in case of a conflict.

    :param ByteSendStream send_stream: the sending byte stream
    :param ByteReceiveStream receive_stream: the receiving byte stream
    """

    _send_closed: bool = field(init=False, default=False)
    _receive_closed: bool = field(init=False, default=False)

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self._receive_closed:
            raise ClosedResourceError

        return await super().receive(max_bytes)

    async def send(self, item: bytes) -> None:
        if self._send_closed:
            raise ClosedResourceError

        await self.send_stream.send(item)

    async def send_eof(self) -> None:
        if self._send_closed:
            return

        self._send_closed = True
        await self.send_stream.aclose()

    async def aclose(self) -> None:
        try:
            await self._close_streams()
        except get_cancelled_exc_class():
            # Finish the remaining cleanup in a shielded scope, then pass the
            # cancellation on to the caller
            with CancelScope(shield=True):
                await self._close_streams()

            raise

    async def _close_streams(self) -> None:
        errors: list[Exception] = []
        if not self._send_closed:
            self._send_closed = True
            try:
                await self.send_stream.aclose()
            except Exception as exc:
                errors.append(exc)

        if not self._receive_closed:
            self._receive_closed = True
            try:
                await self.receive_stream.aclose()
            except Exception as exc:
                errors.append(exc)

        if len(errors) == 1:
            raise errors[0]
        elif errors:
            raise ExceptionGroup("errors while closing the stapled stream", errors)

    @property
    def send_closed(self) -> bool:
        """Return ``True`` if the send side has been closed."""
        return self._send_closed

    @property
    def receive_closed(self) -> bool:
        """Return ``True`` if the receive side has been closed."""
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
