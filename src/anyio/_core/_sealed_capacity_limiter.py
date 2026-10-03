from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING

from ._eventloop import get_async_backend
from ._exceptions import WouldBlock
from ._testing import get_current_task

if TYPE_CHECKING:
    from ._synchronization import CapacityLimiter, Event


class CapacityLimiterSealed(Exception):
    """
    Raised when a task tries to acquire a token from a sealed
    :class:`SealedCapacityLimiter`.

    Once a limiter has been sealed, every attempt to acquire a token -- whether
    through :meth:`~SealedCapacityLimiter.acquire`,
    :meth:`~SealedCapacityLimiter.acquire_on_behalf_of` or the ``nowait``
    variants -- fails immediately with this exception. Tasks that were already
    waiting when the limiter was sealed are also woken up and rejected.

    """


@dataclass(frozen=True)
class SealedCapacityLimiterStatistics:
    """
    Statistics for a :class:`SealedCapacityLimiter`.

    :ivar bool sealed: ``True`` if :meth:`~SealedCapacityLimiter.seal` has been
        called on this limiter
    :ivar int borrowed_tokens: number of tokens currently borrowed by tasks
    :ivar float total_tokens: total number of available tokens
    :ivar tuple borrowers: tasks or other objects currently holding tokens
        borrowed from this limiter
    :ivar int tasks_waiting: number of tasks waiting on
        :meth:`~SealedCapacityLimiter.acquire` or
        :meth:`~SealedCapacityLimiter.acquire_on_behalf_of`
    :ivar int tasks_draining: number of tasks waiting on
        :meth:`~SealedCapacityLimiter.wait_drained`
    """

    sealed: bool
    borrowed_tokens: int
    total_tokens: float
    borrowers: tuple[object, ...]
    tasks_waiting: int
    tasks_draining: int


@dataclass
class _Waiter:
    borrower: object
    event: Event
    granted: bool = False


class SealedCapacityLimiter:
    """
    A wrapper around :class:`CapacityLimiter` that can be sealed to block new
    work from entering while allowing work that already holds a token to
    finish.

    After :meth:`seal` has been called, any attempt to acquire a token raises
    :class:`CapacityLimiterSealed` immediately. Tasks that were already
    waiting for a token when the limiter was sealed are woken up and rejected
    as well. Existing borrowers keep their tokens and are expected to release
    them normally.

    :param limiter: the capacity limiter to wrap

    .. versionadded:: 4.13
    """

    __slots__ = "_drain_waiters", "_limiter", "_sealed", "_waiters"

    def __init__(self, limiter: CapacityLimiter) -> None:
        self._limiter = limiter
        self._sealed = False
        self._waiters: deque[_Waiter] = deque()
        self._drain_waiters: deque[Event] = deque()

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()

    @property
    def total_tokens(self) -> float:
        """
        The total number of tokens available for borrowing.

        This is a read-write property. Increasing the number of tokens wakes up
        the appropriate number of waiting tasks.

        """
        return self._limiter.total_tokens

    @total_tokens.setter
    def total_tokens(self, value: float) -> None:
        self._limiter.total_tokens = value
        self._grant_waiters()

    @property
    def borrowed_tokens(self) -> int:
        """The number of tokens that have currently been borrowed."""
        return self._limiter.borrowed_tokens

    @property
    def available_tokens(self) -> float:
        """The number of tokens currently available to be borrowed."""
        return self._limiter.available_tokens

    @property
    def sealed(self) -> bool:
        """``True`` if :meth:`seal` has been called on this limiter."""
        return self._sealed

    def seal(self) -> None:
        """
        Seal the limiter, rejecting all future acquire attempts.

        Tasks that are currently waiting for a token are woken up and rejected.
        Existing borrowers are unaffected. Calling this method more than once
        has no effect.

        """
        if self._sealed:
            return

        self._sealed = True
        for waiter in self._waiters:
            waiter.event.set()

        self._waiters.clear()

    async def acquire(self) -> None:
        """
        Acquire a token for the current task, waiting if necessary for one to
        become available.

        :raises CapacityLimiterSealed: if the limiter has been sealed

        """
        await self.acquire_on_behalf_of(get_current_task())

    async def acquire_on_behalf_of(self, borrower: object) -> None:
        """
        Acquire a token, waiting if necessary for one to become available.

        :param borrower: the entity borrowing a token
        :raises CapacityLimiterSealed: if the limiter has been sealed

        """
        if self._sealed:
            raise CapacityLimiterSealed(
                "cannot acquire a token from a sealed capacity limiter"
            )

        try:
            self._limiter.acquire_on_behalf_of_nowait(borrower)
            return
        except WouldBlock:
            pass

        waiter = _Waiter(borrower, get_async_backend().create_event())
        self._waiters.append(waiter)
        try:
            await waiter.event.wait()
        except BaseException:
            if waiter.granted:
                # We were granted a token but were cancelled (or otherwise
                # interrupted) before we could claim it -- give it back.
                self._limiter.release_on_behalf_of(waiter.borrower)
                self._grant_waiters()
            else:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass

            raise

        if waiter.granted:
            return

        raise CapacityLimiterSealed(
            "cannot acquire a token from a sealed capacity limiter"
        )

    def acquire_nowait(self) -> None:
        """
        Acquire a token for the current task without waiting for one to become
        available.

        :raises ~anyio.WouldBlock: if there are no tokens available for borrowing
        :raises CapacityLimiterSealed: if the limiter has been sealed

        """
        self.acquire_on_behalf_of_nowait(get_current_task())

    def acquire_on_behalf_of_nowait(self, borrower: object) -> None:
        """
        Acquire a token without waiting for one to become available.

        :param borrower: the entity borrowing a token
        :raises ~anyio.WouldBlock: if there are no tokens available for borrowing
        :raises CapacityLimiterSealed: if the limiter has been sealed

        """
        if self._sealed:
            raise CapacityLimiterSealed(
                "cannot acquire a token from a sealed capacity limiter"
            )

        self._limiter.acquire_on_behalf_of_nowait(borrower)

    def release(self) -> None:
        """
        Release the token held by the current task.

        :raises RuntimeError: if the current task has not borrowed a token from
            this limiter.

        """
        self.release_on_behalf_of(get_current_task())

    def release_on_behalf_of(self, borrower: object) -> None:
        """
        Release the token held by the given borrower.

        :raises RuntimeError: if the borrower has not borrowed a token from
            this limiter.

        """
        self._limiter.release_on_behalf_of(borrower)
        self._grant_waiters()
        self._notify_drain_waiters()

    async def wait_drained(self) -> None:
        """
        Wait until all currently borrowed tokens have been released.

        If no tokens are borrowed when this method is called, it returns
        immediately.

        """
        if self._limiter.borrowed_tokens == 0:
            return

        event = get_async_backend().create_event()
        self._drain_waiters.append(event)
        try:
            await event.wait()
        except BaseException:
            try:
                self._drain_waiters.remove(event)
            except ValueError:
                pass

            raise

    def statistics(self) -> SealedCapacityLimiterStatistics:
        """
        Return statistics about the current state of this limiter.

        """
        base = self._limiter.statistics()
        return SealedCapacityLimiterStatistics(
            sealed=self._sealed,
            borrowed_tokens=base.borrowed_tokens,
            total_tokens=base.total_tokens,
            borrowers=base.borrowers,
            tasks_waiting=len(self._waiters),
            tasks_draining=len(self._drain_waiters),
        )

    def _grant_waiters(self) -> None:
        if self._sealed:
            return

        while self._waiters:
            waiter = self._waiters[0]
            try:
                self._limiter.acquire_on_behalf_of_nowait(waiter.borrower)
            except WouldBlock:
                return

            self._waiters.popleft()
            waiter.granted = True
            waiter.event.set()

    def _notify_drain_waiters(self) -> None:
        if self._limiter.borrowed_tokens == 0:
            for event in self._drain_waiters:
                event.set()

            self._drain_waiters.clear()
