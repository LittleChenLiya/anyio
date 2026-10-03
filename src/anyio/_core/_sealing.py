"""
A sealing wrapper for :class:`~anyio.CapacityLimiter`.

This module provides :class:`SealableCapacityLimiter`, a wrapper that adds a
*seal* operation on top of a regular capacity limiter. Sealing is a one-way
gate typically used at the end of a batch-processing run:

* before :meth:`~SealableCapacityLimiter.seal` is called, the wrapper behaves
  exactly like the wrapped limiter;
* once sealed, every new acquire (waiting or not) is rejected immediately with
  :exc:`LimiterSealedError`, and any tasks already queued in ``acquire()`` are
  woken up and rejected as well;
* tokens borrowed *before* the seal stay valid - sealing never revokes them;
* :meth:`~SealableCapacityLimiter.wait_drained` waits until all such
  pre-existing borrows have been released. Cancelling a ``wait_drained()``
  call only cancels that waiter; it never releases tokens held by others.

Each acquire racing with :meth:`~SealableCapacityLimiter.seal` ends with
exactly one outcome: it either holds a token (and is then tracked as a regular
borrower) or raises :exc:`LimiterSealedError`. The wrapped
:class:`~anyio.CapacityLimiter` itself is not modified in any way.

Short example (printing the state while draining)::

    import anyio
    from anyio._core._sealing import SealableCapacityLimiter

    async def main() -> None:
        limiter = SealableCapacityLimiter(total_tokens=1)
        async with limiter:                 # borrow the only token
            limiter.seal()                  # close the gate for new work
            print(limiter)                  # printable state
        await limiter.wait_drained()        # returns: nothing is borrowed
        print(limiter.statistics())

    anyio.run(main)
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType

from ._synchronization import CapacityLimiter, Event
from ._tasks import CancelScope


class LimiterSealedError(Exception):
    """Raised when trying to acquire a token from a sealed limiter."""


@dataclass(frozen=True)
class SealableCapacityLimiterStatistics:
    """
    Statistics snapshot of a :class:`SealableCapacityLimiter`.

    :ivar bool sealed: ``True`` if the limiter has been sealed and no longer
        accepts new borrows
    :ivar int borrowed_tokens: number of tokens currently borrowed from the
        underlying limiter
    :ivar float total_tokens: total number of tokens available for borrowing
    :ivar int acquire_waiters: number of tasks currently inside an
        ``acquire()`` call (queued waiting for a token, or about to borrow
        one)
    :ivar int drain_waiters: number of tasks currently waiting in
        :meth:`SealableCapacityLimiter.wait_drained`
    """

    sealed: bool
    borrowed_tokens: int
    total_tokens: float
    acquire_waiters: int
    drain_waiters: int


class SealableCapacityLimiter:
    """
    A wrapper around :class:`~anyio.CapacityLimiter` that can be sealed.

    Sealing stops new work from entering while letting already-borrowed tokens
    run to completion, which is useful for gracefully draining a batch of work.

    :param limiter: an existing capacity limiter to wrap. If omitted, a new
        :class:`~anyio.CapacityLimiter` is created with ``total_tokens``.
    :param total_tokens: total number of tokens for the implicitly created
        limiter (mutually exclusive with ``limiter``)

    .. note:: Tokens must be released through this wrapper (``release()`` /
        ``release_on_behalf_of()`` / the async context manager) so that
        :meth:`wait_drained` is notified when the last borrow is released.
    """

    def __init__(
        self,
        limiter: CapacityLimiter | None = None,
        *,
        total_tokens: float | None = None,
    ) -> None:
        if limiter is None:
            if total_tokens is None:
                raise TypeError("pass either a limiter or total_tokens")

            limiter = CapacityLimiter(total_tokens)
        elif total_tokens is not None:
            raise TypeError("pass either a limiter or total_tokens, not both")

        self._limiter = limiter
        self._sealed = False
        self._acquire_scopes: set[CancelScope] = set()
        self._drain_waiters: list[Event] = []

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()

    def __str__(self) -> str:
        statistics = self.statistics()
        return (
            f"{self.__class__.__name__}(sealed={statistics.sealed}, "
            f"borrowed={statistics.borrowed_tokens}/{statistics.total_tokens}, "
            f"acquire_waiters={statistics.acquire_waiters}, "
            f"drain_waiters={statistics.drain_waiters})"
        )

    @property
    def limiter(self) -> CapacityLimiter:
        """The wrapped capacity limiter."""
        return self._limiter

    @property
    def sealed(self) -> bool:
        """``True`` if this limiter has been sealed."""
        return self._sealed

    @property
    def total_tokens(self) -> float:
        """The total number of tokens available for borrowing (read-write)."""
        return self._limiter.total_tokens

    @total_tokens.setter
    def total_tokens(self, value: float) -> None:
        self._limiter.total_tokens = value

    @property
    def borrowed_tokens(self) -> int:
        """The number of tokens that have currently been borrowed."""
        return self._limiter.borrowed_tokens

    @property
    def available_tokens(self) -> float:
        """The number of tokens currently available to be borrowed."""
        return self._limiter.available_tokens

    def seal(self) -> None:
        """
        Seal the limiter, rejecting all current and future acquire attempts.

        Tasks already waiting in :meth:`acquire` are woken up and fail with
        :exc:`LimiterSealedError`. Tokens borrowed before the seal remain
        borrowed; use :meth:`wait_drained` to wait for them to be released.

        This method is idempotent: sealing an already sealed limiter has no
        effect.
        """
        if self._sealed:
            return

        self._sealed = True
        for scope in self._acquire_scopes:
            scope.cancel()

    def _raise_if_sealed(self) -> None:
        if self._sealed:
            raise LimiterSealedError(
                "the limiter has been sealed and no longer accepts new borrows"
            )

    def acquire_nowait(self) -> None:
        """
        Acquire a token for the current task without waiting.

        :raises LimiterSealedError: if the limiter has been sealed
        :raises ~anyio.WouldBlock: if there are no tokens available
        """
        self._raise_if_sealed()
        self._limiter.acquire_nowait()

    def acquire_on_behalf_of_nowait(self, borrower: object) -> None:
        """
        Acquire a token on behalf of ``borrower`` without waiting.

        :raises LimiterSealedError: if the limiter has been sealed
        :raises ~anyio.WouldBlock: if there are no tokens available
        """
        self._raise_if_sealed()
        self._limiter.acquire_on_behalf_of_nowait(borrower)

    async def acquire(self) -> None:
        """
        Acquire a token for the current task, waiting if necessary.

        If the limiter gets sealed while waiting, the wait is aborted.

        :raises LimiterSealedError: if the limiter is (or becomes) sealed
            before a token could be borrowed
        """
        self._raise_if_sealed()
        # The scope is registered before any checkpoint, so a concurrent
        # seal() either sees it (and cancels it) or the acquire completes
        # first. Either way the request gets exactly one outcome.
        with CancelScope() as scope:
            self._acquire_scopes.add(scope)
            try:
                await self._limiter.acquire()
            finally:
                self._acquire_scopes.discard(scope)

        if scope.cancelled_caught:
            # Cancelled by seal(), not by an enclosing scope (an external
            # cancellation would have been re-raised by the scope itself).
            raise LimiterSealedError(
                "the limiter was sealed while waiting for a token"
            ) from None

    async def acquire_on_behalf_of(self, borrower: object) -> None:
        """
        Acquire a token on behalf of ``borrower``, waiting if necessary.

        :raises LimiterSealedError: if the limiter is (or becomes) sealed
            before a token could be borrowed
        """
        self._raise_if_sealed()
        with CancelScope() as scope:
            self._acquire_scopes.add(scope)
            try:
                await self._limiter.acquire_on_behalf_of(borrower)
            finally:
                self._acquire_scopes.discard(scope)

        if scope.cancelled_caught:
            raise LimiterSealedError(
                "the limiter was sealed while waiting for a token"
            ) from None

    def release(self) -> None:
        """
        Release the token held by the current task.

        :raises RuntimeError: if the current task has not borrowed a token
        """
        self._limiter.release()
        self._notify_drain_waiters()

    def release_on_behalf_of(self, borrower: object) -> None:
        """
        Release the token held by ``borrower``.

        :raises RuntimeError: if the borrower has not borrowed a token
        """
        self._limiter.release_on_behalf_of(borrower)
        self._notify_drain_waiters()

    def _notify_drain_waiters(self) -> None:
        if self._limiter.borrowed_tokens == 0 and self._drain_waiters:
            waiters, self._drain_waiters = self._drain_waiters, []
            for event in waiters:
                event.set()

    async def wait_drained(self) -> None:
        """
        Wait until no tokens are borrowed anymore.

        This only waits for *existing* borrows to be released; it does not
        prevent new ones, so it is usually called after :meth:`seal`.
        Cancelling this coroutine only cancels the waiter - tokens borrowed by
        other tasks are never released as a side effect.
        """
        while self._limiter.borrowed_tokens > 0:
            event = Event()
            self._drain_waiters.append(event)
            try:
                await event.wait()
            except BaseException:
                if event in self._drain_waiters:
                    self._drain_waiters.remove(event)

                raise

    def statistics(self) -> SealableCapacityLimiterStatistics:
        """Return statistics about the current state of this limiter."""
        return SealableCapacityLimiterStatistics(
            sealed=self._sealed,
            borrowed_tokens=self._limiter.borrowed_tokens,
            total_tokens=self._limiter.total_tokens,
            acquire_waiters=len(self._acquire_scopes),
            drain_waiters=len(self._drain_waiters),
        )
