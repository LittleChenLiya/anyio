from __future__ import annotations

__all__ = ("KeyedTaskRunner",)

from collections.abc import Awaitable, Callable, Hashable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from ._eventloop import get_cancelled_exc_class
from ._exceptions import ClosedResourceError
from ._synchronization import Event
from ._tasks import CancelScope

if TYPE_CHECKING:
    from ..abc import TaskGroup

T = TypeVar("T")

#: called as ``callback(key, generation, result)`` when the current generation of
#: ``key`` produces a result
ResultCallback = Callable[[Hashable, int, Any], None]

#: called as ``callback(key, generation, exception)`` when the current generation of
#: ``key`` raises an exception
ExceptionCallback = Callable[[Hashable, int, BaseException], None]

#: called as ``callback(key, generation)`` when a task is observed to finish via
#: cancellation (regardless of whether its generation was still current)
CancelledCallback = Callable[[Hashable, int], None]


class KeyedTaskRunner(Generic[T]):
    """
    Runs keyed tasks on a task group, with per-key superseding semantics.

    Each task is started under a *key*. Starting a new task with a key that already
    has a task running cancels the previous task and bumps the key's *generation*
    counter. Tasks with different keys always run in parallel.

    Only the current generation of a key holds publishing rights: if a superseded
    task shields itself from cancellation and finishes (or fails) late, its result
    or exception is silently discarded and can never overwrite the outcome of a
    newer generation. Cancellation, however, is always reported for the task that
    was actually cancelled, so callers can observe the superseding itself.

    The three outcomes are reported through three separate callbacks (never
    conflated into a single nullable value):

    * ``on_result(key, generation, result)``
    * ``on_exception(key, generation, exception)``
    * ``on_cancelled(key, generation)``

    .. rubric:: Lifecycle

    #. Create the runner bound to an open task group, either directly or via
       ``async with KeyedTaskRunner(task_group, ...) as runner:``.
    #. Schedule work with :meth:`start`. It is synchronous (like
       :meth:`TaskGroup.start_soon`) and returns the new generation number, which
       increases monotonically per key.
    #. While running, :meth:`cancel` may be used to cancel a single key's current
       task, and :meth:`wait_idle` to wait until no tasks are in flight.
    #. Closing the runner with :meth:`aclose` (or by exiting the ``async with``
       block) is terminal: further :meth:`start` calls raise
       :exc:`~anyio.ClosedResourceError`, all in-flight tasks are cancelled, and
       ``aclose()`` only returns once every in-flight task has finished its
       cleanup (including shielded cleanup code). A closed runner cannot be
       reused.

    The callbacks are invoked synchronously from within the runner's tasks and
    must not raise; if they do, the exception propagates to the host task group.

    :param task_group: the task group to host the keyed tasks in
    :param on_result: called when the current generation of a key produces a
        result
    :param on_exception: called when the current generation of a key raises an
        exception (other than cancellation)
    :param on_cancelled: called when a task finishes via cancellation
    """

    def __init__(
        self,
        task_group: TaskGroup,
        *,
        on_result: ResultCallback | None = None,
        on_exception: ExceptionCallback | None = None,
        on_cancelled: CancelledCallback | None = None,
    ) -> None:
        self._task_group = task_group
        self._on_result = on_result
        self._on_exception = on_exception
        self._on_cancelled = on_cancelled
        self._generations: dict[Hashable, int] = {}
        self._scopes: dict[Hashable, CancelScope] = {}
        self._inflight = 0
        self._idle_event: Event | None = None
        self._closed = False

    async def __aenter__(self) -> KeyedTaskRunner[T]:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    @property
    def closed(self) -> bool:
        """Whether the runner has been closed."""
        return self._closed

    def generation(self, key: Hashable) -> int:
        """Return the current generation of ``key`` (0 if never started)."""
        return self._generations.get(key, 0)

    def start(
        self, key: Hashable, func: Callable[..., Awaitable[T]], /, *args: Any
    ) -> int:
        """
        Schedule ``func(*args)`` as the new current task for ``key``.

        Any task previously started with the same key is cancelled first; tasks
        with other keys are unaffected and keep running in parallel.

        :return: the generation number assigned to the new task
        :raises ClosedResourceError: if the runner has been closed
        """
        if self._closed:
            raise ClosedResourceError(
                f"this keyed task runner is closed; cannot start key {key!r}"
            )

        generation = self._generations.get(key, 0) + 1
        self._generations[key] = generation
        if (old_scope := self._scopes.get(key)) is not None:
            old_scope.cancel()

        scope = CancelScope()
        self._scopes[key] = scope
        self._inflight += 1
        self._task_group.start_soon(self._run, key, generation, scope, func, args)
        return generation

    def cancel(self, key: Hashable) -> bool:
        """
        Cancel the current task of ``key``, if any.

        The key's generation is *not* bumped, so if the cancelled task ignores
        the cancellation and finishes anyway, it still holds publishing rights.

        :return: ``True`` if there was an in-flight task to cancel
        """
        if (scope := self._scopes.get(key)) is not None:
            scope.cancel()
            return True

        return False

    async def wait_idle(self) -> None:
        """Wait until no keyed tasks are in flight."""
        while self._inflight:
            if self._idle_event is None:
                self._idle_event = Event()

            await self._idle_event.wait()

    async def aclose(self) -> None:
        """
        Close the runner.

        Stops accepting new tasks, cancels all in-flight tasks and waits for
        every one of them to finish, including any shielded cleanup they run
        after being cancelled. Idempotent.
        """
        if self._closed:
            return

        self._closed = True
        for scope in list(self._scopes.values()):
            scope.cancel()

        await self.wait_idle()

    def _is_current(self, key: Hashable, generation: int, scope: CancelScope) -> bool:
        return (
            self._generations.get(key) == generation and self._scopes.get(key) is scope
        )

    async def _run(
        self,
        key: Hashable,
        generation: int,
        scope: CancelScope,
        func: Callable[..., Awaitable[T]],
        args: tuple[Any, ...],
    ) -> None:
        cancelled_exc_class = get_cancelled_exc_class()
        cancelled = False
        result: Any = None
        exception: Exception | None = None
        try:
            with scope:
                try:
                    result = await func(*args)
                except cancelled_exc_class:
                    # Record the cancellation here because the scope may absorb
                    # it on the way out (when the cancellation was its own).
                    cancelled = True
                    raise
                except Exception as exc:
                    exception = exc
        finally:
            if cancelled:
                # Cancellation is always reported for the task it happened to,
                # even if its generation has already been superseded.
                if self._on_cancelled is not None:
                    self._on_cancelled(key, generation)
            elif self._is_current(key, generation, scope):
                # Publishing rights belong to the current generation only: a
                # superseded task that shielded the cancellation and finished
                # (or failed) late must not overwrite the new generation's
                # outcome.
                if exception is not None:
                    if self._on_exception is not None:
                        self._on_exception(key, generation, exception)
                elif self._on_result is not None:
                    self._on_result(key, generation, result)

            self._inflight -= 1
            if self._scopes.get(key) is scope:
                del self._scopes[key]

            if self._inflight == 0 and self._idle_event is not None:
                idle_event, self._idle_event = self._idle_event, None
                idle_event.set()
