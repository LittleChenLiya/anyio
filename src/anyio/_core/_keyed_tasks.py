from __future__ import annotations

import sys
from collections.abc import Callable, Coroutine
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, cast

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from ..abc._tasks import call_for_coroutine
from ._eventloop import get_cancelled_exc_class
from ._tasks import TaskHandle, create_task_group

if TYPE_CHECKING:
    from ..abc import TaskGroup

K = TypeVar("K")
T = TypeVar("T")


class KeyedTaskGroup(Generic[K, T]):
    """
    A component on top of :class:`TaskGroup` that runs at most one task per key.

    Starting a task for a key that already has a running task cancels the previous
    task and starts a new *generation* for that key. Tasks for different keys run
    concurrently.

    Only the current generation for a key may publish its outcome. If an old task
    shields against cancellation and finishes late, its result (or exception, or
    cancellation) is silently discarded — it can never override the outcome of the
    current generation.

    Outcomes are reported through three separate callbacks, so a cancellation is
    never confused with a ``None`` result:

    * ``on_result(key, value)`` — the task finished successfully with ``value``
    * ``on_exception(key, exc)`` — the task raised exception ``exc``
    * ``on_cancel(key)`` — the task was cancelled

    The component is an asynchronous context manager. On :meth:`__aexit__` it closes
    itself to new tasks and waits for all in-flight tasks to finish.

    Lifecycle::

        async with KeyedTaskGroup(on_result=..., on_exception=..., on_cancel=...) as ktg:
            ktg.start("a", task_a)
            ktg.start("b", task_b)   # different key, runs in parallel
            ktg.start("a", task_a2)  # cancels the old "a" task, starts a new generation
        # closed: no new tasks, waits for in-flight tasks to finish

    :param task_group: the task group to spawn tasks in; if omitted, a new task group
        is created and owned by this component
    :param on_result: callback invoked with the key and return value when a task
        finishes successfully
    :param on_exception: callback invoked with the key and exception when a task
        raises an exception
    :param on_cancel: callback invoked with the key when a task is cancelled
    """

    __slots__ = (
        "_closed",
        "_entered",
        "_generations",
        "_handles",
        "_on_cancel",
        "_on_exception",
        "_on_result",
        "_task_group",
    )

    def __init__(
        self,
        task_group: TaskGroup | None = None,
        *,
        on_result: Callable[[K, T], Any] | None = None,
        on_exception: Callable[[K, BaseException], Any] | None = None,
        on_cancel: Callable[[K], Any] | None = None,
    ) -> None:
        self._task_group = task_group if task_group is not None else create_task_group()
        self._on_result = on_result
        self._on_exception = on_exception
        self._on_cancel = on_cancel
        self._generations: dict[K, int] = {}
        self._handles: dict[K, TaskHandle[T]] = {}
        self._closed = False
        self._entered = False

    @property
    def closed(self) -> bool:
        """``True`` if the component has been closed and no new tasks can be started."""
        return self._closed

    def cancel(self, reason: str | None = None) -> None:
        """
        Cancel all tasks in this component.

        This is a shortcut for cancelling the underlying task group's cancel scope.

        :param reason: a message describing the reason for the cancellation
        """
        self._task_group.cancel(reason)

    def start(
        self,
        key: K,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
    ) -> TaskHandle[T]:
        """
        Start a new generation of the task for ``key``.

        If a task for ``key`` is already running, it is cancelled before the new
        generation is started. The old task may still finish late (e.g. because it
        shielded against cancellation), but its outcome is discarded.

        :param key: the key to start the task for
        :param func: a coroutine function
        :param args: positional arguments to call the function with
        :return: a task handle for the new generation
        :raises RuntimeError: if the component is closed or not yet active
        """
        if self._closed:
            raise RuntimeError(
                "This KeyedTaskGroup is closed; no new tasks can be started."
            )
        if not self._entered:
            raise RuntimeError(
                "This KeyedTaskGroup is not active; no new tasks can be started."
            )

        generation = self._generations.get(key, 0) + 1
        self._generations[key] = generation

        old_handle = self._handles.get(key)
        if old_handle is not None:
            old_handle.cancel()

        handle = self._task_group.start_soon(self._run, key, generation, func, args)
        self._handles[key] = cast(TaskHandle[T], handle)
        return cast(TaskHandle[T], handle)

    async def _run(
        self,
        key: K,
        generation: int,
        func: Callable[..., Coroutine[Any, Any, T]],
        args: tuple[Any, ...],
    ) -> None:
        coro = call_for_coroutine(func, args)
        try:
            result = await coro
        except BaseException as exc:
            if self._is_current(key, generation):
                if isinstance(exc, get_cancelled_exc_class()):
                    if self._on_cancel is not None:
                        self._on_cancel(key)
                elif self._on_exception is not None:
                    self._on_exception(key, exc)
            # else: a stale generation finished late; discard its outcome
        else:
            if self._is_current(key, generation) and self._on_result is not None:
                self._on_result(key, result)

    def _is_current(self, key: K, generation: int) -> bool:
        return self._generations.get(key) == generation

    async def __aenter__(self) -> Self:
        await self._task_group.__aenter__()
        self._entered = True
        self._closed = False
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self._closed = True
        self._entered = False
        return await self._task_group.__aexit__(exc_type, exc_val, exc_tb)
