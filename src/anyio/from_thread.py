from __future__ import annotations

__all__ = (
    "BlockingPortal",
    "BlockingPortalProvider",
    "CompletionUnit",
    "CompletionUnitFailedError",
    "CompletionUnitUnreachableError",
    "check_cancelled",
    "run",
    "run_sync",
    "start_blocking_portal",
)

import sys
from collections.abc import Awaitable, Callable, Coroutine, Generator
from concurrent.futures import Future
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    contextmanager,
)
from contextvars import Context
from dataclasses import dataclass, field
from functools import partial
from inspect import isawaitable
from threading import Lock, Thread, current_thread, get_ident
from types import TracebackType
from typing import (
    Any,
    Generic,
    NamedTuple,
    TypeVar,
    cast,
    overload,
)

from ._core._eventloop import (
    get_cancelled_exc_class,
    threadlocals,
)
from ._core._eventloop import run as run_eventloop
from ._core._exceptions import ClosedResourceError, NoEventLoopError
from ._core._synchronization import Event
from ._core._tasks import CancelScope, create_task_group
from .abc._tasks import TaskStatus
from .lowlevel import EventLoopToken, current_token

if sys.version_info >= (3, 11):
    from typing import Self, TypeVarTuple, Unpack
else:
    from typing_extensions import Self, TypeVarTuple, Unpack

T_Retval = TypeVar("T_Retval")
T_co = TypeVar("T_co", covariant=True)
PosArgsT = TypeVarTuple("PosArgsT")


def _token_or_error(token: EventLoopToken | None) -> EventLoopToken:
    if token is not None:
        return token

    try:
        return threadlocals.current_token
    except AttributeError:
        raise NoEventLoopError(
            "Not running inside an AnyIO worker thread, and no event loop token was "
            "provided"
        ) from None


def run(
    func: Callable[[Unpack[PosArgsT]], Coroutine[Any, Any, T_co]],
    *args: Unpack[PosArgsT],
    token: EventLoopToken | None = None,
) -> T_co:
    """
    Call a coroutine function from a worker thread.

    :param func: a coroutine function
    :param args: positional arguments for the callable
    :param token: an event loop token to use to get back to the event loop thread
        (required if calling this function from outside an AnyIO worker thread)
    :return: the return value of the coroutine function
    :raises MissingTokenError: if no token was provided and called from outside an
        AnyIO worker thread
    :raises RunFinishedError: if the event loop tied to ``token`` is no longer running

    .. versionchanged:: 4.11.0
        Added the ``token`` parameter.

    """
    explicit_token = token is not None
    token = _token_or_error(token)
    return token.backend_class.run_async_from_thread(
        func, args, token=token.native_token if explicit_token else None
    )


def run_sync(
    func: Callable[[Unpack[PosArgsT]], T_Retval],
    *args: Unpack[PosArgsT],
    token: EventLoopToken | None = None,
) -> T_Retval:
    """
    Call a function in the event loop thread from a worker thread.

    :param func: a callable
    :param args: positional arguments for the callable
    :param token: an event loop token to use to get back to the event loop thread
        (required if calling this function from outside an AnyIO worker thread)
    :return: the return value of the callable
    :raises MissingTokenError: if no token was provided and called from outside an
        AnyIO worker thread
    :raises RunFinishedError: if the event loop tied to ``token`` is no longer running

    .. versionchanged:: 4.11.0
        Added the ``token`` parameter.

    """
    explicit_token = token is not None
    token = _token_or_error(token)
    return token.backend_class.run_sync_from_thread(
        func, args, token=token.native_token if explicit_token else None
    )


class _BlockingAsyncContextManager(AbstractContextManager, Generic[T_co]):
    _enter_future: Future[T_co]
    _exit_future: Future[bool | None]
    _exit_event: Event
    _exit_exc_info: tuple[
        type[BaseException] | None, BaseException | None, TracebackType | None
    ] = (None, None, None)

    def __init__(
        self, async_cm: AbstractAsyncContextManager[T_co], portal: BlockingPortal
    ):
        self._async_cm = async_cm
        self._portal = portal

    async def run_async_cm(self) -> bool | None:
        try:
            self._exit_event = Event()
            value = await self._async_cm.__aenter__()
        except BaseException as exc:
            self._enter_future.set_exception(exc)
            raise
        else:
            self._enter_future.set_result(value)

        try:
            # Wait for the sync context manager to exit.
            # This next statement can raise `get_cancelled_exc_class()` if
            # something went wrong in a task group in this async context
            # manager.
            await self._exit_event.wait()
        finally:
            # In case of cancellation, it could be that we end up here before
            # `_BlockingAsyncContextManager.__exit__` is called, and an
            # `_exit_exc_info` has been set.
            result = await self._async_cm.__aexit__(*self._exit_exc_info)

        return result

    def __enter__(self) -> T_co:
        self._enter_future = Future()
        self._exit_future = self._portal.start_task_soon(self.run_async_cm)
        return self._enter_future.result()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> bool | None:
        self._exit_exc_info = exc_type, exc_value, traceback
        self._portal.call(self._exit_event.set)
        return self._exit_future.result()


class _BlockingPortalTaskStatus(TaskStatus):
    def __init__(self, future: Future):
        self._future = future

    def started(self, value: object = None) -> None:
        self._future.set_result(value)


class BlockingPortal:
    """
    An object that lets external threads run code in an asynchronous event loop.

    :raises NoEventLoopError: if no supported asynchronous event loop is running in the
        current thread
    """

    def __init__(self) -> None:
        self._token = current_token()
        self._event_loop_thread_id: int | None = get_ident()
        self._stop_event = Event()
        self._task_group = create_task_group()

    async def __aenter__(self) -> Self:
        await self._task_group.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        await self.stop()
        return await self._task_group.__aexit__(exc_type, exc_val, exc_tb)

    def _check_running(self) -> None:
        if self._event_loop_thread_id is None:
            raise RuntimeError("This portal is not running")
        if self._event_loop_thread_id == get_ident():
            raise RuntimeError(
                "This method cannot be called from the event loop thread"
            )

    async def sleep_until_stopped(self) -> None:
        """Sleep until :meth:`stop` is called."""
        await self._stop_event.wait()

    async def stop(self, cancel_remaining: bool = False) -> None:
        """
        Signal the portal to shut down.

        This marks the portal as no longer accepting new calls and exits from
        :meth:`sleep_until_stopped`.

        :param cancel_remaining: ``True`` to cancel all the remaining tasks, ``False``
            to let them finish before returning

        """
        self._event_loop_thread_id = None
        self._stop_event.set()
        if cancel_remaining:
            self._task_group.cancel_scope.cancel("the blocking portal is shutting down")

    async def _call_func(
        self,
        func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval] | T_Retval],
        args: tuple[Unpack[PosArgsT]],
        kwargs: dict[str, Any],
        future: Future[T_Retval],
    ) -> None:
        event_loop_thread_id = self._event_loop_thread_id

        def callback(f: Future[T_Retval]) -> None:
            if f.cancelled():
                if event_loop_thread_id == get_ident():
                    scope.cancel("the future was cancelled")
                elif event_loop_thread_id is not None:
                    run_sync(
                        scope.cancel, "the future was cancelled", token=self._token
                    )

        try:
            retval_or_awaitable = func(*args, **kwargs)
            if isawaitable(retval_or_awaitable):
                with CancelScope() as scope:
                    future.add_done_callback(callback)
                    retval = await retval_or_awaitable
            else:
                retval = retval_or_awaitable
        except get_cancelled_exc_class():
            future.cancel()
            future.set_running_or_notify_cancel()
        except BaseException as exc:
            if not future.cancelled():
                future.set_exception(exc)

            # Let base exceptions fall through
            if not isinstance(exc, Exception):
                raise
        else:
            if not future.cancelled():
                future.set_result(retval)
        finally:
            scope = None  # type: ignore[assignment]

    def _spawn_task_from_thread(
        self,
        func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval] | T_Retval],
        args: tuple[Unpack[PosArgsT]],
        kwargs: dict[str, Any],
        name: object,
        future: Future[T_Retval],
    ) -> None:
        """
        Spawn a new task using the given callable.

        :param func: a callable
        :param args: positional arguments to be passed to the callable
        :param kwargs: keyword arguments to be passed to the callable
        :param name: name of the task (will be coerced to a string if not ``None``)
        :param future: a future that will resolve to the return value of the callable,
            or the exception raised during its execution

        """
        run_sync(
            partial(self._task_group.start_soon, name=name),
            self._call_func,
            func,
            args,
            kwargs,
            future,
            token=self._token,
        )

    @overload
    def call(
        self,
        func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval]],
        *args: Unpack[PosArgsT],
    ) -> T_Retval: ...

    @overload
    def call(
        self, func: Callable[[Unpack[PosArgsT]], T_Retval], *args: Unpack[PosArgsT]
    ) -> T_Retval: ...

    def call(
        self,
        func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval] | T_Retval],
        *args: Unpack[PosArgsT],
    ) -> T_Retval:
        """
        Call the given function in the event loop thread.

        If the callable returns a coroutine object, it is awaited on.

        :param func: any callable
        :raises RuntimeError: if the portal is not running or if this method is called
            from within the event loop thread

        """
        return cast(T_Retval, self.start_task_soon(func, *args).result())

    @overload
    def start_task_soon(
        self,
        func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval]],
        *args: Unpack[PosArgsT],
        name: object = None,
    ) -> Future[T_Retval]: ...

    @overload
    def start_task_soon(
        self,
        func: Callable[[Unpack[PosArgsT]], T_Retval],
        *args: Unpack[PosArgsT],
        name: object = None,
    ) -> Future[T_Retval]: ...

    def start_task_soon(
        self,
        func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval] | T_Retval],
        *args: Unpack[PosArgsT],
        name: object = None,
    ) -> Future[T_Retval]:
        """
        Start a task in the portal's task group.

        The task will be run inside a cancel scope which can be cancelled by cancelling
        the returned future.

        :param func: the target function
        :param args: positional arguments passed to ``func``
        :param name: name of the task (will be coerced to a string if not ``None``)
        :return: a future that resolves with the return value of the callable if the
            task completes successfully, or with the exception raised in the task
        :raises RuntimeError: if the portal is not running or if this method is called
            from within the event loop thread
        :rtype: concurrent.futures.Future[T_Retval]

        .. versionadded:: 3.0

        """
        self._check_running()
        f: Future[T_Retval] = Future()
        self._spawn_task_from_thread(func, args, {}, name, f)
        return f

    def start_task(
        self,
        func: Callable[..., Awaitable[T_Retval]],
        *args: object,
        name: object = None,
    ) -> tuple[Future[T_Retval], Any]:
        """
        Start a task in the portal's task group and wait until it signals for readiness.

        This method works the same way as :meth:`.abc.TaskGroup.start`.

        :param func: the target function
        :param args: positional arguments passed to ``func``
        :param name: name of the task (will be coerced to a string if not ``None``)
        :return: a tuple of (future, task_status_value) where the ``task_status_value``
            is the value passed to ``task_status.started()`` from within the target
            function
        :rtype: tuple[concurrent.futures.Future[T_Retval], Any]

        .. versionadded:: 3.0

        """

        def task_done(future: Future[T_Retval]) -> None:
            if not task_status_future.done():
                if future.cancelled():
                    task_status_future.cancel()
                elif future.exception():
                    task_status_future.set_exception(future.exception())
                else:
                    exc = RuntimeError(
                        "Task exited without calling task_status.started()"
                    )
                    task_status_future.set_exception(exc)

        self._check_running()
        task_status_future: Future = Future()
        task_status = _BlockingPortalTaskStatus(task_status_future)
        f: Future = Future()
        f.add_done_callback(task_done)
        self._spawn_task_from_thread(func, args, {"task_status": task_status}, name, f)
        return f, task_status_future.result()

    def wrap_async_context_manager(
        self, cm: AbstractAsyncContextManager[T_co]
    ) -> AbstractContextManager[T_co]:
        """
        Wrap an async context manager as a synchronous context manager via this portal.

        Spawns a task that will call both ``__aenter__()`` and ``__aexit__()``, stopping
        in the middle until the synchronous context manager exits.

        :param cm: an asynchronous context manager
        :return: a synchronous context manager

        .. versionadded:: 2.1

        """
        return _BlockingAsyncContextManager(cm, self)


class CompletionUnitFailedError(Exception):
    """
    Raised by :meth:`CompletionUnit.wait` when the completion unit was finalized
    with an exception submitted by a worker thread.

    The originally submitted exception is available as the ``__cause__``
    attribute. A fresh instance of this error is raised in every waiter, so the
    tracebacks of concurrent waiters never get mixed into each other or into the
    submitted exception.

    .. versionadded:: 4.16.0
    """


class CompletionUnitUnreachableError(RuntimeError):
    """
    Raised by the submission methods of :class:`CompletionUnit` when the bound
    blocking portal is no longer running, making the event loop unreachable.

    This is raised instead of silently pretending that the submission succeeded.

    .. versionadded:: 4.16.0
    """


class _Outcome(NamedTuple):
    is_error: bool
    payload: Any


class CompletionUnit(Generic[T_Retval]):
    """
    A one-shot completion unit bound to an event loop and a blocking portal.

    Worker threads finalize the unit by submitting either a result
    (:meth:`submit_result`) or an exception (:meth:`submit_exception`). Every
    submission is routed through the bound :class:`BlockingPortal`, so the final
    decision is made atomically in the event loop thread and the internal event
    is never set directly from a foreign thread. Only the first submission to
    arrive wins; the submitting thread is told whether it won (``True``) or lost
    (``False``).

    Any number of asynchronous tasks may :meth:`wait` for the outcome, and they
    all receive the same first submission (``None`` is a valid result value).
    Cancelling an individual waiter has no effect on the unit or on the other
    waiters.

    The lifecycle of a completion unit is as follows:

    #. Create the unit in the event loop thread, bound to a running
       :class:`BlockingPortal`.
    #. Let worker threads submit a result or an exception while asynchronous
       tasks wait for the outcome.
    #. Close the unit with :meth:`aclose` (or by leaving the ``async with``
       block) when it is no longer needed. Closing wakes up any remaining
       waiters with :exc:`~anyio.ClosedResourceError`.

    Submissions made after the unit has been closed are rejected with
    :exc:`~anyio.ClosedResourceError`. Submissions made after the bound portal
    has been shut down fail with :exc:`CompletionUnitUnreachableError` instead
    of being silently accepted.

    :param portal: the blocking portal bound to the event loop that owns this
        unit

    .. versionadded:: 4.16.0
    """

    def __init__(self, portal: BlockingPortal) -> None:
        self._portal = portal
        self._event = Event()
        self._outcome: _Outcome | None = None
        self._closed = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    @property
    def done(self) -> bool:
        """Return ``True`` if the unit has been finalized or closed."""
        return self._outcome is not None or self._closed

    def _finalize(self, outcome: _Outcome) -> bool:
        # This method only ever runs in the event loop thread, via the portal,
        # which makes the first-submission-wins decision atomic.
        if self._closed:
            raise ClosedResourceError("this completion unit has been closed")

        if self._outcome is not None:
            return False

        self._outcome = outcome
        self._event.set()
        return True

    def _submit(self, outcome: _Outcome) -> bool:
        if self._portal._event_loop_thread_id == get_ident():
            raise RuntimeError(
                "submissions to a completion unit must be made from an external "
                "thread, not from the event loop thread"
            )

        try:
            return self._portal.call(self._finalize, outcome)
        except RuntimeError as exc:
            raise CompletionUnitUnreachableError(
                "the event loop bound to this completion unit is unreachable "
                "(the blocking portal has been shut down)"
            ) from exc

    def submit_result(self, value: T_Retval) -> bool:
        """
        Submit a successful result from a worker thread.

        The submission is finalized atomically in the event loop thread via the
        bound blocking portal. ``None`` is a valid result value.

        :param value: the result value to deliver to the waiters
        :return: ``True`` if this submission finalized the unit, ``False`` if
            another submission had already finalized it
        :raises ClosedResourceError: if the unit has been closed
        :raises CompletionUnitUnreachableError: if the bound blocking portal is
            no longer running

        """
        return self._submit(_Outcome(False, value))

    def submit_exception(self, exc: BaseException) -> bool:
        """
        Submit an exception from a worker thread.

        The waiters will raise :exc:`CompletionUnitFailedError` with the
        submitted exception as its ``__cause__``.

        :param exc: the exception to deliver to the waiters
        :return: ``True`` if this submission finalized the unit, ``False`` if
            another submission had already finalized it
        :raises ClosedResourceError: if the unit has been closed
        :raises CompletionUnitUnreachableError: if the bound blocking portal is
            no longer running

        """
        return self._submit(_Outcome(True, exc))

    async def wait(self) -> T_Retval:
        """
        Wait for the first submission to finalize the unit.

        All waiters receive the same outcome. Cancelling one waiter does not
        cancel the unit or affect any other waiter.

        :return: the submitted result value (which may be ``None``)
        :raises CompletionUnitFailedError: if the unit was finalized with an
            exception (available as ``__cause__``)
        :raises ClosedResourceError: if the unit was closed before any
            submission arrived

        """
        await self._event.wait()
        outcome = self._outcome
        if outcome is None:
            raise ClosedResourceError(
                "the completion unit was closed before any submission arrived"
            )

        if outcome.is_error:
            # Raise a fresh wrapper in each waiter so that their tracebacks
            # never get tangled up with each other or with the shared,
            # submitted exception.
            raise CompletionUnitFailedError(
                f"the completion unit was finalized with an exception: "
                f"{outcome.payload!r}"
            ) from outcome.payload

        return cast(T_Retval, outcome.payload)

    async def aclose(self) -> None:
        """
        Close the unit, waking up any remaining waiters.

        Waiters still waiting for an outcome are woken with
        :exc:`~anyio.ClosedResourceError`, and any further submissions are
        rejected. This method must be called in the event loop thread.
        """
        if not self._closed:
            self._closed = True
            self._event.set()


@dataclass
class BlockingPortalProvider:
    """
    A manager for a blocking portal. Used as a context manager. The first thread to
    enter this context manager causes a blocking portal to be started with the specific
    parameters, and the last thread to exit causes the portal to be shut down. Thus,
    there will be exactly one blocking portal running in this context as long as at
    least one thread has entered this context manager.

    The parameters are the same as for :func:`~anyio.run`.

    :param backend: name of the backend
    :param backend_options: backend options

    .. versionadded:: 4.4
    """

    backend: str = "asyncio"
    backend_options: dict[str, Any] | None = None
    _lock: Lock = field(init=False, default_factory=Lock)
    _leases: int = field(init=False, default=0)
    _portal: BlockingPortal = field(init=False)
    _portal_cm: AbstractContextManager[BlockingPortal] | None = field(
        init=False, default=None
    )

    def __enter__(self) -> BlockingPortal:
        with self._lock:
            if self._portal_cm is None:
                self._portal_cm = start_blocking_portal(
                    self.backend, self.backend_options
                )
                self._portal = self._portal_cm.__enter__()

            self._leases += 1
            return self._portal

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        portal_cm: AbstractContextManager[BlockingPortal] | None = None
        with self._lock:
            assert self._portal_cm
            assert self._leases > 0
            self._leases -= 1
            if not self._leases:
                portal_cm = self._portal_cm
                self._portal_cm = None
                del self._portal

        if portal_cm:
            portal_cm.__exit__(None, None, None)


@contextmanager
def start_blocking_portal(
    backend: str = "asyncio",
    backend_options: dict[str, Any] | None = None,
    *,
    name: str | None = None,
) -> Generator[BlockingPortal, Any, None]:
    """
    Start a new event loop in a new thread and run a blocking portal in its main task.

    The parameters are the same as for :func:`~anyio.run`.

    :param backend: name of the backend
    :param backend_options: backend options
    :param name: name of the thread
    :return: a context manager that yields a blocking portal

    .. versionchanged:: 3.0
        Usage as a context manager is now required.

    """

    async def run_portal() -> None:
        async with BlockingPortal() as portal_:
            if name is None:
                current_thread().name = f"{backend}-portal-{id(portal_):x}"

            future.set_result(portal_)
            await portal_.sleep_until_stopped()

    def run_blocking_portal() -> None:
        if future.set_running_or_notify_cancel():
            try:
                run_eventloop(
                    run_portal, backend=backend, backend_options=backend_options
                )
            except BaseException as exc:
                if not future.done():
                    future.set_exception(exc)

    future: Future[BlockingPortal] = Future()
    kwargs: dict[str, Any] = {}
    if sys.version_info >= (3, 14):
        kwargs["context"] = Context()

    thread = Thread(target=run_blocking_portal, daemon=True, name=name, **kwargs)
    thread.start()
    try:
        cancel_remaining_tasks = False
        portal = future.result()
        try:
            yield portal
        except BaseException:
            cancel_remaining_tasks = True
            raise
        finally:
            try:
                portal.call(portal.stop, cancel_remaining_tasks)
            except RuntimeError:
                pass
    finally:
        thread.join()


def check_cancelled() -> None:
    """
    Check if the cancel scope of the host task's running the current worker thread has
    been cancelled.

    If the host task's current cancel scope has indeed been cancelled, the
    backend-specific cancellation exception will be raised.

    :raises RuntimeError: if the current thread was not spawned by
        :func:`.to_thread.run_sync`

    """
    try:
        token: EventLoopToken = threadlocals.current_token
    except AttributeError:
        raise NoEventLoopError(
            "This function can only be called inside an AnyIO worker thread"
        ) from None

    token.backend_class.check_cancelled()
