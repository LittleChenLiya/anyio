from __future__ import annotations

import sys

import pytest

from anyio import (
    BoundedTaskSubmitter,
    CancelScope,
    ClosedResourceError,
    Event,
    create_task_group,
    wait_all_tasks_blocked,
)

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


class Barrier:
    """A simple deterministic barrier for synchronizing tasks without real waiting."""

    def __init__(self, parties: int) -> None:
        self._parties = parties
        self._count = 0
        self._event = Event()

    async def wait(self) -> None:
        self._count += 1
        if self._count == self._parties:
            self._event.set()
        else:
            await self._event.wait()


async def test_capacity_bounded() -> None:
    """Capacity=1: the second task is not created until the first finishes."""
    async with create_task_group() as tg:
        submitter = BoundedTaskSubmitter(tg, capacity=1)
        first_started = Event()
        release_first = Event()
        second_started = Event()

        async def first() -> None:
            first_started.set()
            await release_first.wait()

        async def second() -> None:
            second_started.set()

        handle = await submitter.submit(first)
        await first_started.wait()

        async def submit_second() -> None:
            await submitter.submit(second)

        tg.start_soon(submit_second)
        await wait_all_tasks_blocked()
        assert not second_started.is_set()

        release_first.set()
        await second_started.wait()

        # The first task's result is still retrievable
        assert await handle is None


async def test_exception_propagates_per_taskgroup_rules() -> None:
    """A task exception propagates through the task group as usual."""

    async def failing() -> None:
        raise ValueError("boom")

    with pytest.raises(ExceptionGroup) as exc_info:
        async with create_task_group() as tg:
            submitter = BoundedTaskSubmitter(tg, capacity=1)
            await submitter.submit(failing)

    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], ValueError)
    assert str(exc_info.value.exceptions[0]) == "boom"


async def test_close_rejects_new_submissions() -> None:
    """After close(), new submissions raise ClosedResourceError immediately."""
    async with create_task_group() as tg:
        submitter = BoundedTaskSubmitter(tg, capacity=1)
        submitter.close()

        async def noop() -> None:
            pass

        with pytest.raises(ClosedResourceError):
            await submitter.submit(noop)


async def test_close_wakes_blocked_producers() -> None:
    """A barrier reproduces the close/submit interleaving; the blocked submitter is rejected."""
    async with create_task_group() as tg:
        submitter = BoundedTaskSubmitter(tg, capacity=1)
        first_started = Event()
        release_first = Event()
        about_to_submit = Event()
        second_rejected = Event()
        barrier = Barrier(2)

        async def first() -> None:
            first_started.set()
            await release_first.wait()

        async def second() -> None:
            pass

        async def submit_second() -> None:
            await barrier.wait()
            about_to_submit.set()
            try:
                await submitter.submit(second)
            except ClosedResourceError:
                second_rejected.set()

        async def closer() -> None:
            await barrier.wait()
            await about_to_submit.wait()
            await wait_all_tasks_blocked()
            submitter.close()

        handle = await submitter.submit(first)
        await first_started.wait()

        tg.start_soon(submit_second)
        tg.start_soon(closer)
        await wait_all_tasks_blocked()
        assert not second_rejected.is_set()

        release_first.set()
        await second_rejected.wait()
        assert await handle is None


async def test_cancelled_waiter_does_not_run_factory() -> None:
    """A submission cancelled while waiting for a slot never invokes the factory."""
    async with create_task_group() as tg:
        submitter = BoundedTaskSubmitter(tg, capacity=1)
        first_started = Event()
        release_first = Event()
        factory_called = Event()
        scope_entered = Event()
        scope: CancelScope | None = None

        async def first() -> None:
            first_started.set()
            await release_first.wait()

        async def second() -> None:
            factory_called.set()

        await submitter.submit(first)
        await first_started.wait()

        async def submit_second() -> None:
            nonlocal scope
            with CancelScope() as cancel_scope:
                scope = cancel_scope
                scope_entered.set()
                await submitter.submit(second)

        tg.start_soon(submit_second)
        await scope_entered.wait()
        await wait_all_tasks_blocked()
        assert scope is not None
        scope.cancel()
        await wait_all_tasks_blocked()
        assert not factory_called.is_set()

        release_first.set()
        await wait_all_tasks_blocked()


async def test_cancel_waiter_does_not_leak_slot() -> None:
    """Cancelling a waiting submitter frees the slot for a later submission."""
    async with create_task_group() as tg:
        submitter = BoundedTaskSubmitter(tg, capacity=1)
        first_started = Event()
        release_first = Event()
        second_started = Event()
        scope_entered = Event()
        scope: CancelScope | None = None

        async def first() -> None:
            first_started.set()
            await release_first.wait()

        async def second() -> None:
            second_started.set()

        await submitter.submit(first)
        await first_started.wait()

        async def submit_second() -> None:
            nonlocal scope
            with CancelScope() as cancel_scope:
                scope = cancel_scope
                scope_entered.set()
                await submitter.submit(second)

        tg.start_soon(submit_second)
        await scope_entered.wait()
        await wait_all_tasks_blocked()
        assert scope is not None
        scope.cancel()
        await wait_all_tasks_blocked()
        assert not second_started.is_set()

        release_first.set()
        # The slot must have been freed; a new submission can now run
        third_started = Event()

        async def third() -> None:
            third_started.set()

        await submitter.submit(third)
        await third_started.wait()


async def test_accepted_task_tracked_after_producer_cancel() -> None:
    """Once a task is accepted, cancelling the producer does not lose the task."""
    async with create_task_group() as tg:
        submitter = BoundedTaskSubmitter(tg, capacity=1)
        task_started = Event()
        task_finished = Event()

        async def task() -> None:
            task_started.set()
            task_finished.set()

        async def producer() -> None:
            await submitter.submit(task)

        tg.start_soon(producer)
        await task_started.wait()
        await wait_all_tasks_blocked()
        # The task has been accepted and is tracked by the task group
        assert task_finished.is_set()
