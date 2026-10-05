"""
Controllable race tests for :class:`anyio.KeyedTaskGroup`.

Run with::

    pytest tests/test_keyed_tasks.py -v

The tests use :class:`anyio.Event` to deterministically control the order in which
same-key tasks finish, so that stale-generation outcomes can be observed and
asserted against without relying on timing.
"""

from __future__ import annotations

import pytest

from anyio import (
    CancelScope,
    Event,
    KeyedTaskGroup,
    sleep,
    sleep_forever,
    wait_all_tasks_blocked,
)


async def test_same_key_replaces_old_task_late_finish() -> None:
    """
    A same-key old task that shields against cancellation and finishes *late* must
    not publish its result; only the new generation's result is published.
    """
    started = Event()
    finish = Event()
    results: list[tuple[str, str]] = []

    async def old_task() -> str:
        started.set()
        with CancelScope(shield=True):
            await finish.wait()
        return "old"

    async def new_task() -> str:
        return "new"

    async with KeyedTaskGroup(
        on_result=lambda key, value: results.append((key, value))
    ) as ktg:
        ktg.start("a", old_task)
        await started.wait()
        await wait_all_tasks_blocked()
        ktg.start("a", new_task)
        # Let the old task finish *after* the new generation has started.
        finish.set()

    assert results == [("a", "new")]


async def test_different_keys_run_in_parallel() -> None:
    """Tasks for different keys both get their results published."""
    results: list[tuple[str, str]] = []

    async def task(key: str) -> str:
        return key

    async with KeyedTaskGroup(
        on_result=lambda key, value: results.append((key, value))
    ) as ktg:
        ktg.start("a", task, "a")
        ktg.start("b", task, "b")

    assert sorted(results) == [("a", "a"), ("b", "b")]


async def test_close_waits_for_inflight_cleanup() -> None:
    """On close, the component waits for in-flight tasks (and their cleanup) to finish."""
    cleanup_done = False

    async def task() -> str:
        nonlocal cleanup_done
        try:
            await sleep(0.01)
            return "done"
        finally:
            cleanup_done = True

    results: list[tuple[str, str]] = []
    async with KeyedTaskGroup(
        on_result=lambda key, value: results.append((key, value))
    ) as ktg:
        ktg.start("a", task)

    assert results == [("a", "done")]
    assert cleanup_done


async def test_old_exception_does_not_override_new_result() -> None:
    """
    A same-key old task that raises an exception (even after shielding cancellation)
    must not override the new generation's result.
    """
    started = Event()
    finish = Event()
    results: list[tuple[str, str]] = []
    exceptions: list[tuple[str, BaseException]] = []

    async def old_task() -> str:
        started.set()
        with CancelScope(shield=True):
            await finish.wait()
        raise ValueError("old")

    async def new_task() -> str:
        return "new"

    async with KeyedTaskGroup(
        on_result=lambda key, value: results.append((key, value)),
        on_exception=lambda key, exc: exceptions.append((key, exc)),
    ) as ktg:
        ktg.start("a", old_task)
        await started.wait()
        await wait_all_tasks_blocked()
        ktg.start("a", new_task)
        finish.set()

    assert results == [("a", "new")]
    assert exceptions == []


async def test_old_cancel_does_not_publish() -> None:
    """
    A same-key old task that is cancelled must not publish a cancellation; only the
    new generation's result is published.
    """
    started = Event()
    finish = Event()
    results: list[tuple[str, str]] = []
    cancels: list[str] = []

    async def old_task() -> str:
        started.set()
        await finish.wait()
        return "old"

    async def new_task() -> str:
        return "new"

    async with KeyedTaskGroup(
        on_result=lambda key, value: results.append((key, value)),
        on_cancel=lambda key: cancels.append(key),
    ) as ktg:
        ktg.start("a", old_task)
        await started.wait()
        await wait_all_tasks_blocked()
        ktg.start("a", new_task)
        finish.set()

    assert results == [("a", "new")]
    assert cancels == []


async def test_current_generation_cancel_is_published() -> None:
    """The current generation's cancellation is published via ``on_cancel``."""
    started = Event()
    cancels: list[str] = []

    async def task() -> None:
        started.set()
        await sleep_forever()

    async with KeyedTaskGroup(on_cancel=lambda key: cancels.append(key)) as ktg:
        handle = ktg.start("a", task)
        await started.wait()
        await wait_all_tasks_blocked()
        handle.cancel()

    assert cancels == ["a"]


async def test_start_after_close_raises() -> None:
    """After close, starting a task raises :class:`RuntimeError`."""
    ktg = KeyedTaskGroup()
    async with ktg:
        pass

    with pytest.raises(RuntimeError, match="closed"):
        ktg.start("a", sleep, 0)


async def test_lifecycle() -> None:
    """
    Lifecycle of :class:`KeyedTaskGroup`:

    * before entering, ``start`` raises :class:`RuntimeError`
    * while active, ``start`` accepts tasks
    * after exiting, ``start`` raises :class:`RuntimeError` and ``closed`` is ``True``
    """
    ktg = KeyedTaskGroup()
    assert not ktg.closed

    with pytest.raises(RuntimeError, match="not active"):
        ktg.start("a", sleep, 0)

    async with ktg:
        assert not ktg.closed
        ktg.start("a", sleep, 0)

    assert ktg.closed

    with pytest.raises(RuntimeError, match="closed"):
        ktg.start("a", sleep, 0)


async def test_separate_notifications() -> None:
    """Results, exceptions, and cancellations are notified through separate callbacks."""
    results: list[tuple[str, object]] = []
    exceptions: list[tuple[str, BaseException]] = []
    cancels: list[str] = []

    async def success() -> str:
        return "ok"

    async def failure() -> str:
        raise ValueError("boom")

    async def forever() -> None:
        await sleep_forever()

    async with KeyedTaskGroup(
        on_result=lambda key, value: results.append((key, value)),
        on_exception=lambda key, exc: exceptions.append((key, exc)),
        on_cancel=lambda key: cancels.append(key),
    ) as ktg:
        ktg.start("success", success)
        ktg.start("failure", failure)
        handle = ktg.start("forever", forever)
        await wait_all_tasks_blocked()
        handle.cancel()

    assert results == [("success", "ok")]
    assert len(exceptions) == 1
    assert exceptions[0][0] == "failure"
    assert isinstance(exceptions[0][1], ValueError)
    assert cancels == ["forever"]
