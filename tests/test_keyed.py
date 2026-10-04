from __future__ import annotations

from typing import Any

import pytest

from anyio import (
    CancelScope,
    ClosedResourceError,
    Event,
    KeyedTaskRunner,
    create_task_group,
    get_cancelled_exc_class,
    sleep_forever,
    wait_all_tasks_blocked,
)

pytestmark = pytest.mark.anyio


async def test_same_key_late_result_not_published() -> None:
    """
    A superseded task that shields the cancellation and finishes *after* the new
    generation must not publish its result; only the new generation's result is
    delivered.
    """
    results: list[tuple[Any, int, Any]] = []
    old_started = Event()
    old_release = Event()
    new_release = Event()
    got_result = Event()

    async def old_task() -> str:
        old_started.set()
        try:
            await sleep_forever()
        except get_cancelled_exc_class():
            # Shield the cancellation and finish late, after the new generation
            with CancelScope(shield=True):
                await old_release.wait()

            return "old"

        pytest.fail("old task was never cancelled")

    async def new_task() -> str:
        await new_release.wait()
        return "new"

    def on_result(key: Any, generation: int, value: Any) -> None:
        results.append((key, generation, value))
        got_result.set()

    async with create_task_group() as tg:
        async with KeyedTaskRunner(tg, on_result=on_result) as runner:
            gen1 = runner.start("key", old_task)
            await old_started.wait()

            # Supersede the old task; it receives the cancellation but shields
            # it and keeps running until old_release is set
            gen2 = runner.start("key", new_task)
            assert gen2 == gen1 + 1
            assert runner.generation("key") == gen2

            # The new generation finishes first and publishes
            new_release.set()
            await got_result.wait()
            assert results == [("key", gen2, "new")]

            # The old generation finishes late; its result must be dropped
            old_release.set()
            await runner.wait_idle()
            assert results == [("key", gen2, "new")]


async def test_superseded_task_cancellation_notified() -> None:
    """The old task's cancellation is reported through the cancelled callback."""
    cancelled: list[tuple[Any, int]] = []
    started = Event()
    got_cancelled = Event()

    async def old_task() -> None:
        started.set()
        await sleep_forever()

    async def new_task() -> str:
        return "new"

    def on_cancelled(key: Any, generation: int) -> None:
        cancelled.append((key, generation))
        got_cancelled.set()

    async with create_task_group() as tg:
        async with KeyedTaskRunner(tg, on_cancelled=on_cancelled) as runner:
            gen1 = runner.start("key", old_task)
            await started.wait()
            runner.start("key", new_task)
            await got_cancelled.wait()
            await runner.wait_idle()

    assert cancelled == [("key", gen1)]


async def test_different_keys_run_in_parallel() -> None:
    """Tasks with different keys do not supersede each other; both publish."""
    results: list[tuple[Any, Any]] = []
    gates = {"a": Event(), "b": Event()}
    started = {"a": Event(), "b": Event()}
    got_all = Event()

    async def task(name: str) -> str:
        started[name].set()
        await gates[name].wait()
        return name

    def on_result(key: Any, generation: int, value: Any) -> None:
        results.append((key, value))
        if len(results) == 2:
            got_all.set()

    async with create_task_group() as tg:
        async with KeyedTaskRunner(tg, on_result=on_result) as runner:
            runner.start("a", task, "a")
            runner.start("b", task, "b")
            await started["a"].wait()
            await started["b"].wait()
            # Complete them in reverse start order to prove independence
            gates["b"].set()
            gates["a"].set()
            await got_all.wait()

    assert sorted(results) == [("a", "a"), ("b", "b")]


async def test_stale_exception_does_not_override_new_result() -> None:
    """
    A superseded task that shields the cancellation and then raises late must not
    publish its exception over the new generation's result.
    """
    results: list[tuple[Any, int, Any]] = []
    exceptions: list[tuple[Any, int, BaseException]] = []
    old_started = Event()
    old_release = Event()
    new_release = Event()
    got_result = Event()

    async def old_task() -> str:
        old_started.set()
        try:
            await sleep_forever()
        except get_cancelled_exc_class():
            with CancelScope(shield=True):
                await old_release.wait()

            raise RuntimeError("stale boom") from None

        pytest.fail("old task was never cancelled")

    async def new_task() -> str:
        await new_release.wait()
        return "new"

    def on_result(key: Any, generation: int, value: Any) -> None:
        results.append((key, generation, value))
        got_result.set()

    def on_exception(key: Any, generation: int, exc: BaseException) -> None:
        exceptions.append((key, generation, exc))

    async with create_task_group() as tg:
        async with KeyedTaskRunner(
            tg, on_result=on_result, on_exception=on_exception
        ) as runner:
            runner.start("key", old_task)
            await old_started.wait()
            gen2 = runner.start("key", new_task)

            # New generation publishes first
            new_release.set()
            await got_result.wait()

            # The stale generation fails late; the exception must be dropped
            old_release.set()
            await runner.wait_idle()

    assert results == [("key", gen2, "new")]
    assert exceptions == []


async def test_current_generation_exception_notified() -> None:
    """Exceptions from the current generation go to the exception callback."""
    exceptions: list[tuple[Any, int, BaseException]] = []
    got_exception = Event()
    boom = ValueError("nope")

    async def failing() -> None:
        raise boom

    def on_exception(key: Any, generation: int, exc: BaseException) -> None:
        exceptions.append((key, generation, exc))
        got_exception.set()

    async with create_task_group() as tg:
        async with KeyedTaskRunner(tg, on_exception=on_exception) as runner:
            runner.start("key", failing)
            await got_exception.wait()

    assert exceptions == [("key", 1, boom)]


async def test_cancel_single_key() -> None:
    """cancel() cancels only the given key's current task."""
    cancelled: list[tuple[Any, int]] = []
    started = Event()
    got_cancelled = Event()

    async def task() -> None:
        started.set()
        await sleep_forever()

    def on_cancelled(key: Any, generation: int) -> None:
        cancelled.append((key, generation))
        got_cancelled.set()

    async with create_task_group() as tg:
        async with KeyedTaskRunner(tg, on_cancelled=on_cancelled) as runner:
            runner.start("key", task)
            await started.wait()
            assert runner.cancel("key") is True
            await got_cancelled.wait()
            # The task has fully wound down by now, so there is nothing to cancel
            assert runner.cancel("key") is False

    assert cancelled == [("key", 1)]


async def test_aclose_waits_for_inflight_cleanup_and_rejects_new_tasks() -> None:
    """
    aclose() cancels in-flight tasks, blocks until their (shielded) cleanup has
    finished, and afterwards start() raises ClosedResourceError.
    """
    cancelled: list[tuple[Any, int]] = []
    started = Event()
    cleanup_release = Event()
    cleanup_done = Event()
    aclose_returned = Event()

    async def task() -> None:
        started.set()
        try:
            await sleep_forever()
        finally:
            # Shielded cleanup that aclose() must wait for
            with CancelScope(shield=True):
                await cleanup_release.wait()

            cleanup_done.set()

    def on_cancelled(key: Any, generation: int) -> None:
        cancelled.append((key, generation))

    async with create_task_group() as tg:
        runner = KeyedTaskRunner(tg, on_cancelled=on_cancelled)
        runner.start("key", task)
        await started.wait()

        async def closer() -> None:
            await runner.aclose()
            aclose_returned.set()

        async with create_task_group() as closer_tg:
            closer_tg.start_soon(closer)
            await wait_all_tasks_blocked()
            # aclose() must still be waiting for the shielded cleanup
            assert not aclose_returned.is_set()
            assert not cleanup_done.is_set()
            cleanup_release.set()

        assert aclose_returned.is_set()
        assert cleanup_done.is_set()
        assert cancelled == [("key", 1)]
        assert runner.closed

        # A closed runner rejects new tasks
        with pytest.raises(ClosedResourceError):
            runner.start("key", task)

        # aclose() is idempotent
        await runner.aclose()
