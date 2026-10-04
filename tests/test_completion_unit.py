from __future__ import annotations

import threading

import pytest

from anyio import (
    CancelScope,
    ClosedResourceError,
    create_task_group,
    sleep,
    to_thread,
)
from anyio.from_thread import (
    BlockingPortal,
    CompletionUnit,
    CompletionUnitFailedError,
    CompletionUnitUnreachableError,
)

pytestmark = pytest.mark.anyio


class TestCompletionUnit:
    async def test_racing_threads_single_winner(self) -> None:
        """Two threads racing to submit: exactly one of them wins."""
        async with BlockingPortal() as portal:
            unit: CompletionUnit[str] = CompletionUnit(portal)
            barrier = threading.Barrier(2)
            outcomes: dict[str, bool] = {}

            def submit(name: str, value: str) -> None:
                barrier.wait(timeout=10)
                outcomes[name] = unit.submit_result(value)

            thread1 = threading.Thread(target=submit, args=("thread1", "first"))
            thread2 = threading.Thread(target=submit, args=("thread2", "second"))
            thread1.start()
            thread2.start()
            await to_thread.run_sync(thread1.join)
            await to_thread.run_sync(thread2.join)

            assert sorted(outcomes.values()) == [False, True]
            expected = "first" if outcomes["thread1"] else "second"
            assert await unit.wait() == expected
            assert unit.done

    async def test_waiters_share_result_and_cancellation_is_local(self) -> None:
        """
        Three waiters wait on the unit; cancelling one of them leaves the unit
        and the other waiters intact, and the remaining waiters receive the
        very same result object.
        """
        async with BlockingPortal() as portal:
            unit: CompletionUnit[object] = CompletionUnit(portal)
            received: list[object] = []
            cancelled_scope: CancelScope | None = None
            waiter_cancelled = False

            async def waiter() -> None:
                received.append(await unit.wait())

            async def cancellable_waiter() -> None:
                nonlocal cancelled_scope, waiter_cancelled
                with CancelScope() as scope:
                    cancelled_scope = scope
                    await unit.wait()

                waiter_cancelled = True

            result = object()
            async with create_task_group() as tg:
                tg.start_soon(waiter)
                tg.start_soon(waiter)
                tg.start_soon(cancellable_waiter)
                await sleep(0.1)

                # Cancel one of the three waiters; the unit itself must not be
                # cancelled along with it
                assert cancelled_scope is not None
                cancelled_scope.cancel()
                await sleep(0.1)
                assert not unit.done

                assert await to_thread.run_sync(unit.submit_result, result)

            assert waiter_cancelled
            assert len(received) == 2
            assert received[0] is result
            assert received[1] is result

    async def test_none_is_valid_result(self) -> None:
        """None must be deliverable as a legitimate success value."""
        async with BlockingPortal() as portal:
            unit: CompletionUnit[None] = CompletionUnit(portal)
            async with create_task_group() as tg:

                async def submit() -> None:
                    await sleep(0.1)
                    assert await to_thread.run_sync(unit.submit_result, None)

                tg.start_soon(submit)
                assert await unit.wait() is None

    async def test_exception_submission_does_not_tangle_tracebacks(self) -> None:
        """
        Waiters receive a fresh CompletionUnitFailedError each, with the
        submitted exception as the cause, and their tracebacks are not shared.
        """
        async with BlockingPortal() as portal:
            unit: CompletionUnit[None] = CompletionUnit(portal)
            errors: list[CompletionUnitFailedError] = []
            original = ValueError("boom")

            async def waiter() -> None:
                with pytest.raises(CompletionUnitFailedError) as exc_info:
                    await unit.wait()

                errors.append(exc_info.value)

            async with create_task_group() as tg:
                tg.start_soon(waiter)
                tg.start_soon(waiter)
                await sleep(0.1)
                assert await to_thread.run_sync(unit.submit_exception, original)

            assert len(errors) == 2
            assert errors[0] is not errors[1]
            assert errors[0].__cause__ is original
            assert errors[1].__cause__ is original
            assert errors[0].__traceback__ is not errors[1].__traceback__
            # The shared exception must not have been raised in any waiter
            assert original.__traceback__ is None

    async def test_losing_submission_returns_false(self) -> None:
        """A submission made after finalization loses, but is not an error."""
        async with BlockingPortal() as portal:
            unit: CompletionUnit[int] = CompletionUnit(portal)
            assert await to_thread.run_sync(unit.submit_result, 1)
            assert not await to_thread.run_sync(unit.submit_result, 2)
            assert not await to_thread.run_sync(
                unit.submit_exception, ValueError("late")
            )
            assert await unit.wait() == 1

    async def test_close_wakes_waiters(self) -> None:
        """Closing the unit wakes up waiters that never got a submission."""
        async with BlockingPortal() as portal:
            unit: CompletionUnit[None] = CompletionUnit(portal)
            woken = 0

            async def waiter() -> None:
                nonlocal woken
                with pytest.raises(ClosedResourceError):
                    await unit.wait()

                woken += 1

            async with create_task_group() as tg:
                tg.start_soon(waiter)
                tg.start_soon(waiter)
                await sleep(0.1)
                await unit.aclose()

            assert woken == 2
            assert unit.done

    async def test_submit_after_close_rejected(self) -> None:
        """Submissions made after closing are rejected, not accepted."""
        async with BlockingPortal() as portal:
            unit: CompletionUnit[int] = CompletionUnit(portal)
            await unit.aclose()
            with pytest.raises(ClosedResourceError):
                await to_thread.run_sync(unit.submit_result, 1)
            with pytest.raises(ClosedResourceError):
                await to_thread.run_sync(unit.submit_exception, ValueError("x"))

    async def test_submit_after_portal_shutdown_is_unreachable(self) -> None:
        """
        Submitting after the bound portal has shut down must report the loop
        as unreachable instead of pretending to succeed.
        """
        async with BlockingPortal() as portal:
            unit: CompletionUnit[int] = CompletionUnit(portal)

        with pytest.raises(CompletionUnitUnreachableError):
            await to_thread.run_sync(unit.submit_result, 1)
        with pytest.raises(CompletionUnitUnreachableError):
            await to_thread.run_sync(unit.submit_exception, ValueError("x"))

    async def test_submit_from_event_loop_thread_rejected(self) -> None:
        async with BlockingPortal() as portal:
            unit: CompletionUnit[int] = CompletionUnit(portal)
            with pytest.raises(RuntimeError, match="external thread"):
                unit.submit_result(1)

    async def test_async_context_manager(self) -> None:
        """Leaving the async with block closes the unit."""
        async with BlockingPortal() as portal:
            async with CompletionUnit(portal) as unit:
                assert not unit.done

            assert unit.done
            with pytest.raises(ClosedResourceError):
                await to_thread.run_sync(unit.submit_result, 1)
