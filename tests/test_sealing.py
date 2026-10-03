from __future__ import annotations

import pytest

from anyio import (
    CapacityLimiter,
    Event,
    WouldBlock,
    create_task_group,
    fail_after,
)
from anyio._core._sealing import (
    LimiterSealedError,
    SealableCapacityLimiter,
    SealableCapacityLimiterStatistics,
)
from anyio._core._testing import wait_all_tasks_blocked

pytestmark = pytest.mark.anyio


async def test_seal_rejects_queued_waiters_then_drains() -> None:
    """
    Capacity 1: hold the token, queue two waiters, seal. Both waiters must be
    rejected, and wait_drained() must complete once the holder releases.
    """
    limiter = SealableCapacityLimiter(total_tokens=1)
    await limiter.acquire()  # hold the only token in this task

    rejected: list[str] = []
    w1_ready, w2_ready = Event(), Event()

    async def waiter(name: str, ready: Event) -> None:
        ready.set()
        try:
            await limiter.acquire()
        except LimiterSealedError:
            rejected.append(name)

    drained = Event()

    async def drain_watcher() -> None:
        await limiter.wait_drained()
        drained.set()

    async with create_task_group() as tg:
        tg.start_soon(waiter, "w1", w1_ready)
        tg.start_soon(waiter, "w2", w2_ready)
        # Event barrier: both waiters have started and are queued
        await w1_ready.wait()
        await w2_ready.wait()
        await wait_all_tasks_blocked()
        assert limiter.statistics().acquire_waiters == 2

        limiter.seal()
        assert limiter.sealed
        limiter.seal()  # idempotent

        tg.start_soon(drain_watcher)
        await wait_all_tasks_blocked()
        # The token is still borrowed, so the drain watcher must not finish
        assert not drained.is_set()
        assert limiter.statistics().drain_waiters == 1

        # New acquires are rejected immediately after sealing
        with pytest.raises(LimiterSealedError):
            await limiter.acquire()
        with pytest.raises(LimiterSealedError):
            limiter.acquire_nowait()

        # Releasing the last borrow lets wait_drained() complete
        limiter.release()
        with fail_after(5):
            await drained.wait()

    assert sorted(rejected) == ["w1", "w2"]
    statistics = limiter.statistics()
    assert statistics.borrowed_tokens == 0
    assert statistics.acquire_waiters == 0
    assert statistics.drain_waiters == 0


async def test_cancel_drain_waiter_does_not_release_borrowers() -> None:
    """
    Cancelling a coroutine waiting in wait_drained() must only cancel the
    waiter; the existing borrower's token count must stay at 1.
    """
    limiter = SealableCapacityLimiter(total_tokens=1)
    await limiter.acquire()
    limiter.seal()

    entered = Event()

    async def drain_waiter() -> None:
        entered.set()
        await limiter.wait_drained()

    async with create_task_group() as tg:
        tg.start_soon(drain_waiter)
        await entered.wait()
        await wait_all_tasks_blocked()
        assert limiter.statistics().drain_waiters == 1
        tg.cancel_scope.cancel()

    # The cancellation only affected the waiter, not the borrower
    assert limiter.borrowed_tokens == 1
    assert limiter.limiter.borrowed_tokens == 1
    assert limiter.statistics().borrowed_tokens == 1
    assert limiter.statistics().drain_waiters == 0

    # Once the holder releases, draining completes immediately
    limiter.release()
    with fail_after(5):
        await limiter.wait_drained()
    assert limiter.borrowed_tokens == 0


async def test_seal_acquire_race_has_single_outcome() -> None:
    """
    An acquire racing with seal() must end in exactly one state: granted
    (token borrowed) or rejected (LimiterSealedError), never both or neither.
    """
    limiter = SealableCapacityLimiter(total_tokens=1)
    await limiter.acquire()  # occupy the only token

    barrier = Event()
    outcomes: list[str] = []

    async def contender() -> None:
        await barrier.wait()
        try:
            await limiter.acquire()
        except LimiterSealedError:
            outcomes.append("rejected")
        else:
            outcomes.append("granted")
            limiter.release()

    async with create_task_group() as tg:
        tg.start_soon(contender)
        await wait_all_tasks_blocked()
        # Wake the contender and seal before it can run again: with no
        # checkpoint in between, seal() deterministically wins the race.
        barrier.set()
        limiter.seal()

    assert outcomes == ["rejected"]
    # The pre-seal borrow is untouched and can still be released normally
    assert limiter.borrowed_tokens == 1
    limiter.release()
    assert limiter.borrowed_tokens == 0

    # Granted path: without a seal, the same race acquires the token
    limiter2 = SealableCapacityLimiter(total_tokens=1)
    barrier2 = Event()
    outcomes2: list[str] = []

    async def contender2() -> None:
        await barrier2.wait()
        try:
            await limiter2.acquire()
        except LimiterSealedError:
            outcomes2.append("rejected")
        else:
            outcomes2.append("granted")
            limiter2.release()

    async with create_task_group() as tg:
        tg.start_soon(contender2)
        await wait_all_tasks_blocked()
        barrier2.set()

    assert outcomes2 == ["granted"]
    assert limiter2.borrowed_tokens == 0


async def test_seal_does_not_revoke_existing_borrowers() -> None:
    limiter = SealableCapacityLimiter(total_tokens=2)
    await limiter.acquire()
    limiter.acquire_on_behalf_of_nowait(object())
    limiter.seal()
    # Both pre-seal borrows survive the seal
    assert limiter.borrowed_tokens == 2
    limiter.release()
    assert limiter.borrowed_tokens == 1


async def test_acquire_on_behalf_of_rejected_after_seal() -> None:
    limiter = SealableCapacityLimiter(total_tokens=1)
    borrower = object()
    limiter.acquire_on_behalf_of_nowait(borrower)

    rejected = Event()

    async def waiter() -> None:
        try:
            await limiter.acquire_on_behalf_of(object())
        except LimiterSealedError:
            rejected.set()

    async with create_task_group() as tg:
        tg.start_soon(waiter)
        await wait_all_tasks_blocked()
        limiter.seal()
        with fail_after(5):
            await rejected.wait()

    with pytest.raises(LimiterSealedError):
        limiter.acquire_on_behalf_of_nowait(object())

    limiter.release_on_behalf_of(borrower)
    assert limiter.borrowed_tokens == 0


async def test_wait_drained_returns_immediately_when_idle() -> None:
    limiter = SealableCapacityLimiter(total_tokens=1)
    with fail_after(5):
        await limiter.wait_drained()


async def test_context_manager() -> None:
    limiter = SealableCapacityLimiter(total_tokens=1)
    async with limiter:
        assert limiter.borrowed_tokens == 1

    assert limiter.borrowed_tokens == 0

    limiter.seal()
    with pytest.raises(LimiterSealedError):
        async with limiter:
            pass


async def test_wraps_existing_limiter_without_changing_it() -> None:
    inner = CapacityLimiter(1)
    limiter = SealableCapacityLimiter(inner)
    assert limiter.limiter is inner
    assert limiter.total_tokens == 1

    await limiter.acquire()
    # The wrapped limiter observes the borrow through its own interface
    assert inner.borrowed_tokens == 1
    with pytest.raises(WouldBlock):
        inner.acquire_on_behalf_of_nowait(object())

    limiter.release()
    # The underlying limiter still works directly, unchanged
    inner.acquire_nowait()
    inner.release()

    with pytest.raises(TypeError):
        SealableCapacityLimiter(inner, total_tokens=1)
    with pytest.raises(TypeError):
        SealableCapacityLimiter()


async def test_statistics_and_printable_state() -> None:
    limiter = SealableCapacityLimiter(total_tokens=1)
    statistics = limiter.statistics()
    assert isinstance(statistics, SealableCapacityLimiterStatistics)
    assert statistics.sealed is False
    assert statistics.borrowed_tokens == 0
    assert statistics.total_tokens == 1
    assert statistics.acquire_waiters == 0
    assert statistics.drain_waiters == 0
    assert "sealed=False" in str(limiter)

    await limiter.acquire()
    limiter.seal()
    statistics = limiter.statistics()
    assert statistics.sealed is True
    assert statistics.borrowed_tokens == 1

    text = str(limiter)
    assert "SealableCapacityLimiter" in text
    assert "sealed=True" in text
    assert "borrowed=1/1" in text
    limiter.release()
