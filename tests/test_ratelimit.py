"""Tier 1: the rate limiter meters requests deterministically, bounds concurrency, and is shared across sync and async callers."""

import asyncio
import threading
from collections.abc import Coroutine

import pytest

from portfolio_optimizer.ratelimit import RateLimit, RateLimiter, fan_out


class FakeTime:
    """A clock the limiter reads and a sleep that advances it, so token timing is exact."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        """Yield once before advancing the clock, so concurrent callers reserve before anyone wakes — as with a real clock."""
        due = self.now + delay
        self.sleeps.append(delay)
        await asyncio.sleep(0)
        self.now = max(self.now, due)


def run[T](coroutine: Coroutine[object, object, T]) -> T:
    return asyncio.run(coroutine)


# --- RateLimit ---


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({}, "requests_per_second, max_in_flight, or both"),
        ({"requests_per_second": 0.0}, "requests_per_second must be positive"),
        ({"requests_per_second": 1.0, "burst": 0}, "burst must be at least 1"),
        ({"max_in_flight": 0}, "max_in_flight must be at least 1"),
    ],
)
def test_rate_limit_rejects_meaningless_bounds(kwargs: dict[str, object], fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        RateLimit(**kwargs)  # ty: ignore[invalid-argument-type]  # the bad values are the case under test


# --- token bucket ---


def test_burst_is_free_and_then_callers_wait_exactly_one_token_apart() -> None:
    fake = FakeTime()

    async def scenario() -> RateLimiter:
        limiter = RateLimiter(RateLimit(requests_per_second=10.0, burst=2), name="api", clock=fake.clock, sleep=fake.sleep)
        for _ in range(4):
            async with limiter:
                pass
        return limiter

    limiter = run(scenario())
    assert fake.sleeps == [pytest.approx(0.1), pytest.approx(0.1)]
    assert limiter.acquired == 4
    assert limiter.waited_s == pytest.approx(0.2)


def test_bucket_refills_while_idle_up_to_the_burst() -> None:
    fake = FakeTime()

    async def scenario() -> None:
        limiter = RateLimiter(RateLimit(requests_per_second=2.0, burst=3), clock=fake.clock, sleep=fake.sleep)
        for _ in range(3):
            await limiter.acquire()
        fake.now += 100.0  # a long pause refills to the burst, not beyond it
        for _ in range(3):
            await limiter.acquire()
        await limiter.acquire()

    run(scenario())
    assert fake.sleeps == [pytest.approx(0.5)]


def test_reservations_are_handed_out_in_call_order_without_a_thundering_herd() -> None:
    fake = FakeTime()
    order: list[int] = []

    async def scenario() -> None:
        limiter = RateLimiter(RateLimit(requests_per_second=1.0, burst=1), clock=fake.clock, sleep=fake.sleep)

        async def one(index: int) -> None:
            async with limiter:
                order.append(index)

        async with asyncio.TaskGroup() as group:
            for index in range(4):
                group.create_task(one(index))

    run(scenario())
    assert fake.sleeps == [pytest.approx(1.0), pytest.approx(2.0), pytest.approx(3.0)]
    assert sorted(order) == [0, 1, 2, 3]


# --- in-flight bound ---


def test_max_in_flight_bounds_concurrency() -> None:
    peak = 0

    async def scenario() -> None:
        nonlocal peak
        limiter = RateLimiter(RateLimit(max_in_flight=2))
        active = 0

        async def one() -> None:
            nonlocal active, peak
            async with limiter:
                active += 1
                peak = max(peak, active)
                for _ in range(3):
                    await asyncio.sleep(0)
                active -= 1

        async with asyncio.TaskGroup() as group:
            for _ in range(6):
                group.create_task(one())

    run(scenario())
    assert peak == 2


def test_a_caller_cancelled_while_waiting_for_its_token_gives_its_slot_back() -> None:
    entered_sleep = 0

    async def blocking_sleep(delay: float) -> None:
        nonlocal entered_sleep
        del delay
        entered_sleep += 1
        await asyncio.Event().wait()  # never set: sleeps until cancelled

    async def scenario() -> None:
        limiter = RateLimiter(RateLimit(requests_per_second=1.0, burst=1, max_in_flight=1), sleep=blocking_sleep)
        async with limiter:
            pass  # spends the burst
        first = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(limiter.acquire())
        for _ in range(3):
            await asyncio.sleep(0)
        assert entered_sleep == 2, "the second caller never got the slot, so the first leaked it"
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second

    run(scenario())


# --- the sync bridge ---


def test_sync_loaders_in_threads_share_the_pool_with_async_callers() -> None:
    fake = FakeTime()
    entered = threading.Event()

    async def scenario() -> RateLimiter:
        limiter = RateLimiter(RateLimit(requests_per_second=10.0, burst=1, max_in_flight=1), clock=fake.clock, sleep=fake.sleep)

        def worker() -> None:
            with limiter.sync:
                entered.set()

        await limiter.acquire()  # hold the only slot from the loop side
        thread = asyncio.ensure_future(asyncio.to_thread(worker))
        await asyncio.sleep(0.05)
        assert not entered.is_set(), "the thread entered while the slot was held"
        limiter.release()
        await thread
        assert entered.is_set()
        return limiter

    limiter = run(scenario())
    assert limiter.acquired == 2
    assert fake.sleeps == [pytest.approx(0.1)]  # the thread's token came from the same bucket


def test_sync_bridge_refuses_to_block_the_event_loop_thread() -> None:
    async def scenario() -> None:
        limiter = RateLimiter(RateLimit(max_in_flight=1))
        with pytest.raises(RuntimeError, match="would block the event loop"), limiter.sync:
            pass

    run(scenario())


def test_sync_bridge_needs_the_loop_the_engine_binds() -> None:
    limiter = RateLimiter(RateLimit(max_in_flight=1), name="orphan")
    with pytest.raises(RuntimeError, match="created outside an event loop"), limiter.sync:
        pass


def test_unlimited_never_waits_anywhere() -> None:
    limiter = RateLimiter.unlimited()
    assert not limiter.is_limited
    with limiter.sync:
        pass

    async def scenario() -> None:
        async with limiter:
            pass

    run(scenario())
    assert limiter.waited_s == 0.0


# --- fan_out ---


def test_fan_out_returns_results_in_item_order_under_the_limiter() -> None:
    async def scenario() -> tuple[list[int], RateLimiter]:
        limiter = RateLimiter(RateLimit(max_in_flight=3))

        async def double(item: int) -> int:
            for _ in range(item % 3):
                await asyncio.sleep(0)  # finish out of order
            return item * 2

        return await fan_out(range(10), double, limiter=limiter), limiter

    results, limiter = run(scenario())
    assert results == [item * 2 for item in range(10)]
    assert limiter.acquired == 10


def test_fan_out_propagates_a_failure_and_cancels_the_rest() -> None:
    finished: list[int] = []

    async def scenario() -> None:
        limiter = RateLimiter(RateLimit(max_in_flight=1))

        async def call(item: int) -> int:
            if item == 1:
                msg = "upstream 503"
                raise ConnectionError(msg)
            await asyncio.sleep(0)
            finished.append(item)
            return item

        await fan_out(range(4), call, limiter=limiter)

    with pytest.raises(ExceptionGroup) as info:
        run(scenario())
    assert info.group_contains(ConnectionError, match="upstream 503")
    assert 3 not in finished
