"""Rate limiting for loaders that make many calls to an API or a database in one run.

A :class:`RateLimiter` bounds the request rate — a token bucket refilled continuously at
``requests_per_second`` with a ``burst`` allowance — and the number of simultaneous requests,
``max_in_flight``. The engine creates one limiter per pool named in the run config's
``rate_limits`` and hands it to every loader whose dataset names that pool as
``request.rate_limiter``, so datasets that share a backend share its budget.

An async loader wraps each call in ``async with request.rate_limiter:``. A sync loader — which the
engine runs in a worker thread — wraps each call in ``with request.rate_limiter.sync:``; the bridge
hands the acquisition to the engine's event loop, so both styles draw from the same bucket and the
same in-flight slots. :func:`fan_out` packages the common pattern: one limited call per item,
started concurrently, results returned in item order.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from types import TracebackType
from typing import Self

type Clock = Callable[[], float]
type Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class RateLimit:
    """What one pool allows. At least one of the two bounds must be set."""

    requests_per_second: float | None = None
    burst: int = 1
    max_in_flight: int | None = None

    def __post_init__(self) -> None:
        if self.requests_per_second is None and self.max_in_flight is None:
            msg = "a rate limit needs requests_per_second, max_in_flight, or both"
            raise ValueError(msg)
        if self.requests_per_second is not None and self.requests_per_second <= 0:
            msg = f"requests_per_second must be positive, got {self.requests_per_second}"
            raise ValueError(msg)
        if self.burst < 1:
            msg = f"burst must be at least 1, got {self.burst}"
            raise ValueError(msg)
        if self.max_in_flight is not None and self.max_in_flight < 1:
            msg = f"max_in_flight must be at least 1, got {self.max_in_flight}"
            raise ValueError(msg)


class RateLimiter:
    """A token bucket plus an in-flight bound, shared by every loader in one pool.

    :meth:`acquire`/:meth:`release` and the ``async with`` form are for coroutines on the engine's
    event loop; :attr:`sync` is the blocking form for sync loaders in worker threads. Tokens are
    reserved in call order: a caller that finds the bucket empty takes the next token and sleeps
    until it is due, so a burst of callers drains at exactly the configured rate with no thundering
    herd. ``acquired`` and ``waited_s`` are the pool's statistics for the run log.
    """

    def __init__(self, limit: RateLimit | None = None, *, name: str = "unlimited", clock: Clock = time.monotonic, sleep: Sleep = asyncio.sleep) -> None:
        self.name = name
        self.limit = limit
        self.acquired = 0
        self.waited_s = 0.0
        self._clock = clock
        self._sleep = sleep
        self._slots = None if limit is None or limit.max_in_flight is None else asyncio.Semaphore(limit.max_in_flight)
        self._tokens = 0.0 if limit is None else float(limit.burst)
        self._refilled_at = clock()
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    @classmethod
    def unlimited(cls) -> Self:
        """A limiter that never waits; what a loader gets when its dataset names no pool."""
        return cls()

    @property
    def is_limited(self) -> bool:
        """False for :meth:`unlimited`."""
        return self.limit is not None

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The event loop this limiter was created on, which the sync bridge submits to."""
        return self._loop

    async def acquire(self) -> None:
        """Wait for an in-flight slot and then a token. Pair with :meth:`release` once the call is done."""
        if self.limit is None:
            return
        if self._slots is not None:
            await self._slots.acquire()
        try:
            delay = self._reserve_token()
            if delay > 0.0:
                self.waited_s += delay
                await self._sleep(delay)
        except BaseException:
            self.release()  # a caller cancelled while waiting must not keep its slot
            raise
        self.acquired += 1

    def release(self) -> None:
        """Give the in-flight slot back."""
        if self._slots is not None:
            self._slots.release()

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.release()

    @property
    def sync(self) -> "SyncLimiter":
        """The blocking form, for sync loaders running in the engine's worker threads."""
        return SyncLimiter(self)

    def _reserve_token(self) -> float:
        """Take the next token and return how long until it is valid; the bucket may go negative."""
        limit = self.limit
        if limit is None or limit.requests_per_second is None:
            return 0.0
        now = self._clock()
        self._tokens = min(float(limit.burst), self._tokens + (now - self._refilled_at) * limit.requests_per_second)
        self._refilled_at = now
        self._tokens -= 1.0
        return 0.0 if self._tokens >= 0.0 else -self._tokens / limit.requests_per_second


class SyncLimiter:
    """``with request.rate_limiter.sync:`` — blocking access to a limiter from a worker thread."""

    def __init__(self, limiter: RateLimiter) -> None:
        self._limiter = limiter

    def __enter__(self) -> Self:
        limiter = self._limiter
        if limiter.limit is None:
            return self
        if limiter.loop is None:
            msg = f"rate limiter {limiter.name!r} was created outside an event loop; the engine binds one when loading starts"
            raise RuntimeError(msg)
        if _on_event_loop_thread():
            msg = "use `async with request.rate_limiter` inside a coroutine; the sync bridge would block the event loop"
            raise RuntimeError(msg)
        asyncio.run_coroutine_threadsafe(limiter.acquire(), limiter.loop).result()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        limiter = self._limiter
        if limiter.limit is not None and limiter.loop is not None:
            limiter.loop.call_soon_threadsafe(limiter.release)


def _on_event_loop_thread() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


async def fan_out[T, R](items: Iterable[T], call: Callable[[T], Awaitable[R]], *, limiter: RateLimiter) -> list[R]:
    """Run ``call`` for every item concurrently, each under ``limiter``, and return the results in item order.

    Every call is started immediately; the limiter decides when each one actually runs. If a call
    raises, the others are cancelled and the failures propagate as an ``ExceptionGroup``.
    """

    async def one(item: T) -> R:
        async with limiter:
            return await call(item)

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(one(item)) for item in items]
    return [task.result() for task in tasks]
