"""Where per-portfolio work runs: the backend seam and the local backends behind it.

A backend is what executes tasks — a pool of spawned processes, threads, or a Dask cluster the run
owns (``engine/dask_backend.py``). The runner asks for it right after config resolution (:meth:`Backend.start`,
non-blocking, so a cluster warms up under the load stage), scales it and waits for the first worker
after assembly (:meth:`Backend.scale`, :meth:`Backend.ready`), hands it the run's shared data once
(:meth:`Backend.share`), submits one task per portfolio carrying only the portfolio id
(:meth:`Backend.submit`), and closes it in a ``finally`` (:meth:`Backend.close`). The task functions
(``engine/tasks.py``) are the same whatever the backend; a task never raises, it returns a
:class:`TaskOutput` whose outcome is the portfolio's result or failure and whose environment is the
fingerprint of the process that produced it.
"""

import importlib
import logging
import threading
from collections.abc import Callable
from concurrent.futures import BrokenExecutor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.synchronize import Barrier
from typing import Protocol, runtime_checkable

from portfolio_optimizer.config.models import RunConfig
from portfolio_optimizer.domain.results import PortfolioFailure
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.environment import WorkerEnvironment
from portfolio_optimizer.engine.load import AssembledDatasets
from portfolio_optimizer.settings import ExecutionSettings

log = logging.getLogger(__name__)

PROCESS_START_TIMEOUT_S = 300.0
"""How long a local pool may take to spawn its interpreters before the run gives up on it."""


class ExecutionSettingsError(ValueError):
    """The execution settings cannot run this config."""


def check_execution(config: RunConfig, execution: ExecutionSettings) -> None:
    """Refuse a mode/executor pair that cannot work, before anything loads."""
    if config.execution.mode == "parallel" and execution.executor == "thread":
        msg = "execution.mode 'parallel' solves in the worker, which executor 'thread' cannot do: cvxpy solves are not thread-safe; use 'process' or 'dask'"
        raise ExecutionSettingsError(msg)


class ClusterError(RuntimeError):
    """The backend could not provide a worker: provisioning failed or timed out."""


@dataclass(frozen=True, slots=True)
class SharedRunData:
    """Everything a task needs besides its portfolio id; shipped to each worker once per run, never per task."""

    assembled: AssembledDatasets
    config: RunConfig
    config_sha256: str
    run_id: str


@dataclass(frozen=True, slots=True)
class SharedRef:
    """What a process-pool task receives in place of the data: the run id, looked up in the worker's registry."""

    run_id: str


type SharedArg = SharedRunData | SharedRef


@dataclass(frozen=True, slots=True)
class TaskOutput[T]:
    """What every task returns: the portfolio's outcome and the fingerprint and host of the process that produced it.

    ``environment`` is ``None`` only when the worker could not even describe itself — it was handed a run
    it holds no shared data for — and the outcome is then already a failure at stage ``worker``.
    """

    outcome: T | PortfolioFailure
    environment: WorkerEnvironment | None
    host: str


type Task[T] = Callable[[SharedArg, PortfolioId], TaskOutput[T]]


class Pending[T](Protocol):
    """The part of a future the runner uses; satisfied by ``concurrent.futures.Future`` and ``distributed.Future`` alike."""

    def result(self, timeout: float | None = None) -> T: ...

    def cancel(self) -> object: ...


@dataclass(frozen=True, slots=True)
class WorkersReady:
    """What :meth:`Backend.ready` reports once tasks can run."""

    workers: int
    scheduler_address: str | None


@runtime_checkable
class Backend(Protocol):
    """The seam between the runner and whatever executes tasks; see the module docstring for the lifecycle."""

    @property
    def kind(self) -> str:
        """``process``, ``thread``, ``local``, ``kubernetes``, or ``address``, for the manifest."""
        ...

    def start(self) -> None:
        """Begin provisioning and return at once."""
        ...

    def scale(self, workers: int) -> None:
        """Ask for ``workers`` in total; non-blocking. A pool that is sized once ignores it."""
        ...

    def ready(self, workers: int, timeout_s: float) -> WorkersReady:
        """Block until at least ``workers`` can take a task, or raise :class:`ClusterError`."""
        ...

    def share(self, data: SharedRunData) -> object:
        """Deliver the run's shared data to the workers once; the handle returned is what :meth:`submit` forwards."""
        ...

    def submit[T](self, task: Task[T], shared: object, portfolio_id: PortfolioId) -> Pending[TaskOutput[T]]:
        """Schedule ``task(shared, portfolio_id)``."""
        ...

    def close(self) -> None:
        """Release every worker; idempotent, and always called."""
        ...


class BackendFactory(Protocol):
    """How the runner obtains a backend; :func:`make_backend` in production, a fake in tests."""

    def __call__(self, execution: ExecutionSettings, *, run_id: str) -> Backend: ...


def make_backend(execution: ExecutionSettings, *, run_id: str) -> Backend:
    """The backend the execution settings ask for; the Dask one is imported only when asked for, since its dependency is optional."""
    if execution.executor == "process":
        return ProcessBackend(execution.max_workers)
    if execution.executor == "thread":
        return ThreadBackend(execution.max_workers)
    module = importlib.import_module("portfolio_optimizer.engine.dask_backend")
    backend = module.DaskBackend(execution, run_id=run_id)
    if not isinstance(backend, Backend):
        msg = f"{module.__name__}.DaskBackend does not implement the backend seam"
        raise TypeError(msg)
    return backend


# --- the process pool: spawned interpreters, shared data delivered once per worker ---

_SHARED: dict[str, SharedRunData] = {}
"""Per worker process: the shared data of every run this process has been handed."""
_RENDEZVOUS: Barrier | None = None
"""Per worker process: the barrier the pool uses to address every worker exactly once."""


def resolve_shared(shared: SharedArg) -> SharedRunData:
    """The shared data behind what a task received."""
    if isinstance(shared, SharedRunData):
        return shared
    try:
        return _SHARED[shared.run_id]
    except KeyError:
        msg = f"this worker holds no shared data for run {shared.run_id!r}; the pool's share() did not reach it"
        raise ClusterError(msg) from None


def _init_worker(rendezvous: Barrier) -> None:
    global _RENDEZVOUS  # noqa: PLW0603  # one barrier per worker process, set once at spawn
    _RENDEZVOUS = rendezvous


def _rendezvous() -> Barrier:
    if _RENDEZVOUS is None:
        msg = "worker was not initialized by ProcessBackend"
        raise ClusterError(msg)
    return _RENDEZVOUS


def _warm_up() -> None:
    """Import the solver stack, then wait for every sibling: the pool spawns one interpreter per waiting task."""
    importlib.import_module("portfolio_optimizer.engine.tasks")
    _rendezvous().wait()


def _install(data: SharedRunData) -> None:
    """Keep the run's shared data, then wait for every sibling so each worker takes exactly one copy."""
    _SHARED[data.run_id] = data
    _rendezvous().wait()


class ProcessBackend:
    """A ``ProcessPoolExecutor`` of ``spawn``-ed interpreters, warmed up under the load stage.

    Every worker is addressed exactly once by making each warm-up and each install wait at a barrier
    sized to the pool, so the shared data is pickled once per worker and a task carries only a run id
    and a portfolio id.
    """

    kind = "process"

    def __init__(self, workers: int, *, start_timeout_s: float = PROCESS_START_TIMEOUT_S) -> None:
        self._workers = workers
        self._start_timeout_s = start_timeout_s
        self._pool: ProcessPoolExecutor | None = None
        self._warm: list[Future[None]] = []

    def start(self) -> None:
        """Spawn the interpreters now, so they import cvxpy while data loads."""
        context = get_context("spawn")
        rendezvous = context.Barrier(self._workers, timeout=self._start_timeout_s)
        self._pool = ProcessPoolExecutor(max_workers=self._workers, mp_context=context, initializer=_init_worker, initargs=(rendezvous,))
        self._warm = [self._pool.submit(_warm_up) for _ in range(self._workers)]

    def scale(self, workers: int) -> None:
        """A pool is sized once; the request is noted for the log and otherwise ignored."""
        if workers != self._workers:
            log.info("process pool is sized once: %d worker(s), scale(%d) ignored", self._workers, workers)

    def ready(self, workers: int, timeout_s: float) -> WorkersReady:
        """Wait for every warm-up: the barrier releases them together, so one done means all done."""
        del workers
        try:
            for future in self._warm:
                future.result(timeout=timeout_s)
        except (TimeoutError, BrokenExecutor, threading.BrokenBarrierError) as error:
            msg = f"process pool did not come up within {timeout_s:.0f}s: {type(error).__name__}: {error}"
            raise ClusterError(msg) from error
        return WorkersReady(workers=self._workers, scheduler_address=None)

    def share(self, data: SharedRunData) -> SharedRef:
        """Hand every worker the data once; the barrier inside ``_install`` is what makes it exactly once."""
        installs = [self._require_pool().submit(_install, data) for _ in range(self._workers)]
        try:
            for future in installs:
                future.result(timeout=self._start_timeout_s)
        except (TimeoutError, BrokenExecutor, threading.BrokenBarrierError) as error:
            msg = f"could not deliver shared data to every worker: {type(error).__name__}: {error}"
            raise ClusterError(msg) from error
        return SharedRef(run_id=data.run_id)

    def submit[T](self, task: Task[T], shared: object, portfolio_id: PortfolioId) -> Pending[TaskOutput[T]]:
        """Queue one portfolio."""
        return self._require_pool().submit(task, _shared_arg(shared), portfolio_id)

    def close(self) -> None:
        """Stop the pool; queued tasks are dropped, running ones finish."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def _require_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            msg = "backend was not started"
            raise ClusterError(msg)
        return self._pool


class ThreadBackend:
    """A ``ThreadPoolExecutor``; in-process, so shared data is shared by reference.

    Builds are pure Python under the GIL, so this parallelizes nothing — it exists so the pipeline can be
    stepped through in one process, and it cannot solve (the config check refuses ``parallel`` with it).
    """

    kind = "thread"

    def __init__(self, workers: int) -> None:
        self._workers = workers
        self._pool: ThreadPoolExecutor | None = None

    def start(self) -> None:
        """Create the pool; threads need no warm-up."""
        self._pool = ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix="portfolio")

    def scale(self, workers: int) -> None:
        """A pool is sized once."""
        del workers

    def ready(self, workers: int, timeout_s: float) -> WorkersReady:
        """Threads are always ready."""
        del workers, timeout_s
        return WorkersReady(workers=self._workers, scheduler_address=None)

    def share(self, data: SharedRunData) -> SharedRunData:
        """Same process: the data itself is the handle."""
        return data

    def submit[T](self, task: Task[T], shared: object, portfolio_id: PortfolioId) -> Pending[TaskOutput[T]]:
        """Queue one portfolio."""
        if self._pool is None:
            msg = "backend was not started"
            raise ClusterError(msg)
        return self._pool.submit(task, _shared_arg(shared), portfolio_id)

    def close(self) -> None:
        """Stop the pool; queued tasks are dropped, running ones finish."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None


def _shared_arg(shared: object) -> SharedArg:
    if isinstance(shared, SharedRunData | SharedRef):
        return shared
    msg = f"submit() received a {type(shared).__name__} handle, expected what share() returned"
    raise TypeError(msg)
