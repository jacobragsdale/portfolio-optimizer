"""Where per-portfolio work runs: the backend seam.

A backend is what executes tasks. There is one real implementation — a Dask cluster the run owns
(``engine/dask_backend.py``): local worker processes on a laptop, pods a Dask Gateway creates, or a scheduler
someone else runs — and the seam exists so the runner can be exercised against a fake. The runner asks
for the backend right after config resolution (:meth:`Backend.start`, non-blocking, so the cluster
warms up under the load stage), scales it and waits for the first worker after assembly
(:meth:`Backend.scale`, :meth:`Backend.ready`), checks every worker that has joined can do the run's
work (:meth:`Backend.probe`), hands it the run's shared data once
(:meth:`Backend.share`), submits every portfolio's build and then, once the schedule is known, every
solve with its predecessors' contributions as dependencies (:meth:`Backend.submit`), collects results
as they complete (:meth:`Backend.as_completed`), cancels what a failure makes pointless
(:meth:`Backend.cancel`), and closes it in a ``finally`` (:meth:`Backend.close`). The task functions
(``engine/tasks.py``) never raise for a portfolio's own failure: each returns a :class:`TaskOutput`
whose outcome is the portfolio's result or failure and whose environment is the fingerprint of the
process that produced it. A task whose *dependency* raised — a worker died under it — never runs,
and its handle raises instead.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from portfolio_optimizer.config.models import RunConfig
from portfolio_optimizer.domain.results import PortfolioFailure
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.environment import WorkerEnvironment
from portfolio_optimizer.engine.load import AssembledDatasets
from portfolio_optimizer.engine.timing import Span
from portfolio_optimizer.settings import ExecutionSettings


class ClusterError(RuntimeError):
    """The backend could not provide a worker: provisioning failed or timed out."""


class WorkerEnvironmentError(ClusterError):
    """A worker the run started with cannot do its work: the config does not resolve there, or its environment differs from the run's."""


@dataclass(frozen=True, slots=True)
class SharedRunData:
    """Everything a task needs besides its portfolio id; shipped to each worker once per run, never per task."""

    assembled: AssembledDatasets
    config: RunConfig
    config_sha256: str
    run_id: str


@dataclass(frozen=True, slots=True)
class TaskOutput[T]:
    """What every task returns: the portfolio's outcome, the fingerprint and host of the process that produced it, and the spans it timed.

    ``spans`` is observability, never identity: the runner folds them into the manifest's ``timing``
    block, which ``diff-manifests`` does not compare.
    """

    outcome: T | PortfolioFailure
    environment: WorkerEnvironment
    host: str
    spans: tuple[Span, ...] = ()


class Pending[T](Protocol):
    """The part of a future the runner uses."""

    def result(self, timeout: float | None = None) -> T: ...


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
        """``local``, ``gateway``, or ``address``, for the manifest."""
        ...

    def start(self) -> None:
        """Begin provisioning and return at once."""
        ...

    def scale(self, workers: int) -> None:
        """Ask for ``workers`` in total; non-blocking."""
        ...

    def ready(self, workers: int, timeout_s: float) -> WorkersReady:
        """Block until at least ``workers`` can take a task, or raise :class:`ClusterError`."""
        ...

    def probe[T](self, fn: Callable[..., T], /, *args: object) -> Mapping[str, T]:
        """Run ``fn(*args)`` once on every worker connected now and return the results by worker address.

        Called once, after :meth:`ready` and before :meth:`share`, with a function that never raises for
        its own findings; a worker that raises anyway is the backend's failure to report.
        """
        ...

    def share(self, data: SharedRunData) -> object:
        """Deliver the run's shared data to the workers once; the handle returned is what :meth:`submit` forwards."""
        ...

    def submit[T](self, fn: Callable[..., T], /, *args: object, key: str, priority: int) -> Pending[T]:
        """Schedule ``fn(*args)`` under ``key``; higher ``priority`` runs first.

        An argument may be the shared-data handle or a :class:`Pending` this backend returned: both
        are resolved on the worker before ``fn`` runs, so a pending argument is a dependency.
        """
        ...

    def as_completed(self, pendings: Mapping[PortfolioId, Pending[object]]) -> Iterator[PortfolioId]:
        """Yield each key as its pending completes, in completion order; every key exactly once."""
        ...

    def cancel(self, pendings: Sequence[Pending[object]]) -> None:
        """Drop tasks that have not started, and everything that depends on them; a running task is not interrupted."""
        ...

    def close(self) -> None:
        """Release every worker; idempotent, and always called."""
        ...


class BackendFactory(Protocol):
    """How the runner obtains a backend: the Dask backend in production, a fake in tests."""

    def __call__(self, execution: ExecutionSettings, *, run_id: str) -> Backend: ...
