"""Where per-portfolio work runs: the backend seam.

A backend is what executes tasks. There are two: :class:`InlineBackend`, this process, one task
after another — the default, which needs no cluster and is where a rule is debugged — and a Dask
cluster the run owns (``engine/dask_backend.py``): local worker processes on a laptop, pods a Dask
Gateway creates, or a scheduler someone else runs. The seam is also what the runner is exercised
against with a fake. The runner asks
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
    packages: tuple[str, ...] | None = None
    """The step packages a qualified name may import from, so every worker resolves the config under the run's own allowlist; ``None`` allows any."""


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


@dataclass(frozen=True, slots=True)
class Done[T]:
    """A task that has already run: its value, or the exception it — or a dependency — raised; satisfies :class:`Pending`."""

    value: T | None = None
    error: Exception | None = None

    def result(self, timeout: float | None = None) -> T:
        """The value, or the error re-raised, as a Dask future does for a task whose dependency died."""
        del timeout
        if self.error is not None:
            raise self.error
        return self.value  # ty: ignore[invalid-return-type]  # a done task without an error holds its value


class InlineBackend:
    """Every task runs in this process the moment it is submitted, one after another.

    Nothing is provisioned, shared, or cancelled: ``submit`` resolves its pending arguments, runs the
    function, and remembers the answer, so the runner sees exactly the seam a cluster gives it — a
    dependency's failure reaches its dependents as a raised handle — with no worker in between. The
    default backend, and the one a rule is stepped through under a debugger with.
    """

    kind = "inline"

    def start(self) -> None:
        """Nothing to provision."""

    def scale(self, workers: int) -> None:
        """One process is all there is."""
        del workers

    def ready(self, workers: int, timeout_s: float) -> WorkersReady:
        """Always ready: this process is the worker."""
        del workers, timeout_s
        return WorkersReady(workers=1, scheduler_address=None)

    def probe[T](self, fn: Callable[..., T], /, *args: object) -> Mapping[str, T]:
        """``fn`` here, keyed by this backend's name."""
        return {self.kind: fn(*args)}

    def share(self, data: SharedRunData) -> SharedRunData:
        """The data itself; there is nowhere to send it."""
        return data

    def submit[T](self, fn: Callable[..., T], /, *args: object, key: str, priority: int) -> Pending[T]:
        """Run ``fn`` now over its resolved arguments; a dependency that raised makes this task raise the same, unrun."""
        del key, priority
        resolved: list[object] = []
        for argument in args:
            if isinstance(argument, Done):
                if argument.error is not None:
                    failed: Done[T] = Done(error=argument.error)
                    return failed
                resolved.append(argument.value)
            else:
                resolved.append(argument)
        try:
            done: Done[T] = Done(value=fn(*resolved))
        except Exception as error:  # noqa: BLE001  # remembered and re-raised on the handle, as a Dask future does
            done = Done(error=error)
        return done

    def as_completed(self, pendings: Mapping[PortfolioId, Pending[object]]) -> Iterator[PortfolioId]:
        """Every task is already done, so submission order is completion order."""
        yield from list(pendings)

    def cancel(self, pendings: Sequence[Pending[object]]) -> None:
        """Nothing is pending; there is nothing to cancel."""
        del pendings

    def close(self) -> None:
        """Nothing to release."""
