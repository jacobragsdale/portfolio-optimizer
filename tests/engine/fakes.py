"""An in-process backend that behaves like Dask's seam — lazy futures, dependencies resolved first, dead and tampered workers — and records what the runner did to it.

``LazyBackend.trace`` interleaves ``submit:<key>`` with the ``run:<key>`` of the first time a task's
output was asked for, which is what shows whether the runner waited on a build before submitting the
solves that do not depend on it.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import BrokenExecutor
from dataclasses import dataclass, field

from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.backends import Backend, BackendFactory, ClusterError, Pending, SharedRunData, TaskOutput, WorkersReady
from portfolio_optimizer.settings import ExecutionSettings

type Tamper = Callable[[TaskOutput[object]], TaskOutput[object]]


@dataclass(slots=True)
class LazyPending:
    """A future that runs its function when first asked — resolving pending arguments first, as Dask does — and remembers the answer."""

    fn: Callable[..., object]
    args: tuple[object, ...]
    key: str
    backend: "LazyBackend"
    done: bool = False
    value: object = None
    error: Exception | None = None

    def result(self, timeout: float | None = None) -> object:
        del timeout
        if not self.done:
            self.done = True
            try:
                self.value = self._run()
            except Exception as error:  # noqa: BLE001  # remembered and re-raised on every ask, as a Dask future does
                self.error = error
        if self.error is not None:
            raise self.error
        return self.value

    def _run(self) -> object:
        resolved = [argument.result() if isinstance(argument, LazyPending) else argument for argument in self.args]  # a dead dependency raises here, before fn runs
        self.backend.trace.append(f"run:{self.key}")
        if self.key in self.backend.dead_keys:
            msg = "worker died"
            raise BrokenExecutor(msg)
        output = self.fn(*resolved)
        if isinstance(output, TaskOutput):
            erased: TaskOutput[object] = TaskOutput(outcome=output.outcome, environment=output.environment, host=output.host)
            return self.backend.tamper(erased)
        return output


@dataclass(slots=True)
class LazyBackend:
    """In-process backend that records the lifecycle the runner drives it through."""

    kind: str = "lazy"
    fail_ready: bool = False
    dead_keys: frozenset[str] = frozenset()
    tamper: Tamper = lambda output: output
    probe_tamper: Tamper = lambda output: output
    started: bool = False
    probed: int = 0
    closed: bool = False
    scaled_to: int | None = None
    shared_count: int = 0
    submitted: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    priorities: dict[str, int] = field(default_factory=dict)
    cancelled: list[str] = field(default_factory=list)

    def start(self) -> None:
        self.started = True

    def scale(self, workers: int) -> None:
        self.scaled_to = workers

    def ready(self, workers: int, timeout_s: float) -> WorkersReady:
        del timeout_s
        if self.fail_ready:
            msg = "no worker within the timeout"
            raise ClusterError(msg)
        return WorkersReady(workers=workers, scheduler_address="fake://scheduler")

    def probe[T](self, fn: Callable[..., T], /, *args: object) -> Mapping[str, T]:
        self.probed += 1
        output = fn(*args)
        if isinstance(output, TaskOutput):
            erased: TaskOutput[object] = TaskOutput(outcome=output.outcome, environment=output.environment, host=output.host)
            return {"fake://worker-1": self.probe_tamper(erased)}  # ty: ignore[invalid-return-type]  # the fake erases T
        return {"fake://worker-1": output}

    def share(self, data: SharedRunData) -> SharedRunData:
        self.shared_count += 1
        return data

    def submit[T](self, fn: Callable[..., T], /, *args: object, key: str, priority: int) -> Pending[T]:
        self.submitted.append(key)
        self.trace.append(f"submit:{key}")
        self.priorities[key] = priority
        pending = LazyPending(fn, args, key, self)
        return pending  # ty: ignore[invalid-return-type]  # the fake erases T

    def as_completed(self, pendings: Mapping[PortfolioId, Pending[object]]) -> Iterator[PortfolioId]:
        yield from list(pendings)

    def cancel(self, pendings: Sequence[Pending[object]]) -> None:
        self.cancelled.extend(pending.key for pending in pendings if isinstance(pending, LazyPending))

    def close(self) -> None:
        self.closed = True


def factory_for(backend: Backend) -> BackendFactory:
    """A backend factory that hands the runner ``backend`` whatever the settings say, so a test can inspect it afterwards."""

    def factory(execution: ExecutionSettings, *, run_id: str) -> Backend:
        del execution, run_id
        return backend

    return factory
