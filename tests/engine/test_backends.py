"""Tier 1/2: the backend seam — a window of tasks consumed in order, fail-fast cancellation, a dead worker, a stale worker, and a cluster that never comes up."""

from collections.abc import Callable
from concurrent.futures import BrokenExecutor
from dataclasses import dataclass, field
from pathlib import Path

from pandas.testing import assert_frame_equal

from portfolio_optimizer.domain.results import PortfolioFailure, PortfolioResult
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.backends import Backend, ClusterError, Pending, SharedRunData, Task, TaskOutput, WorkersReady
from portfolio_optimizer.engine.environment import GitInfo, WorkerEnvironment, environment_for, external_modules
from portfolio_optimizer.engine.load import assemble, load_datasets
from portfolio_optimizer.engine.runner import EXIT_INFRASTRUCTURE, EXIT_OK, EXIT_PORTFOLIO_FAILED, RunReport, run
from portfolio_optimizer.engine.tasks import BuildResult, build_task
from portfolio_optimizer.settings import ExecutionSettings
from tests.conftest import EXAMPLE_DATA, example_config, example_config_real, execution_on, io_context, resolved_example_real

GIT = GitInfo(sha="0123456789abcdef", dirty=False)
NO_CHAIN_CONSTRAINTS = ["trade_balance", "long_only", "max_weight", "cash_bounds", "turnover_cap", "sector_bounds"]

type Tamper = Callable[[TaskOutput[object]], TaskOutput[object]]


@dataclass(slots=True)
class LazyPending:
    """A future that runs its task when the result is first asked for, so a test can see how many were queued ahead."""

    run: Callable[[], object]
    portfolio_id: PortfolioId
    backend: "LazyBackend"
    cancelled: bool = False

    def result(self, timeout: float | None = None) -> TaskOutput[object]:
        del timeout
        self.backend.outstanding_at_first_result = self.backend.outstanding_at_first_result or len(self.backend.pending)
        self.backend.pending.remove(self)
        if self.backend.dead_worker:
            msg = "worker died"
            raise BrokenExecutor(msg)
        output = self.run()
        assert isinstance(output, TaskOutput)
        erased: TaskOutput[object] = TaskOutput(outcome=output.outcome, environment=output.environment, host=output.host)
        return self.backend.tamper(erased)

    def cancel(self) -> None:
        self.cancelled = True
        self.backend.cancelled.append(self.portfolio_id)


def _shared_arg(shared: object) -> SharedRunData:
    assert isinstance(shared, SharedRunData)
    return shared


@dataclass(slots=True)
class LazyBackend:
    """In-process backend that records the lifecycle the runner drives it through."""

    kind: str = "lazy"
    fail_ready: bool = False
    dead_worker: bool = False
    tamper: Tamper = lambda output: output
    started: bool = False
    closed: bool = False
    scaled_to: int | None = None
    shared_count: int = 0
    submitted: list[PortfolioId] = field(default_factory=list)
    pending: list[LazyPending] = field(default_factory=list)
    cancelled: list[PortfolioId] = field(default_factory=list)
    outstanding_at_first_result: int = 0

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

    def share(self, data: SharedRunData) -> SharedRunData:
        self.shared_count += 1
        return data

    def submit[T](self, task: Task[T], shared: object, portfolio_id: PortfolioId) -> Pending[TaskOutput[T]]:
        self.submitted.append(portfolio_id)
        pending = LazyPending(lambda: task(_shared_arg(shared), portfolio_id), portfolio_id, self)
        self.pending.append(pending)
        return pending  # ty: ignore[invalid-return-type]  # see above: the fake erases T

    def close(self) -> None:
        self.closed = True


def execute(
    tmp_path: Path, backend: LazyBackend, *, mode: str = "parallel_build_sequential_solve", max_workers: int = 2, on_error: str = "fail_fast", data_root: Path = EXAMPLE_DATA, run_id: str = "run-test"
) -> RunReport:
    constraints = NO_CHAIN_CONSTRAINTS if mode == "parallel" or on_error == "continue" else None
    overrides: dict[str, object] = {"constraints": constraints} if constraints is not None else {}
    resolved = resolved_example_real(execution={"mode": mode, "on_error": on_error}, sink="orders_to_parquet", **overrides)
    execution = execution_on("tcp://fake:8786", max_workers=max_workers)

    def factory(execution: ExecutionSettings, *, run_id: str) -> Backend:
        del execution, run_id
        return backend

    return run(resolved, io_context(tmp_path / run_id, data_root=data_root, run_id=run_id), execution=execution, git=GIT, config_path="c.json", settings={}, backend_factory=factory)


def test_the_runner_drives_the_lifecycle_and_keeps_a_window_of_tasks_outstanding(tmp_path: Path) -> None:
    backend = LazyBackend()
    report = execute(tmp_path, backend, max_workers=1)
    assert report.exit_code == EXIT_OK
    assert backend.started and backend.closed
    assert backend.scaled_to == 1
    assert backend.shared_count == 1, "shared data is delivered once per run, not once per task"
    assert backend.submitted == ["P1", "P2"]
    assert backend.outstanding_at_first_result == 2, "window is twice max_workers, so both were queued before the first result was needed"
    assert backend.cancelled == []
    cluster = report.manifest.cluster
    assert cluster is not None
    assert (cluster.kind, cluster.min_workers, cluster.max_workers, cluster.workers_ready, cluster.scheduler_address) == ("lazy", 1, 1, 1, "fake://scheduler")
    assert cluster.provision_started_at is not None and cluster.first_worker_ready_at is not None and cluster.closed_at is not None
    (worker,) = report.manifest.versions.workers
    same_config = resolved_example_real(execution={"mode": "parallel_build_sequential_solve", "on_error": "fail_fast"}, sink="orders_to_parquet").config
    assert worker.portfolios == 2 and worker.environment == environment_for(same_config, cwd=Path.cwd(), image_digest=None)


def test_a_worker_whose_environment_differs_fails_its_portfolios_at_stage_worker(tmp_path: Path) -> None:
    def stale(output: TaskOutput[object]) -> TaskOutput[object]:
        assert output.environment is not None
        return TaskOutput(outcome=output.outcome, environment=output.environment.model_copy(update={"git_sha": "stale"}), host="pod-7")

    report = execute(tmp_path, LazyBackend(tamper=stale), on_error="continue")
    assert report.exit_code == EXIT_PORTFOLIO_FAILED
    for outcome in report.outcomes:
        assert isinstance(outcome, PortfolioFailure)
        assert (outcome.stage, outcome.error_type) == ("worker", "EnvironmentMismatch")
        assert "git_sha: '0" not in outcome.message and "'stale' there" in outcome.message
    (worker,) = report.manifest.versions.workers
    assert worker.hosts == ("pod-7",) and worker.environment.git_sha == "stale"


def test_a_worker_that_dies_is_a_per_portfolio_failure_and_fail_fast_cancels_the_rest(tmp_path: Path) -> None:
    backend = LazyBackend(dead_worker=True)
    report = execute(tmp_path, backend)
    assert report.exit_code == EXIT_PORTFOLIO_FAILED
    first, second = report.outcomes
    assert isinstance(first, PortfolioFailure) and (first.stage, first.error_type) == ("worker", "BrokenExecutor")
    assert isinstance(second, PortfolioFailure) and second.stage == "skipped"
    assert backend.cancelled == ["P2"]


def test_a_cluster_that_never_comes_up_is_infrastructure_and_still_leaves_a_manifest(tmp_path: Path) -> None:
    backend = LazyBackend(fail_ready=True)
    report = execute(tmp_path, backend)
    assert report.exit_code == EXIT_INFRASTRUCTURE
    assert backend.closed
    assert [(o.stage, o.error_type) for o in report.outcomes if isinstance(o, PortfolioFailure)] == [("skipped", "ClusterUnavailable")] * 2
    records = {record.portfolio_id: record for record in report.manifest.portfolios}
    assert records["*"].failure_stage == "cluster" and records["*"].error is not None and "no worker within the timeout" in records["*"].error
    assert report.manifest.cluster is not None and report.manifest.cluster.first_worker_ready_at is None
    assert report.manifest_path.exists()


def test_parallel_mode_through_a_backend_matches_the_sequential_run(tmp_path: Path) -> None:
    parallel = execute(tmp_path, LazyBackend(), mode="parallel", run_id="par")
    resolved = resolved_example_real(execution={"mode": "sequential", "on_error": "fail_fast"}, sink="orders_to_parquet", constraints=NO_CHAIN_CONSTRAINTS)
    sequential = run(resolved, io_context(tmp_path / "seq", run_id="seq"), execution=execution_on("tcp://fake:8786", max_workers=1), git=GIT, config_path="c.json", settings={})
    assert parallel.exit_code == EXIT_OK
    for left, right in zip(sequential.solved, parallel.solved, strict=True):
        assert_frame_equal(left.orders.drop(columns=["run_id"]), right.orders.drop(columns=["run_id"]))
    assert report_kinds(sequential) == report_kinds(parallel) == [PortfolioResult, PortfolioResult]


def report_kinds(report: RunReport) -> list[type]:
    return [type(outcome) for outcome in report.outcomes]


# --- the task functions and the shared-data handles ---


def _shared() -> SharedRunData:
    resolved = resolved_example_real(sink="orders_to_parquet")  # every step from the template modules: a spawned worker cannot import tests.conftest
    assembled = assemble(load_datasets(resolved, data_root=EXAMPLE_DATA, run_id="run-x"), resolved)
    return SharedRunData(assembled=assembled, config=resolved.config, config_sha256="example", run_id="run-x")


def test_build_task_slices_rules_and_builds_from_the_shared_data_and_reports_its_environment() -> None:
    shared = _shared()
    output = build_task(shared, PortfolioId("P1"))
    assert isinstance(output.outcome, BuildResult)
    assert output.outcome.spec.security_ids == ("A", "B", "C")
    assert output.environment == environment_for(shared.config, cwd=Path.cwd(), image_digest=None)
    assert output.host
    missing = build_task(shared, PortfolioId("P9"))
    assert isinstance(missing.outcome, PortfolioFailure) and missing.outcome.stage == "slice"


def test_a_step_package_the_worker_cannot_import_is_a_worker_failure() -> None:
    shared = _shared()
    unresolvable = SharedRunData(assembled=shared.assembled, config=example_config_real(sink="no_such_package.sinks:publish"), config_sha256="other", run_id="run-y")
    output = build_task(unresolvable, PortfolioId("P1"))
    assert isinstance(output.outcome, PortfolioFailure) and (output.outcome.stage, output.outcome.error_type) == ("worker", "ConfigResolutionError")
    assert output.environment.packages == (("no_such_package", "unknown"),)


def test_environment_fingerprint_names_what_differs_and_which_packages_it_covers() -> None:
    config = example_config()
    here = environment_for(config, cwd=Path.cwd(), image_digest=None)
    there = environment_for(config, cwd=Path.cwd(), image_digest="sha256:abc")
    assert here.differences(here) == []
    assert here.differences(there) == ["image_digest: None here, 'sha256:abc' there"]
    assert external_modules(config) == ("tests.conftest",)
    assert dict(here.packages) == {"tests": "unknown"}
    assert isinstance(here, WorkerEnvironment) and hash(here) == hash(environment_for(config, cwd=Path.cwd(), image_digest=None))
