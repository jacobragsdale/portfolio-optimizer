"""Tier 1/2: the backend seam — workers probed before data is shared, builds first, solves along the schedule with dependencies, fail-fast cancellation, dead and stale workers, and a cluster that never comes up."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import BrokenExecutor
from dataclasses import dataclass, field
from pathlib import Path

from pandas.testing import assert_frame_equal

from portfolio_optimizer.domain.results import PortfolioFailure, PortfolioResult
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.backends import Backend, ClusterError, Pending, SharedRunData, TaskOutput, WorkersReady
from portfolio_optimizer.engine.environment import GitInfo, WorkerEnvironment, environment_for, external_modules
from portfolio_optimizer.engine.load import assemble, load_datasets
from portfolio_optimizer.engine.runner import EXIT_INFRASTRUCTURE, EXIT_OK, EXIT_PORTFOLIO_FAILED, RunContext, RunReport, run
from portfolio_optimizer.engine.tasks import BuildResult, build_task
from portfolio_optimizer.settings import ExecutionSettings
from tests.conftest import BUY_ONLY_OBJECTIVE, EXAMPLE_DATA, NO_CHAIN_CONSTRAINTS, example_config, example_config_real, execution_on, half_cash_book, io_context, resolved_example_real, sell_book

GIT = GitInfo(sha="0123456789abcdef", dirty=False)

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
        self.priorities[key] = priority
        pending = LazyPending(fn, args, key, self)
        return pending  # ty: ignore[invalid-return-type]  # the fake erases T

    def as_completed(self, pendings: Mapping[PortfolioId, Pending[object]]) -> Iterator[PortfolioId]:
        yield from list(pendings)

    def cancel(self, pendings: Sequence[Pending[object]]) -> None:
        self.cancelled.extend(pending.key for pending in pendings if isinstance(pending, LazyPending))

    def close(self) -> None:
        self.closed = True


def execute(
    tmp_path: Path,
    backend: LazyBackend,
    *,
    max_workers: int = 2,
    on_error: str = "fail_fast",
    dependencies: str = "overlap",
    data_root: Path = EXAMPLE_DATA,
    run_id: str = "run-test",
    **overrides: object,
) -> RunReport:
    resolved = resolved_example_real(execution={"on_error": on_error, "dependencies": dependencies}, sink="orders_to_parquet", **overrides)
    execution = execution_on("tcp://fake:8786", max_workers=max_workers)

    def factory(execution: ExecutionSettings, *, run_id: str) -> Backend:
        del execution, run_id
        return backend

    context = RunContext(io=io_context(tmp_path / run_id, data_root=data_root, run_id=run_id), execution=execution, git=GIT, config_path="c.json", settings={})
    return run(resolved, context, backend_factory=factory)


def test_the_runner_builds_everything_first_then_solves_along_the_schedule(tmp_path: Path) -> None:
    backend = LazyBackend()
    report = execute(tmp_path, backend, max_workers=1)
    assert report.exit_code == EXIT_OK
    assert backend.started and backend.closed
    assert backend.scaled_to == 1
    assert backend.shared_count == 1, "shared data is delivered once per run, not once per task"
    assert backend.submitted == [
        "run-test/P1/build",
        "run-test/P1/summary",
        "run-test/P2/build",
        "run-test/P2/summary",
        "run-test/P1/solve",
        "run-test/P1/contribution",
        "run-test/P2/solve",
        "run-test/P2/contribution",
    ]
    assert backend.priorities["run-test/P1/solve"] > backend.priorities["run-test/P2/solve"] > backend.priorities["run-test/P1/build"], (
        "the head of the longest chain solves first, and any solve beats any build"
    )
    assert backend.cancelled == []
    schedule = report.manifest.schedule
    assert schedule is not None
    assert (schedule.coupling, schedule.portfolios, schedule.edges, schedule.components, schedule.largest_component, schedule.critical_path) == ("overlap", 2, 1, 1, 2, 2)
    assert [(record.solve_order, record.predecessors) for record in report.manifest.portfolios] == [("0", 0), ("1", 1)]
    cluster = report.manifest.cluster
    assert cluster is not None
    assert (cluster.kind, cluster.min_workers, cluster.max_workers, cluster.workers_ready, cluster.scheduler_address) == ("lazy", 1, 1, 1, "fake://scheduler")
    assert cluster.provision_started_at is not None and cluster.first_worker_ready_at is not None and cluster.closed_at is not None
    (worker,) = report.manifest.versions.workers
    same_config = resolved_example_real(execution={"on_error": "fail_fast"}, sink="orders_to_parquet").config
    assert worker.portfolios == 2 and worker.environment == environment_for(same_config, cwd=Path.cwd(), image_digest=None)


def test_a_pure_function_solve_step_runs_the_whole_pipeline_and_is_verified(tmp_path: Path) -> None:
    report = execute(tmp_path, LazyBackend(), solve="tests.conftest:hold_still")
    assert report.exit_code == EXIT_OK
    for outcome in report.outcomes:
        assert isinstance(outcome, PortfolioResult)
        assert len(outcome.orders) == 0 and outcome.solution.objective is None and outcome.report.passed
    record = report.manifest.portfolios[0]
    assert record.solve is not None and record.solve.solver == "tests.conftest:hold_still" and record.solve.objective_value is None
    assert record.check is not None and record.check.objective_passed


def test_a_buy_only_run_reproduces_the_hand_checked_buys_and_couples_through_them(tmp_path: Path) -> None:
    report = execute(tmp_path, LazyBackend(), data_root=half_cash_book(tmp_path), sides="buy", objective=BUY_ONLY_OBJECTIVE)
    assert report.exit_code == EXIT_OK, [outcome for outcome in report.outcomes if isinstance(outcome, PortfolioFailure)]
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioResult) and isinstance(p2, PortfolioResult)
    assert p1.orders[["security_id", "side", "quantity"]].to_dict("records") == [
        {"security_id": "A", "side": "BUY", "quantity": 1250},
        {"security_id": "B", "side": "BUY", "quantity": 2500},
        {"security_id": "C", "side": "BUY", "quantity": 25000},
    ]
    assert p2.orders[["security_id", "side", "quantity"]].to_dict("records") == [{"security_id": "A", "side": "BUY", "quantity": 2500}, {"security_id": "B", "side": "BUY", "quantity": 5000}]
    assert p2.chain_state.traded_shares.tolist() == [1250.0, 2500.0, 25000.0] and p2.chain_state.predecessors == ("P1",), (
        "every buy P1 made reaches P2: under buy-only the chain carries the whole trade"
    )
    for outcome in (p1, p2):
        assert outcome.report.passed and (outcome.solution.w >= outcome.spec.w0 - 1e-9).all() and outcome.solution.sell.tolist() == [0.0, 0.0, 0.0]
    assert report.manifest.config.resolved["sides"] == "buy"


def test_a_sell_only_run_reproduces_the_hand_checked_sells_and_couples_through_them(tmp_path: Path) -> None:
    report = execute(tmp_path, LazyBackend(), data_root=sell_book(tmp_path), sides="sell")
    assert report.exit_code == EXIT_OK, [outcome for outcome in report.outcomes if isinstance(outcome, PortfolioFailure)]
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioResult) and isinstance(p2, PortfolioResult)
    assert p1.orders[["security_id", "side", "quantity"]].to_dict("records") == [{"security_id": "A", "side": "SELL", "quantity": 1000}, {"security_id": "B", "side": "SELL", "quantity": 3333}]
    assert p2.orders[["security_id", "side", "quantity"]].to_dict("records") == [{"security_id": "B", "side": "SELL", "quantity": 3333}], "P1's sells of A used up its ADV budget"
    assert p2.chain_state.traded_shares.tolist() == [1000.0, 3333.0, 0.0] and p2.chain_state.predecessors == ("P1",)
    for outcome in (p1, p2):
        assert outcome.report.passed and (outcome.solution.w <= outcome.spec.w0 + 1e-9).all() and outcome.solution.buy.tolist() == [0.0, 0.0, 0.0]
    assert report.manifest.config.resolved["sides"] == "sell"


def test_nothing_reading_the_chain_means_no_portfolio_waits(tmp_path: Path) -> None:
    report = execute(tmp_path, LazyBackend(), constraints=NO_CHAIN_CONSTRAINTS)
    assert report.exit_code == EXIT_OK
    schedule = report.manifest.schedule
    assert schedule is not None and (schedule.coupling, schedule.edges, schedule.components) == ("none", 0, 2)
    p1, p2 = report.solved
    assert len(p1.orders) == 3 and len(p2.orders) == 3, "without the chained ADV cap P2 buys C too"
    assert p2.chain_state.predecessors == ()


def test_a_worker_that_cannot_resolve_the_config_stops_the_run_before_any_data_is_shared(tmp_path: Path) -> None:
    def missing_solver(output: TaskOutput[object]) -> TaskOutput[object]:
        rejected = PortfolioFailure("*", "worker", "ConfigResolutionError", "1 config resolution failure(s): solver: solver 'CLARABEL' is not installed in this environment; installed: []")
        return TaskOutput(outcome=rejected, environment=output.environment, host="pod-3")

    backend = LazyBackend(probe_tamper=missing_solver)
    report = execute(tmp_path, backend)
    assert report.exit_code == EXIT_INFRASTRUCTURE
    assert backend.probed == 1 and backend.shared_count == 0 and backend.submitted == [] and backend.closed
    assert [(o.stage, o.error_type) for o in report.outcomes if isinstance(o, PortfolioFailure)] == [("skipped", "ClusterUnavailable")] * 2
    records = {record.portfolio_id: record for record in report.manifest.portfolios}
    assert records["*"].failure_stage == "cluster" and records["*"].error is not None
    assert "worker pod-3 (fake://worker-1) cannot resolve the config" in records["*"].error and "'CLARABEL' is not installed" in records["*"].error
    (worker,) = report.manifest.versions.workers
    assert worker.hosts == ("pod-3",) and worker.portfolios == 0


def test_a_worker_whose_environment_differs_at_the_start_stops_the_run(tmp_path: Path) -> None:
    def stale(output: TaskOutput[object]) -> TaskOutput[object]:
        return TaskOutput(outcome=output.outcome, environment=output.environment.model_copy(update={"image_digest": "sha256:old"}), host="pod-7")

    backend = LazyBackend(probe_tamper=stale)
    report = execute(tmp_path, backend)
    assert report.exit_code == EXIT_INFRASTRUCTURE
    assert backend.shared_count == 0 and backend.submitted == []
    records = {record.portfolio_id: record for record in report.manifest.portfolios}
    assert records["*"].error is not None and "worker pod-7 (fake://worker-1) runs a different environment: image_digest: None here, 'sha256:old' there" in records["*"].error


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
    by_hosts = {worker.hosts: worker for worker in report.manifest.versions.workers}
    assert by_hosts[("pod-7",)].environment.git_sha == "stale", "the tasks' environment is recorded beside the probe's"
    assert report.manifest.schedule is not None and report.manifest.schedule.edges == 1, "two failed builds are treated as overlapping"


def test_a_dead_worker_under_a_solve_is_a_worker_failure_and_fail_fast_cancels_the_rest(tmp_path: Path) -> None:
    backend = LazyBackend(dead_keys=frozenset({"run-test/P1/solve"}))
    report = execute(tmp_path, backend)
    assert report.exit_code == EXIT_PORTFOLIO_FAILED
    first, second = report.outcomes
    assert isinstance(first, PortfolioFailure) and (first.stage, first.error_type) == ("worker", "BrokenExecutor")
    assert isinstance(second, PortfolioFailure) and second.stage == "skipped"
    assert backend.cancelled == ["run-test/P2/solve"]


def test_under_continue_a_dead_predecessor_skips_only_the_portfolios_that_depended_on_it(tmp_path: Path) -> None:
    backend = LazyBackend(dead_keys=frozenset({"run-test/P1/solve"}))
    report = execute(tmp_path, backend, on_error="continue")
    first, second = report.outcomes
    assert isinstance(first, PortfolioFailure) and first.stage == "worker"
    assert isinstance(second, PortfolioFailure) and second.stage == "skipped" and "predecessor 'P1' failed at stage 'worker'" in second.message
    assert backend.cancelled == []


def test_a_dead_worker_under_a_build_stops_a_fail_fast_run_before_any_solve(tmp_path: Path) -> None:
    backend = LazyBackend(dead_keys=frozenset({"run-test/P1/build"}))
    report = execute(tmp_path, backend)
    first, second = report.outcomes
    assert isinstance(first, PortfolioFailure) and (first.stage, first.error_type) == ("worker", "BrokenExecutor")
    assert isinstance(second, PortfolioFailure) and second.stage == "skipped"
    assert not any(key.endswith("/solve") for key in backend.submitted)


def test_a_cluster_that_never_comes_up_is_infrastructure_and_still_leaves_a_manifest(tmp_path: Path) -> None:
    backend = LazyBackend(fail_ready=True)
    report = execute(tmp_path, backend)
    assert report.exit_code == EXIT_INFRASTRUCTURE
    assert backend.closed
    assert [(o.stage, o.error_type) for o in report.outcomes if isinstance(o, PortfolioFailure)] == [("skipped", "ClusterUnavailable")] * 2
    records = {record.portfolio_id: record for record in report.manifest.portfolios}
    assert records["*"].failure_stage == "cluster" and records["*"].error is not None and "no worker within the timeout" in records["*"].error
    assert report.manifest.cluster is not None and report.manifest.cluster.first_worker_ready_at is None
    assert report.manifest.schedule is None
    assert report.manifest_path.exists()


def test_every_earlier_portfolio_as_a_predecessor_gives_the_same_answer_as_overlap(tmp_path: Path) -> None:
    overlap = execute(tmp_path, LazyBackend(), run_id="overlap")
    line = execute(tmp_path, LazyBackend(), dependencies="all", run_id="line")
    assert overlap.exit_code == line.exit_code == EXIT_OK
    for left, right in zip(overlap.solved, line.solved, strict=True):
        assert_frame_equal(left.orders.drop(columns=["run_id"]), right.orders.drop(columns=["run_id"]))
        assert left.chain_state.content_hash() == right.chain_state.content_hash()
    assert report_kinds(overlap) == report_kinds(line) == [PortfolioResult, PortfolioResult]
    assert line.manifest.schedule is not None and line.manifest.schedule.coupling == "all"


def report_kinds(report: RunReport) -> list[type]:
    return [type(outcome) for outcome in report.outcomes]


# --- the task functions and the shared-data handles ---


def _shared() -> SharedRunData:
    resolved = resolved_example_real(sink="orders_to_parquet")  # every step from the template modules: a spawned worker cannot import tests.conftest
    assembled = assemble(load_datasets(resolved, data_root=EXAMPLE_DATA, run_id="run-x"), resolved, run_id="run-x")
    return SharedRunData(assembled=assembled, config=resolved.config, config_sha256="example", run_id="run-x")


def test_build_task_slices_rules_and_builds_from_the_shared_data_and_reports_its_environment() -> None:
    shared = _shared()
    output = build_task(shared, PortfolioId("P1"))
    assert isinstance(output.outcome, BuildResult)
    assert output.outcome.spec.security_ids == ("A", "B", "C")
    assert output.outcome.solve_order == 0
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
