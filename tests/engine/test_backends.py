"""Tier 1/2: the backend seam — workers probed before data is shared, builds first, solves along the schedule with dependencies, fail-fast cancellation, dead and stale workers, and a cluster that never comes up."""

import json
from pathlib import Path

from pandas.testing import assert_frame_equal

from portfolio_optimizer.domain.results import PortfolioFailure, PortfolioResult
from portfolio_optimizer.engine.backends import TaskOutput
from portfolio_optimizer.engine.environment import environment_for
from portfolio_optimizer.engine.runner import EXIT_INFRASTRUCTURE, EXIT_OK, EXIT_PORTFOLIO_FAILED, RunReport
from tests.conftest import BUY_ONLY_OBJECTIVE, NO_CHAIN_CONSTRAINTS, resolved_example_real
from tests.engine.fakes import LazyBackend, factory_for
from tests.engine.support import HALF_CASH_ORDERS_P1, HALF_CASH_ORDERS_P2, SELL_BOOK_ORDERS_P1, SELL_BOOK_ORDERS_P2, constraints_json, example_book, execute, half_cash_book, sell_book


def test_the_runner_submits_every_build_first_then_solves_along_the_schedule(tmp_path: Path) -> None:
    backend = LazyBackend()
    report = execute(tmp_path, backend_factory=factory_for(backend), max_workers=1)
    assert report.exit_code == EXIT_OK
    assert backend.started and backend.closed
    assert backend.scaled_to == 1
    assert backend.shared_count == 1, "shared data is delivered once per run, not once per task"
    assert backend.submitted == ["run-test/P1/build", "run-test/P1/summary", "run-test/P2/build", "run-test/P2/summary", "run-test/P1/solve", "run-test/P1/contribution", "run-test/P2/solve"], (
        "P2 is the tail of the chain, so nothing ever asks it to contribute"
    )
    assert backend.priorities["run-test/P1/solve"] > backend.priorities["run-test/P2/solve"] > backend.priorities["run-test/P1/build"], "solves run in solve order, and any solve beats any build"
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


def test_a_solve_is_submitted_before_the_builds_behind_it_are_read(tmp_path: Path) -> None:
    backend = LazyBackend()
    assert execute(tmp_path, backend_factory=factory_for(backend)).exit_code == EXIT_OK
    assert backend.trace.index("submit:run-test/P1/solve") < backend.trace.index("run:run-test/P2/build"), (
        "P1 has no predecessor, so its solve goes in as soon as its own build reports; waiting for P2 would idle the cluster for a build wave"
    )


def test_a_portfolio_the_load_stage_rejected_is_never_built_and_does_not_hold_up_the_others(tmp_path: Path) -> None:
    backend = LazyBackend()
    data_root = example_book(tmp_path, **{"constraints.json": json.dumps({"P2": json.loads(constraints_json())["P2"]})})
    report = execute(tmp_path, backend_factory=factory_for(backend), data_root=data_root, on_error="continue")
    assert report.exit_code == EXIT_PORTFOLIO_FAILED
    assert not any(key.startswith("run-test/P1/") for key in backend.submitted), "P1 has no inputs, so nothing is submitted for it at all"
    assert "run-test/P2/solve" in backend.submitted, "P2 has everything it needs and is solved"
    first, second = report.outcomes
    assert isinstance(first, PortfolioFailure) and (first.stage, first.error_type) == ("load", "MissingInput")
    assert isinstance(second, PortfolioResult) and second.portfolio_id == "P2"
    schedule = report.manifest.schedule
    assert schedule is not None and (schedule.edges, schedule.components) == (0, 2), "a portfolio that never entered the run traded nothing, so it couples to nobody and P2 waits for no one"


def test_a_configured_solve_order_step_reorders_the_run_and_is_read_before_any_solve(tmp_path: Path) -> None:
    backend = LazyBackend()
    report = execute(tmp_path, backend_factory=factory_for(backend), solve_order="tests.steps:last_portfolio_id_first")
    assert report.exit_code == EXIT_OK
    assert [outcome.portfolio_id for outcome in report.outcomes] == ["P2", "P1"], "the step replaces the portfolios frame's column, and P2 now has first pick"
    assert [record.solve_order for record in report.manifest.portfolios] == ["-2", "-1"]
    assert backend.trace.index("run:run-test/P1/build") < backend.trace.index("submit:run-test/P2/solve"), (
        "the step reads the ruled bundle, so the order is itself a build output and no solve can be placed until every build has reported"
    )


def test_an_uncoupled_run_submits_every_solve_behind_its_own_build_and_reads_no_summary_first(tmp_path: Path) -> None:
    backend = LazyBackend()
    report = execute(tmp_path, backend_factory=factory_for(backend), constraints=NO_CHAIN_CONSTRAINTS)
    assert report.exit_code == EXIT_OK
    assert backend.submitted == ["run-test/P1/build", "run-test/P1/summary", "run-test/P2/build", "run-test/P2/summary", "run-test/P1/solve", "run-test/P2/solve"], (
        "nothing reads the chain, so no solve waits and no portfolio is asked to contribute"
    )
    assert backend.trace.index("submit:run-test/P2/solve") < backend.trace.index("run:run-test/P1/build"), "no build is read before every solve is in flight"
    assert report.manifest.schedule is not None and (report.manifest.schedule.coupling, report.manifest.schedule.edges) == ("none", 0)
    assert [record.solve_order for record in report.manifest.portfolios] == ["0", "1"], "the order is still read back, for the manifest and the order outcomes are classified in"


def test_a_pure_function_solve_step_runs_the_whole_pipeline_and_is_verified(tmp_path: Path) -> None:
    report = execute(tmp_path, backend_factory=factory_for(LazyBackend()), solve="tests.steps:hold_still")
    assert report.exit_code == EXIT_OK
    for outcome in report.outcomes:
        assert isinstance(outcome, PortfolioResult)
        assert len(outcome.orders) == 0 and outcome.solution.objective is None and outcome.report.passed
    record = report.manifest.portfolios[0]
    assert record.solve is not None and record.solve.solver == "tests.steps:hold_still" and record.solve.objective_value is None
    assert record.check is not None and record.check.objective_passed


def test_a_buy_only_run_reproduces_the_hand_checked_buys_and_couples_through_them(tmp_path: Path) -> None:
    report = execute(tmp_path, backend_factory=factory_for(LazyBackend()), data_root=half_cash_book(tmp_path), sides="buy", objective=BUY_ONLY_OBJECTIVE)
    assert report.exit_code == EXIT_OK, [outcome for outcome in report.outcomes if isinstance(outcome, PortfolioFailure)]
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioResult) and isinstance(p2, PortfolioResult)
    assert p1.orders[["security_id", "side", "quantity"]].to_dict("records") == HALF_CASH_ORDERS_P1
    assert p2.orders[["security_id", "side", "quantity"]].to_dict("records") == HALF_CASH_ORDERS_P2
    assert p2.chain_state.traded_shares.tolist() == [1250.0, 2500.0, 25000.0] and p2.chain_state.predecessors == ("P1",), (
        "every buy P1 made reaches P2: under buy-only the chain carries the whole trade"
    )
    for outcome in (p1, p2):
        assert outcome.report.passed and (outcome.solution.w >= outcome.spec.w0 - 1e-9).all() and outcome.solution.sell.tolist() == [0.0, 0.0, 0.0]
    assert report.manifest.config.resolved["sides"] == "buy"
    assert [p.status for p in report.manifest.portfolios] == ["solved", "solved"]


def test_a_sell_only_run_reproduces_the_hand_checked_sells_and_couples_through_them(tmp_path: Path) -> None:
    report = execute(tmp_path, backend_factory=factory_for(LazyBackend()), data_root=sell_book(tmp_path), sides="sell")
    assert report.exit_code == EXIT_OK, [outcome for outcome in report.outcomes if isinstance(outcome, PortfolioFailure)]
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioResult) and isinstance(p2, PortfolioResult)
    assert p1.orders[["security_id", "side", "quantity"]].to_dict("records") == SELL_BOOK_ORDERS_P1
    assert p2.orders[["security_id", "side", "quantity"]].to_dict("records") == SELL_BOOK_ORDERS_P2, "P1's sells of A used up its ADV budget"
    assert p2.chain_state.traded_shares.tolist() == [1000.0, 3333.0, 0.0] and p2.chain_state.predecessors == ("P1",)
    for outcome in (p1, p2):
        assert outcome.report.passed and (outcome.solution.w <= outcome.spec.w0 + 1e-9).all() and outcome.solution.buy.tolist() == [0.0, 0.0, 0.0]
    assert report.manifest.config.resolved["sides"] == "sell"
    assert [p.status for p in report.manifest.portfolios] == ["solved", "solved"]


def test_nothing_reading_the_chain_means_no_portfolio_waits(tmp_path: Path) -> None:
    report = execute(tmp_path, backend_factory=factory_for(LazyBackend()), constraints=NO_CHAIN_CONSTRAINTS)
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
    report = execute(tmp_path, backend_factory=factory_for(backend))
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
    report = execute(tmp_path, backend_factory=factory_for(backend))
    assert report.exit_code == EXIT_INFRASTRUCTURE
    assert backend.shared_count == 0 and backend.submitted == []
    records = {record.portfolio_id: record for record in report.manifest.portfolios}
    assert records["*"].error is not None and "worker pod-7 (fake://worker-1) runs a different environment: image_digest: None here, 'sha256:old' there" in records["*"].error


def test_a_worker_whose_environment_differs_fails_its_portfolios_at_stage_worker(tmp_path: Path) -> None:
    def stale(output: TaskOutput[object]) -> TaskOutput[object]:
        assert output.environment is not None
        return TaskOutput(outcome=output.outcome, environment=output.environment.model_copy(update={"git_sha": "stale"}), host="pod-7")

    report = execute(tmp_path, backend_factory=factory_for(LazyBackend(tamper=stale)), on_error="continue")
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
    report = execute(tmp_path, backend_factory=factory_for(backend))
    assert report.exit_code == EXIT_PORTFOLIO_FAILED
    first, second = report.outcomes
    assert isinstance(first, PortfolioFailure) and (first.stage, first.error_type) == ("worker", "BrokenExecutor")
    assert isinstance(second, PortfolioFailure) and second.stage == "skipped"
    assert backend.cancelled == ["run-test/P2/solve"]


def test_under_continue_a_dead_predecessor_skips_only_the_portfolios_that_depended_on_it(tmp_path: Path) -> None:
    backend = LazyBackend(dead_keys=frozenset({"run-test/P1/solve"}))
    report = execute(tmp_path, backend_factory=factory_for(backend), on_error="continue")
    first, second = report.outcomes
    assert isinstance(first, PortfolioFailure) and first.stage == "worker"
    assert isinstance(second, PortfolioFailure) and second.stage == "skipped" and "predecessor 'P1' failed at stage 'worker'" in second.message
    assert backend.cancelled == []


def test_a_dead_worker_under_a_build_stops_a_fail_fast_run_before_any_solve(tmp_path: Path) -> None:
    backend = LazyBackend(dead_keys=frozenset({"run-test/P1/build"}))
    report = execute(tmp_path, backend_factory=factory_for(backend))
    first, second = report.outcomes
    assert isinstance(first, PortfolioFailure) and (first.stage, first.error_type) == ("worker", "BrokenExecutor")
    assert isinstance(second, PortfolioFailure) and second.stage == "skipped"
    assert not any(key.endswith("/solve") for key in backend.submitted)
    assert report.manifest.schedule is not None and report.manifest.schedule.edges == 1, "the builds behind the failure are still read, so the manifest records the whole graph"


def test_a_cluster_that_never_comes_up_is_infrastructure_and_still_leaves_a_manifest(tmp_path: Path) -> None:
    backend = LazyBackend(fail_ready=True)
    report = execute(tmp_path, backend_factory=factory_for(backend))
    assert report.exit_code == EXIT_INFRASTRUCTURE
    assert backend.closed
    assert [(o.stage, o.error_type) for o in report.outcomes if isinstance(o, PortfolioFailure)] == [("skipped", "ClusterUnavailable")] * 2
    records = {record.portfolio_id: record for record in report.manifest.portfolios}
    assert records["*"].failure_stage == "cluster" and records["*"].error is not None and "no worker within the timeout" in records["*"].error
    assert report.manifest.cluster is not None and report.manifest.cluster.first_worker_ready_at is None
    assert report.manifest.schedule is None
    assert report.manifest_path.exists()


def test_every_earlier_portfolio_as_a_predecessor_gives_the_same_answer_as_overlap(tmp_path: Path) -> None:
    overlap = execute(tmp_path, backend_factory=factory_for(LazyBackend()), run_id="overlap")
    line = execute(tmp_path, backend_factory=factory_for(LazyBackend()), dependencies="all", run_id="line")
    assert overlap.exit_code == line.exit_code == EXIT_OK
    for left, right in zip(overlap.solved, line.solved, strict=True):
        assert_frame_equal(left.orders.drop(columns=["run_id"]), right.orders.drop(columns=["run_id"]))
        assert left.chain_state.content_hash() == right.chain_state.content_hash()
    assert report_kinds(overlap) == report_kinds(line) == [PortfolioResult, PortfolioResult]
    assert line.manifest.schedule is not None and line.manifest.schedule.coupling == "all"


def report_kinds(report: RunReport) -> list[type]:
    return [type(outcome) for outcome in report.outcomes]
