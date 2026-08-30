"""Tier 2: the derived schedule reproduces the line, chaining works, failures skip only what depended on them (or everything under fail_fast), and nothing partial is published."""

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from portfolio_optimizer.domain.results import PortfolioFailure, PortfolioResult
from portfolio_optimizer.engine.backends import Backend
from portfolio_optimizer.engine.environment import GitInfo
from portfolio_optimizer.engine.runner import EXIT_INFRASTRUCTURE, EXIT_OK, EXIT_PORTFOLIO_FAILED, InputRejectedError, RunReport, run
from portfolio_optimizer.settings import ExecutionSettings
from tests.conftest import BUY_ONLY_OBJECTIVE, EXAMPLE_DATA, NO_CHAIN_CONSTRAINTS, execution_on, half_cash_book, io_context, resolved_example_real, sell_book
from tests.engine.test_backends import LazyBackend

GIT = GitInfo(sha="0123456789abcdef", dirty=False)


def execute(
    tmp_path: Path,
    scheduler_address: str,
    *,
    max_workers: int = 2,
    on_error: str = "fail_fast",
    dependencies: str = "overlap",
    data_root: Path = EXAMPLE_DATA,
    run_id: str = "run-test",
    sink: str = "orders_to_parquet",
    **overrides: object,
) -> RunReport:
    resolved = resolved_example_real(execution={"on_error": on_error, "dependencies": dependencies}, sink=sink, **overrides)
    execution = execution_on(scheduler_address, max_workers=max_workers)
    return run(
        resolved, io_context(tmp_path / run_id, data_root=data_root, run_id=run_id), execution=execution, git=GIT, config_path="configs/example_run.json", settings={"data_root": str(data_root)}
    )


def test_the_run_reproduces_the_hand_checked_orders(tmp_path: Path, scheduler_address: str) -> None:
    report = execute(tmp_path, scheduler_address)
    assert report.exit_code == EXIT_OK
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioResult)
    assert isinstance(p2, PortfolioResult)
    assert p1.orders[["security_id", "side", "quantity"]].to_dict("records") == [
        {"security_id": "A", "side": "SELL", "quantity": 1250},
        {"security_id": "B", "side": "SELL", "quantity": 2500},
        {"security_id": "C", "side": "BUY", "quantity": 25000},
    ]
    assert len(p2.orders) == 0, "P2 wants C too, but P1 spent C's ADV budget, and A and B already sit symmetrically against the target"
    assert p2.chain_state.traded_shares.tolist() == [0.0, 0.0, 25000.0], "only P1's buys reach P2; its sells of A and B do not"
    assert p2.chain_state.predecessors == ("P1",)
    run_dir = tmp_path / "run-test" / "run-test"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "orders" / "orders.parquet").exists()
    assert {path.name for path in (run_dir / "problem_specs").iterdir()} == {"P1.npz", "P2.npz"}
    assert report.manifest.exit_code == EXIT_OK
    assert [p.status for p in report.manifest.portfolios] == ["solved", "solved"]
    schedule = report.manifest.schedule
    assert schedule is not None and (schedule.coupling, schedule.edges, schedule.components, schedule.critical_path) == ("overlap", 1, 1, 2)


@pytest.mark.parametrize("max_workers", [2, 1])
def test_every_earlier_portfolio_as_a_predecessor_matches_the_overlap_schedule(tmp_path: Path, scheduler_address: str, max_workers: int) -> None:
    line = execute(tmp_path, scheduler_address, dependencies="all", run_id="line")
    overlap = execute(tmp_path, scheduler_address, max_workers=max_workers, run_id="overlap")
    assert overlap.exit_code == EXIT_OK
    for left, right in zip(line.solved, overlap.solved, strict=True):
        assert left.spec.content_hash() == right.spec.content_hash()
        assert left.chain_state.content_hash() == right.chain_state.content_hash()
        assert_frame_equal(left.orders.drop(columns=["run_id"]), right.orders.drop(columns=["run_id"]))
    assert [p.orders for p in line.manifest.portfolios] == [p.orders for p in overlap.manifest.portfolios]


def test_a_buy_only_run_on_the_cluster_reproduces_the_hand_checked_buys(tmp_path: Path, scheduler_address: str) -> None:
    report = execute(tmp_path, scheduler_address, data_root=half_cash_book(tmp_path), sides="buy", objective=BUY_ONLY_OBJECTIVE)
    assert report.exit_code == EXIT_OK
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioResult) and isinstance(p2, PortfolioResult)
    assert p1.orders[["security_id", "side", "quantity"]].to_dict("records") == [
        {"security_id": "A", "side": "BUY", "quantity": 1250},
        {"security_id": "B", "side": "BUY", "quantity": 2500},
        {"security_id": "C", "side": "BUY", "quantity": 25000},
    ]
    assert p2.orders[["security_id", "side", "quantity"]].to_dict("records") == [{"security_id": "A", "side": "BUY", "quantity": 2500}, {"security_id": "B", "side": "BUY", "quantity": 5000}]
    assert p2.chain_state.traded_shares.tolist() == [1250.0, 2500.0, 25000.0]
    assert [p.status for p in report.manifest.portfolios] == ["solved", "solved"]


def test_a_sell_only_run_on_the_cluster_reproduces_the_hand_checked_sells(tmp_path: Path, scheduler_address: str) -> None:
    report = execute(tmp_path, scheduler_address, data_root=sell_book(tmp_path), sides="sell")
    assert report.exit_code == EXIT_OK
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioResult) and isinstance(p2, PortfolioResult)
    assert p1.orders[["security_id", "side", "quantity"]].to_dict("records") == [{"security_id": "A", "side": "SELL", "quantity": 1000}, {"security_id": "B", "side": "SELL", "quantity": 3333}]
    assert p2.orders[["security_id", "side", "quantity"]].to_dict("records") == [{"security_id": "B", "side": "SELL", "quantity": 3333}]
    assert p2.chain_state.traded_shares.tolist() == [1000.0, 3333.0, 0.0]
    assert [p.status for p in report.manifest.portfolios] == ["solved", "solved"]


def test_nothing_reading_the_chain_frees_every_portfolio(tmp_path: Path, scheduler_address: str) -> None:
    report = execute(tmp_path, scheduler_address, constraints=NO_CHAIN_CONSTRAINTS)
    assert report.exit_code == EXIT_OK
    assert report.manifest.schedule is not None and report.manifest.schedule.coupling == "none" and report.manifest.schedule.edges == 0
    p2 = report.solved[1]
    assert len(p2.orders) > 0, "without the chained ADV cap P2 is free to buy C"


def _data_with(tmp_path: Path, **files: str) -> Path:
    root = tmp_path / "data"
    shutil.copytree(EXAMPLE_DATA, root)
    for name, content in files.items():
        (root / name).write_text(content)
    return root


def _constraints(**per_portfolio: Mapping[str, object]) -> str:
    base = json.loads((EXAMPLE_DATA / "constraints.json").read_text())
    for portfolio_id, overrides in per_portfolio.items():
        base[portfolio_id] = {**base[portfolio_id], **overrides}
    return json.dumps(base)


def test_fail_fast_skips_every_lower_priority_portfolio_and_publishes_nothing(tmp_path: Path, scheduler_address: str) -> None:
    data_root = _data_with(tmp_path, **{"constraints.json": _constraints(P1={"max_weight": "0.3"})})
    report = execute(tmp_path, scheduler_address, data_root=data_root)
    assert report.exit_code == EXIT_PORTFOLIO_FAILED
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioFailure)
    assert p1.stage == "solve"
    assert p1.error_type == "InfeasibleError"
    assert "upper bounds sum to 0.900000" in p1.message
    assert isinstance(p2, PortfolioFailure)
    assert p2.stage == "skipped"
    assert not (tmp_path / "run-test" / "run-test" / "orders").exists()
    assert (tmp_path / "run-test" / "run-test" / "manifest.json").exists()
    assert [p.failure_stage for p in report.manifest.portfolios] == ["solve", "skipped"]


def test_continue_isolates_a_failure_nothing_depended_on(tmp_path: Path, scheduler_address: str) -> None:
    data_root = _data_with(tmp_path, **{"constraints.json": _constraints(P1={"max_weight": "0.3"})})
    report = execute(tmp_path, scheduler_address, on_error="continue", data_root=data_root, constraints=NO_CHAIN_CONSTRAINTS)
    assert report.exit_code == EXIT_PORTFOLIO_FAILED
    assert [type(o).__name__ for o in report.outcomes] == ["PortfolioFailure", "PortfolioResult"]
    assert (tmp_path / "run-test" / "run-test" / "orders" / "orders.parquet").exists()
    assert report.manifest.portfolios[1].status == "solved"


def test_continue_skips_the_portfolios_that_depended_on_the_failure_and_names_it(tmp_path: Path, scheduler_address: str) -> None:
    data_root = _data_with(tmp_path, **{"constraints.json": _constraints(P1={"max_weight": "0.3"})})
    report = execute(tmp_path, scheduler_address, on_error="continue", data_root=data_root)
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioFailure) and p1.stage == "solve"
    assert isinstance(p2, PortfolioFailure) and p2.stage == "skipped"
    assert "predecessor 'P1' failed at stage 'solve'" in p2.message
    assert report.manifest.portfolios[1].predecessors == 1


def test_a_portfolio_holding_a_name_the_build_cannot_place_fails_at_build(tmp_path: Path, scheduler_address: str) -> None:
    holdings = (EXAMPLE_DATA / "holdings.csv").read_text().replace("P2,B,10000,50", "P2,Z,10000,50")
    data_root = _data_with(tmp_path, **{"holdings.csv": holdings})
    report = execute(tmp_path, scheduler_address, on_error="continue", data_root=data_root, constraints=NO_CHAIN_CONSTRAINTS)
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioResult)
    assert isinstance(p2, PortfolioFailure)
    assert p2.stage == "build"
    assert "held securities missing from universe ['Z']" in p2.message


def test_a_failed_build_is_treated_as_overlapping_everything_after_it(tmp_path: Path, scheduler_address: str) -> None:
    holdings = (EXAMPLE_DATA / "holdings.csv").read_text().replace("P1,B,10000,50", "P1,Z,10000,50")
    data_root = _data_with(tmp_path, **{"holdings.csv": holdings})
    report = execute(tmp_path, scheduler_address, on_error="continue", data_root=data_root)
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioFailure) and p1.stage == "build"
    assert isinstance(p2, PortfolioFailure) and p2.stage == "skipped" and "predecessor 'P1' failed at stage 'build'" in p2.message


def test_a_portfolio_whose_bundle_is_inconsistent_fails_at_slice(tmp_path: Path, scheduler_address: str) -> None:
    targets = (EXAMPLE_DATA / "targets.csv").read_text().replace("B1,C,", "B1,Z,")
    data_root = _data_with(tmp_path, **{"targets.csv": targets})
    report = execute(tmp_path, scheduler_address, on_error="continue", data_root=data_root, constraints=NO_CHAIN_CONSTRAINTS)
    assert [outcome.stage for outcome in report.outcomes if isinstance(outcome, PortfolioFailure)] == ["slice", "slice"]
    assert "target securities in neither holdings nor universe ['Z']" in report.outcomes[0].message  # ty: ignore[unresolved-attribute]  # both outcomes are failures, asserted above


def test_sink_failure_is_infrastructure_and_the_manifest_still_records_it(tmp_path: Path, scheduler_address: str) -> None:
    run_dir = tmp_path / "run-test" / "run-test"
    run_dir.mkdir(parents=True)
    (run_dir / "orders").write_text("not a directory")  # the parquet sink cannot create its output directory
    report = execute(tmp_path, scheduler_address)
    assert report.exit_code == EXIT_INFRASTRUCTURE
    assert report.manifest.exit_code == EXIT_INFRASTRUCTURE
    assert [type(outcome).__name__ for outcome in report.outcomes] == ["PortfolioResult", "PortfolioResult"], "the portfolios solved; only publishing failed"
    sink_record = report.manifest.portfolios[-1]
    assert sink_record.portfolio_id == "*"
    assert sink_record.failure_stage == "sink"
    assert sink_record.error is not None
    assert "orders" in sink_record.error
    assert (run_dir / "manifest.json").exists()


def test_inputs_that_cannot_be_assembled_reject_the_run_before_solving(tmp_path: Path, scheduler_address: str) -> None:
    data_root = _data_with(tmp_path, **{"constraints.json": json.dumps({"P1": json.loads(_constraints())["P1"]})})
    with pytest.raises(InputRejectedError, match="constraints missing for portfolios \\['P2'\\]"):
        execute(tmp_path, scheduler_address, data_root=data_root)


def test_manifest_records_provenance_for_every_stage(tmp_path: Path, scheduler_address: str) -> None:
    manifest = execute(tmp_path, scheduler_address).manifest
    assert manifest.git_sha == GIT.sha
    assert manifest.config.sha256 == "example"
    assert {d.name for d in manifest.datasets} == {"portfolios", "holdings", "universe", "details", "constraints", "targets", "prices"}
    p1 = manifest.portfolios[0]
    assert [r.qualname for r in p1.rules] == ["portfolio_optimizer.rules:restrict_low_liquidity", "portfolio_optimizer.rules:add_zero_alpha"]
    assert p1.solve is not None
    assert p1.solve.status == "optimal"
    assert p1.check is not None
    assert p1.check.passed
    assert p1.drift is not None
    assert p1.drift.max_weight_error <= 1e-8
    assert p1.orders is not None
    assert p1.orders.count == 3
    assert p1.orders.gross_notional == "500000"
    assert (p1.solve_order, p1.predecessors) == ("0", 0)
    assert len(manifest.manifest_sha256) == 64
    assert {a.path.rsplit("/", 1)[-1] for a in manifest.artifacts} >= {"P1.npz", "P2.npz", "orders.parquet"}
    assert manifest.versions.packages == {}  # every step came from the template modules; git_sha covers them


def test_manifest_records_the_package_behind_every_external_step(tmp_path: Path) -> None:
    def in_process(execution: ExecutionSettings, *, run_id: str) -> Backend:  # a spawned worker cannot import tests.conftest, so this run stays in one process
        del execution, run_id
        return LazyBackend()

    resolved = resolved_example_real(sink="tests.conftest:noop_sink")
    report = run(resolved, io_context(tmp_path / "run-test", run_id="run-test"), execution=execution_on("tcp://fake:8786"), git=GIT, config_path="c.json", settings={}, backend_factory=in_process)
    assert report.exit_code == EXIT_OK
    assert report.manifest.versions.packages == {"tests": "unknown"}  # tests.conftest is importable but no installed distribution provides it


def test_two_runs_over_the_same_inputs_are_identical_except_for_identity(tmp_path: Path, scheduler_address: str) -> None:
    first = execute(tmp_path, scheduler_address, run_id="one").manifest
    second = execute(tmp_path, scheduler_address, dependencies="all", run_id="two").manifest
    strip = ("problem_spec_sha256", "chain_inputs_sha256", "orders", "rules", "solve_order")
    for left, right in zip(first.portfolios, second.portfolios, strict=True):
        for field in strip:
            assert getattr(left, field) == getattr(right, field), field
    assert [d.content_sha256 for d in first.datasets] == [d.content_sha256 for d in second.datasets]
