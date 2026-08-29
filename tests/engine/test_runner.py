"""Tier 2: the three execution modes agree, chaining works, failures are isolated or stop the run, and nothing partial is published."""

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from portfolio_optimizer.domain.results import PortfolioFailure, PortfolioResult
from portfolio_optimizer.engine.environment import GitInfo
from portfolio_optimizer.engine.runner import EXIT_INFRASTRUCTURE, EXIT_OK, EXIT_PORTFOLIO_FAILED, InputRejectedError, RunReport, run
from tests.conftest import EXAMPLE_DATA, execution_on, io_context, resolved_example_real

GIT = GitInfo(sha="0123456789abcdef", dirty=False)
NO_CHAIN_CONSTRAINTS = ["trade_balance", "long_only", "max_weight", "cash_bounds", "turnover_cap", "sector_bounds"]


def execute(
    tmp_path: Path,
    scheduler_address: str,
    *,
    mode: str,
    max_workers: int = 2,
    on_error: str = "fail_fast",
    data_root: Path = EXAMPLE_DATA,
    run_id: str = "run-test",
    sink: str = "orders_to_parquet",
    **overrides: object,
) -> RunReport:
    resolved = resolved_example_real(execution={"mode": mode, "on_error": on_error}, sink=sink, **overrides)
    execution = execution_on(scheduler_address, max_workers=max_workers)
    return run(
        resolved, io_context(tmp_path / run_id, data_root=data_root, run_id=run_id), execution=execution, git=GIT, config_path="configs/example_run.json", settings={"data_root": str(data_root)}
    )


def test_sequential_run_reproduces_the_hand_checked_orders(tmp_path: Path, scheduler_address: str) -> None:
    report = execute(tmp_path, scheduler_address, mode="sequential")
    assert report.exit_code == EXIT_OK
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioResult)
    assert isinstance(p2, PortfolioResult)
    assert p1.orders[["security_id", "side", "quantity"]].to_dict("records") == [
        {"security_id": "A", "side": "SELL", "quantity": 1250},
        {"security_id": "B", "side": "SELL", "quantity": 2500},
        {"security_id": "C", "side": "BUY", "quantity": 25000},
    ]
    assert len(p2.orders) == 0, "P2's ADV budget for C is spent by P1, so it must not trade"
    assert p2.chain_state.cumulative_shares.tolist() == [1250.0, 2500.0, 25000.0]
    run_dir = tmp_path / "run-test" / "run-test"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "orders" / "orders.parquet").exists()
    assert {path.name for path in (run_dir / "problem_specs").iterdir()} == {"P1.npz", "P2.npz"}
    assert report.manifest.exit_code == EXIT_OK
    assert [p.status for p in report.manifest.portfolios] == ["solved", "solved"]


@pytest.mark.parametrize("max_workers", [2, 1])
def test_parallel_build_matches_the_sequential_run(tmp_path: Path, scheduler_address: str, max_workers: int) -> None:
    sequential = execute(tmp_path, scheduler_address, mode="sequential", run_id="seq")
    parallel_build = execute(tmp_path, scheduler_address, mode="parallel_build_sequential_solve", max_workers=max_workers, run_id="pbss")
    assert parallel_build.exit_code == EXIT_OK
    for left, right in zip(sequential.solved, parallel_build.solved, strict=True):
        assert left.spec.content_hash() == right.spec.content_hash()
        assert_frame_equal(left.orders.drop(columns=["run_id"]), right.orders.drop(columns=["run_id"]))
    assert [p.orders for p in sequential.manifest.portfolios] == [p.orders for p in parallel_build.manifest.portfolios]


def test_fully_parallel_mode_matches_sequential_when_nothing_chains(tmp_path: Path, scheduler_address: str) -> None:
    sequential = execute(tmp_path, scheduler_address, mode="sequential", run_id="seq", constraints=NO_CHAIN_CONSTRAINTS)
    parallel = execute(tmp_path, scheduler_address, mode="parallel", run_id="par", constraints=NO_CHAIN_CONSTRAINTS)
    assert parallel.exit_code == EXIT_OK
    for left, right in zip(sequential.solved, parallel.solved, strict=True):
        assert_frame_equal(left.orders.drop(columns=["run_id"]), right.orders.drop(columns=["run_id"]))
    p2 = parallel.solved[1]
    assert len(p2.orders) > 0, "without the chained ADV cap P2 is free to trade"


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


def test_fail_fast_skips_later_portfolios_and_publishes_nothing(tmp_path: Path, scheduler_address: str) -> None:
    data_root = _data_with(tmp_path, **{"constraints.json": _constraints(P1={"max_weight": "0.3"})})
    report = execute(tmp_path, scheduler_address, mode="sequential", data_root=data_root)
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


def test_continue_isolates_the_failure_and_publishes_the_rest(tmp_path: Path, scheduler_address: str) -> None:
    data_root = _data_with(tmp_path, **{"constraints.json": _constraints(P1={"max_weight": "0.3"})})
    report = execute(tmp_path, scheduler_address, mode="parallel_build_sequential_solve", on_error="continue", data_root=data_root, constraints=NO_CHAIN_CONSTRAINTS)
    assert report.exit_code == EXIT_PORTFOLIO_FAILED
    assert [type(o).__name__ for o in report.outcomes] == ["PortfolioFailure", "PortfolioResult"]
    assert (tmp_path / "run-test" / "run-test" / "orders" / "orders.parquet").exists()
    assert report.manifest.portfolios[1].status == "solved"


def test_a_portfolio_holding_a_name_the_build_cannot_place_fails_at_build(tmp_path: Path, scheduler_address: str) -> None:
    holdings = (EXAMPLE_DATA / "holdings.csv").read_text().replace("P2,C,100000,10", "P2,Z,100000,10")
    data_root = _data_with(tmp_path, **{"holdings.csv": holdings})
    report = execute(tmp_path, scheduler_address, mode="sequential", on_error="continue", data_root=data_root, constraints=NO_CHAIN_CONSTRAINTS)
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioResult)
    assert isinstance(p2, PortfolioFailure)
    assert p2.stage == "build"
    assert "held securities missing from universe ['Z']" in p2.message


def test_a_portfolio_whose_bundle_is_inconsistent_fails_at_slice(tmp_path: Path, scheduler_address: str) -> None:
    targets = (EXAMPLE_DATA / "targets.csv").read_text().replace("B1,C,", "B1,Z,")
    data_root = _data_with(tmp_path, **{"targets.csv": targets})
    report = execute(tmp_path, scheduler_address, mode="sequential", on_error="continue", data_root=data_root, constraints=NO_CHAIN_CONSTRAINTS)
    assert [outcome.stage for outcome in report.outcomes if isinstance(outcome, PortfolioFailure)] == ["slice", "slice"]
    assert "target securities in neither holdings nor universe ['Z']" in report.outcomes[0].message  # ty: ignore[unresolved-attribute]  # both outcomes are failures, asserted above


def test_sink_failure_is_infrastructure_and_the_manifest_still_records_it(tmp_path: Path, scheduler_address: str) -> None:
    report = execute(tmp_path, scheduler_address, mode="sequential", sink="tests.conftest:failing_sink")
    assert report.exit_code == EXIT_INFRASTRUCTURE
    assert report.manifest.exit_code == EXIT_INFRASTRUCTURE
    sink_record = report.manifest.portfolios[-1]
    assert sink_record.portfolio_id == "*"
    assert sink_record.failure_stage == "sink"
    assert sink_record.error is not None
    assert "trading gateway unreachable" in sink_record.error
    assert (tmp_path / "run-test" / "run-test" / "manifest.json").exists()


def test_inputs_that_cannot_be_assembled_reject_the_run_before_solving(tmp_path: Path, scheduler_address: str) -> None:
    data_root = _data_with(tmp_path, **{"constraints.json": json.dumps({"P1": json.loads(_constraints())["P1"]})})
    with pytest.raises(InputRejectedError, match="constraints missing for portfolios \\['P2'\\]"):
        execute(tmp_path, scheduler_address, mode="sequential", data_root=data_root)


def test_manifest_records_provenance_for_every_stage(tmp_path: Path, scheduler_address: str) -> None:
    manifest = execute(tmp_path, scheduler_address, mode="sequential").manifest
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
    assert len(manifest.manifest_sha256) == 64
    assert {a.path.rsplit("/", 1)[-1] for a in manifest.artifacts} >= {"P1.npz", "P2.npz", "orders.parquet"}
    assert manifest.versions.packages == {}  # every step came from the template modules; git_sha covers them


def test_manifest_records_the_package_behind_every_external_step(tmp_path: Path, scheduler_address: str) -> None:
    manifest = execute(tmp_path, scheduler_address, mode="sequential", sink="tests.conftest:noop_sink").manifest
    assert manifest.versions.packages == {"tests": "unknown"}  # tests.conftest is importable but no installed distribution provides it


def test_two_runs_over_the_same_inputs_are_identical_except_for_identity(tmp_path: Path, scheduler_address: str) -> None:
    first = execute(tmp_path, scheduler_address, mode="sequential", run_id="one").manifest
    second = execute(tmp_path, scheduler_address, mode="parallel_build_sequential_solve", run_id="two").manifest
    strip = ("problem_spec_sha256", "chain_inputs_sha256", "orders", "rules")
    for left, right in zip(first.portfolios, second.portfolios, strict=True):
        for field in strip:
            assert getattr(left, field) == getattr(right, field), field
    assert [d.content_sha256 for d in first.datasets] == [d.content_sha256 for d in second.datasets]
