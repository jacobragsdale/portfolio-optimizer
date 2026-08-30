"""Tier 2: on the real cluster — the golden orders, the line equals the overlap schedule, failures skip only what depended on them (or everything under fail_fast), and nothing partial is published."""

from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from portfolio_optimizer.domain.results import PortfolioFailure, PortfolioResult
from portfolio_optimizer.engine.runner import EXIT_INFRASTRUCTURE, EXIT_OK, EXIT_PORTFOLIO_FAILED, InputRejectedError
from tests.conftest import EXAMPLE_DATA, resolved_example_real
from tests.engine.fakes import LazyBackend, factory_for
from tests.engine.support import EXAMPLE_ORDERS_P1, GIT, details_csv, example_book, execute, no_details_csv

CAPPED_P1 = details_csv("P1", max_weight="0.25")
"""P1's cap at a quarter: three names cannot hold the 0.8 of NAV its cash bounds oblige it to invest, so its solve is infeasible."""


def test_the_run_reproduces_the_hand_checked_orders(tmp_path: Path, scheduler_address: str) -> None:
    report = execute(tmp_path, scheduler_address=scheduler_address)
    assert report.exit_code == EXIT_OK
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioResult)
    assert isinstance(p2, PortfolioResult)
    assert p1.orders[["security_id", "side", "quantity"]].to_dict("records") == EXAMPLE_ORDERS_P1
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


def test_every_earlier_portfolio_as_a_predecessor_matches_the_overlap_schedule(tmp_path: Path, scheduler_address: str) -> None:
    line = execute(tmp_path, scheduler_address=scheduler_address, dependencies="all", run_id="line")
    overlap = execute(tmp_path, scheduler_address=scheduler_address, run_id="overlap")
    assert overlap.exit_code == EXIT_OK
    for left, right in zip(line.solved, overlap.solved, strict=True):
        assert left.spec.content_hash() == right.spec.content_hash()
        assert left.chain_state.content_hash() == right.chain_state.content_hash()
        assert_frame_equal(left.orders.drop(columns=["run_id"]), right.orders.drop(columns=["run_id"]))
    assert [p.orders for p in line.manifest.portfolios] == [p.orders for p in overlap.manifest.portfolios]


def test_fail_fast_skips_every_lower_priority_portfolio_and_publishes_nothing(tmp_path: Path, scheduler_address: str) -> None:
    data_root = example_book(tmp_path, **{"details/P1.csv": CAPPED_P1})
    report = execute(tmp_path, scheduler_address=scheduler_address, data_root=data_root)
    assert report.exit_code == EXIT_PORTFOLIO_FAILED
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioFailure)
    assert p1.stage == "solve"
    assert p1.error_type == "InfeasibleError"
    assert "upper bounds sum to 0.750000 < required investment 0.800000" in p1.message
    assert isinstance(p2, PortfolioFailure)
    assert p2.stage == "skipped"
    assert not (tmp_path / "run-test" / "run-test" / "orders").exists()
    assert (tmp_path / "run-test" / "run-test" / "manifest.json").exists()
    assert [p.failure_stage for p in report.manifest.portfolios] == ["solve", "skipped"]


def test_continue_isolates_a_failure_nothing_depended_on(tmp_path: Path, scheduler_address: str) -> None:
    data_root = example_book(tmp_path, **{"details/P1.csv": CAPPED_P1})
    report = execute(tmp_path, scheduler_address=scheduler_address, on_error="continue", data_root=data_root, dependencies="none")
    assert report.exit_code == EXIT_PORTFOLIO_FAILED
    assert [type(o).__name__ for o in report.outcomes] == ["PortfolioFailure", "PortfolioResult"]
    assert (tmp_path / "run-test" / "run-test" / "orders" / "orders.parquet").exists()
    assert report.manifest.portfolios[1].status == "solved"


def test_continue_skips_the_portfolios_that_depended_on_the_failure_and_names_it(tmp_path: Path, scheduler_address: str) -> None:
    data_root = example_book(tmp_path, **{"details/P1.csv": CAPPED_P1})
    report = execute(tmp_path, scheduler_address=scheduler_address, on_error="continue", data_root=data_root)
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioFailure) and p1.stage == "solve"
    assert isinstance(p2, PortfolioFailure) and p2.stage == "skipped"
    assert "predecessor 'P1' failed at stage 'solve'" in p2.message
    assert report.manifest.portfolios[1].predecessors == 1


def test_a_portfolio_holding_a_name_the_build_cannot_place_fails_at_build(tmp_path: Path, scheduler_address: str) -> None:
    holdings = (EXAMPLE_DATA / "holdings" / "P2.csv").read_text().replace("P2,B,10000,40", "P2,Z,10000,40")
    data_root = example_book(tmp_path, **{"holdings/P2.csv": holdings})
    report = execute(tmp_path, scheduler_address=scheduler_address, on_error="continue", data_root=data_root, dependencies="none")
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioResult)
    assert isinstance(p2, PortfolioFailure)
    assert p2.stage == "build"
    assert "held securities missing from universe ['Z']" in p2.message


def test_a_failed_build_is_treated_as_overlapping_everything_after_it(tmp_path: Path, scheduler_address: str) -> None:
    holdings = (EXAMPLE_DATA / "holdings" / "P1.csv").read_text().replace("P1,B,10000,40", "P1,Z,10000,40")
    data_root = example_book(tmp_path, **{"holdings/P1.csv": holdings})
    report = execute(tmp_path, scheduler_address=scheduler_address, on_error="continue", data_root=data_root)
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioFailure) and p1.stage == "build"
    assert isinstance(p2, PortfolioFailure) and p2.stage == "skipped" and "predecessor 'P1' failed at stage 'build'" in p2.message


def test_a_portfolio_whose_bundle_is_inconsistent_fails_at_slice(tmp_path: Path, scheduler_address: str) -> None:
    bounds = (EXAMPLE_DATA / "sector_bounds.csv").read_text().replace(",TECH,", ",ENERGY,")
    data_root = example_book(tmp_path, **{"sector_bounds.csv": bounds})
    report = execute(tmp_path, scheduler_address=scheduler_address, on_error="continue", data_root=data_root, dependencies="none")
    assert [outcome.stage for outcome in report.outcomes if isinstance(outcome, PortfolioFailure)] == ["slice", "slice"]
    assert "sector_bounds reference sectors absent from universe ['ENERGY']" in report.outcomes[0].message  # ty: ignore[unresolved-attribute]  # both outcomes are failures, asserted above


def test_sink_failure_is_infrastructure_and_the_manifest_still_records_it(tmp_path: Path, scheduler_address: str) -> None:
    run_dir = tmp_path / "run-test" / "run-test"
    run_dir.mkdir(parents=True)
    (run_dir / "orders").write_text("not a directory")  # the parquet sink cannot create its output directory
    report = execute(tmp_path, scheduler_address=scheduler_address)
    assert report.exit_code == EXIT_INFRASTRUCTURE
    assert report.manifest.exit_code == EXIT_INFRASTRUCTURE
    assert [type(outcome).__name__ for outcome in report.outcomes] == ["PortfolioResult", "PortfolioResult"], "the portfolios solved; only publishing failed"
    sink_record = report.manifest.portfolios[-1]
    assert sink_record.portfolio_id == "*"
    assert sink_record.failure_stage == "sink"
    assert sink_record.error is not None
    assert "orders" in sink_record.error
    assert (run_dir / "manifest.json").exists()


def test_a_portfolio_the_inputs_do_not_cover_fails_alone_and_the_rest_of_the_book_solves(tmp_path: Path, scheduler_address: str) -> None:
    data_root = example_book(tmp_path, **{"details/P2.csv": no_details_csv("P2")})
    report = execute(tmp_path, scheduler_address=scheduler_address, data_root=data_root, on_error="continue")
    assert report.exit_code == EXIT_PORTFOLIO_FAILED
    solved, failed = report.solved, report.failed
    assert [outcome.portfolio_id for outcome in solved] == ["P1"], "P1 has everything it needs and is not held back by P2"
    assert [(o.portfolio_id, o.stage, o.error_type) for o in failed] == [("P2", "load", "MissingInput")]
    record = {r.portfolio_id: r for r in report.manifest.portfolios}["P2"]
    assert record.error is not None and "no details for this portfolio" in record.error


def test_a_required_dataset_that_does_not_load_at_all_still_rejects_the_run(tmp_path: Path, scheduler_address: str) -> None:
    data_root = example_book(tmp_path, **dict.fromkeys(("holdings/P1.csv", "holdings/P2.csv"), "portfolio_id,security_id,quantity\n"))
    with pytest.raises(InputRejectedError, match="holdings"):
        execute(tmp_path, scheduler_address=scheduler_address, data_root=data_root)


def test_manifest_records_provenance_for_every_stage(tmp_path: Path, scheduler_address: str) -> None:
    manifest = execute(tmp_path, scheduler_address=scheduler_address).manifest
    assert manifest.git_sha == GIT.sha
    assert manifest.config.sha256 == resolved_example_real(sink="orders_to_parquet").config_sha256
    assert {d.name for d in manifest.datasets} == {"portfolios", "holdings", "universe", "details", "sector_bounds", "constraints"}
    p1 = manifest.portfolios[0]
    assert [r.qualname for r in p1.rules] == ["portfolio_optimizer.rules:restrict_low_liquidity"]
    assert p1.solve is not None
    assert p1.solve.status == "optimal"
    assert p1.check is not None
    assert p1.check.passed
    assert p1.drift is not None
    assert p1.drift.max_weight_error <= 1e-6, "the example's deltas are whole shares, so what drift is left is the solver's own slack, not rounding"
    assert p1.orders is not None
    assert p1.orders.count == 3
    assert p1.orders.gross_notional == "500000"
    assert (p1.solve_order, p1.predecessors) == ("0", 0)
    assert len(manifest.manifest_sha256) == 64
    assert {a.path.rsplit("/", 1)[-1] for a in manifest.artifacts} >= {"P1.npz", "P2.npz", "orders.parquet"}
    assert manifest.versions.packages == {}  # every step came from the template modules; git_sha covers them


def test_manifest_records_the_package_behind_every_external_step(tmp_path: Path) -> None:
    report = execute(tmp_path, backend_factory=factory_for(LazyBackend()), sink="tests.steps:noop_sink")  # a spawned worker cannot import tests.steps, so this run stays in one process
    assert report.exit_code == EXIT_OK
    assert report.manifest.versions.packages == {"tests": "unknown"}  # tests.steps is importable but no installed distribution provides it


def test_two_runs_over_the_same_inputs_are_identical_except_for_identity(tmp_path: Path, scheduler_address: str) -> None:
    first = execute(tmp_path, scheduler_address=scheduler_address, run_id="one").manifest
    second = execute(tmp_path, scheduler_address=scheduler_address, dependencies="all", run_id="two").manifest
    strip = ("problem_spec_sha256", "chain_inputs_sha256", "orders", "rules", "solve_order")
    for left, right in zip(first.portfolios, second.portfolios, strict=True):
        for field in strip:
            assert getattr(left, field) == getattr(right, field), field
    assert [d.content_sha256 for d in first.datasets] == [d.content_sha256 for d in second.datasets]
