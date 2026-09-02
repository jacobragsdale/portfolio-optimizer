"""Tier 2: on the real cluster — the golden orders, the line equals the overlap schedule, failures skip only what depended on them (or everything under fail_fast), and nothing partial is published."""

import hashlib
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from portfolio_optimizer.domain.results import PortfolioFailure, PortfolioResult
from portfolio_optimizer.engine.runner import EXIT_INFRASTRUCTURE, EXIT_OK, EXIT_PORTFOLIO_FAILED, InputRejectedError
from tests.conftest import EXAMPLE_DATA, resolved_example_real
from tests.engine.fakes import LazyBackend, factory_for
from tests.engine.support import BUY_ORDERS_P1, BUY_ORDERS_P2, GIT, details_csv, details_without, example_book, execute, uncoupled_book

CAPPED_P1 = {"details.csv": details_csv(P1={"max_weight": "0.25"})}
"""P1's cap at a quarter: it holds A and B at 0.3 each, and a buy program cannot sell them down, so its solve is infeasible."""


def test_the_run_reproduces_the_hand_checked_orders(tmp_path: Path, scheduler_address: str) -> None:
    report = execute(tmp_path, scheduler_address=scheduler_address)
    assert report.exit_code == EXIT_OK
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioResult)
    assert isinstance(p2, PortfolioResult)
    assert p1.orders[["security_id", "side", "quantity"]].to_dict("records") == BUY_ORDERS_P1
    assert p2.orders[["security_id", "side", "quantity"]].to_dict("records") == BUY_ORDERS_P2, "P2 wants C too, but P1 spent C's ADV budget, so its cash goes to A"
    assert p2.chain_state.traded_shares.tolist() == [1000.0, 0.0, 25000.0], "every buy P1 made reaches P2"
    assert p2.chain_state.predecessors == ("P1",)
    assert "adv/cumulative_participation" in p2.report.active, "why P2 did not buy C, in the report"
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
    data_root = example_book(tmp_path, **CAPPED_P1)
    report = execute(tmp_path, scheduler_address=scheduler_address, data_root=data_root)
    assert report.exit_code == EXIT_PORTFOLIO_FAILED
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioFailure)
    assert p1.stage == "solve"
    assert p1.error_type == "InfeasibleError"
    assert "names whose cap is below their holding, which this side cannot trade out of: ['A', 'B']" in p1.message
    assert isinstance(p2, PortfolioFailure)
    assert p2.stage == "skipped"
    assert not (tmp_path / "run-test" / "run-test" / "orders").exists()
    assert (tmp_path / "run-test" / "run-test" / "manifest.json").exists()
    assert [p.failure_stage for p in report.manifest.portfolios] == ["solve", "skipped"]


def test_continue_isolates_a_failure_nothing_depended_on(tmp_path: Path, scheduler_address: str) -> None:
    data_root = uncoupled_book(tmp_path, **CAPPED_P1)
    report = execute(tmp_path, scheduler_address=scheduler_address, on_error="continue", data_root=data_root)
    assert report.exit_code == EXIT_PORTFOLIO_FAILED
    assert [type(o).__name__ for o in report.outcomes] == ["PortfolioFailure", "PortfolioResult"]
    assert (tmp_path / "run-test" / "run-test" / "orders" / "orders.parquet").exists()
    assert report.manifest.portfolios[1].status == "solved"


def test_continue_skips_the_portfolios_that_depended_on_the_failure_and_names_it(tmp_path: Path, scheduler_address: str) -> None:
    data_root = example_book(tmp_path, **CAPPED_P1)
    report = execute(tmp_path, scheduler_address=scheduler_address, on_error="continue", data_root=data_root)
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioFailure) and p1.stage == "solve"
    assert isinstance(p2, PortfolioFailure) and p2.stage == "skipped"
    assert "predecessor 'P1' failed at stage 'solve'" in p2.message
    assert report.manifest.portfolios[1].predecessors == 1


def test_a_portfolio_holding_a_name_the_build_cannot_place_fails_at_build(tmp_path: Path, scheduler_address: str) -> None:
    holdings = (EXAMPLE_DATA / "holdings.csv").read_text().replace("P2,B,6000,60", "P2,Z,6000,60")
    data_root = uncoupled_book(tmp_path, **{"holdings.csv": holdings})
    report = execute(tmp_path, scheduler_address=scheduler_address, on_error="continue", data_root=data_root)
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioResult)
    assert isinstance(p2, PortfolioFailure)
    assert p2.stage == "build"
    assert "held securities missing from universe ['Z']" in p2.message


def test_a_failed_build_is_treated_as_overlapping_everything_after_it(tmp_path: Path, scheduler_address: str) -> None:
    holdings = (EXAMPLE_DATA / "holdings.csv").read_text().replace("P1,B,6000,60", "P1,Z,6000,60")
    data_root = example_book(tmp_path, **{"holdings.csv": holdings})
    report = execute(tmp_path, scheduler_address=scheduler_address, on_error="continue", data_root=data_root)
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioFailure) and p1.stage == "build"
    assert isinstance(p2, PortfolioFailure) and p2.stage == "skipped" and "predecessor 'P1' failed at stage 'build'" in p2.message


def test_a_portfolio_whose_bundle_is_inconsistent_fails_at_slice(tmp_path: Path, scheduler_address: str) -> None:
    new_york = {"state": "NEW YORK"}  # a string the frame schema types but the account model refuses
    data_root = uncoupled_book(tmp_path, **{"details.csv": details_csv(P1=new_york, P2=new_york)})
    report = execute(tmp_path, scheduler_address=scheduler_address, on_error="continue", data_root=data_root)
    assert [outcome.stage for outcome in report.outcomes if isinstance(outcome, PortfolioFailure)] == ["slice", "slice"]
    assert "String should match pattern" in report.outcomes[0].message  # ty: ignore[unresolved-attribute]  # both outcomes are failures, asserted above


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
    data_root = example_book(tmp_path, **{"details.csv": details_without("P2")})
    report = execute(tmp_path, scheduler_address=scheduler_address, data_root=data_root, on_error="continue")
    assert report.exit_code == EXIT_PORTFOLIO_FAILED
    solved, failed = report.solved, report.failed
    assert [outcome.portfolio_id for outcome in solved] == ["P1"], "P1 has everything it needs and is not held back by P2"
    assert [(o.portfolio_id, o.stage, o.error_type) for o in failed] == [("P2", "load", "MissingInput")]
    record = {r.portfolio_id: r for r in report.manifest.portfolios}["P2"]
    assert record.error is not None and "no details for this portfolio" in record.error


def test_a_required_dataset_that_does_not_load_at_all_still_rejects_the_run(tmp_path: Path, scheduler_address: str) -> None:
    data_root = example_book(tmp_path, **{"holdings.csv": "portfolio_id,security_id,quantity\n"})
    with pytest.raises(InputRejectedError, match="holdings"):
        execute(tmp_path, scheduler_address=scheduler_address, data_root=data_root)


def test_manifest_records_provenance_for_every_stage(tmp_path: Path, scheduler_address: str) -> None:
    manifest = execute(tmp_path, scheduler_address=scheduler_address).manifest
    assert manifest.git_sha == GIT.sha
    assert manifest.config.sha256 == resolved_example_real(sink="orders_to_parquet").config_sha256
    assert {d.name for d in manifest.datasets} == {"portfolios", "holdings", "universe", "details", "constraints", "global_parameters", "buy_universe_parameters"}
    assert [term["name"] for term in manifest.terms] == ["alpha", "transaction_cost"], "the terms as records, readable by verify without the config"
    p1 = manifest.portfolios[0]
    assert [r.qualname for r in p1.rules] == ["portfolio_optimizer.rules:restrict_low_liquidity"]
    assert [record["kind"] for record in p1.constraints] == ["cash_limit", "cash_limit", "turnover_limit", "group_limit", "group_limit", "participation_limit"]
    assert p1.solve is not None
    assert p1.solve.status == "optimal"
    assert p1.solve.duals["adv"] > 0.0, "the shadow price of the ADV budget: what P1 would gain from one more dollar of it"
    assert p1.check is not None
    assert p1.check.passed
    assert "ub" in p1.check.active and "adv/participation" in p1.check.active
    assert p1.drift is not None
    assert p1.drift.max_weight_error <= 1e-6, "the example's deltas are whole shares, so what drift is left is the solver's own slack, not rounding"
    assert p1.orders is not None
    assert p1.orders.count == 2
    assert p1.orders.gross_notional == "350000"
    assert (p1.solve_order, p1.predecessors) == ("0", 0)
    assert len(manifest.manifest_sha256) == 64
    assert {a.path.rsplit("/", 1)[-1] for a in manifest.artifacts} >= {"P1.npz", "P2.npz", "orders.parquet", "trace.json"}
    assert manifest.versions.packages == {}  # every step came from the template modules; git_sha covers them
    stages = {span.stage for span in manifest.timing}
    assert {"load", "assembly", "dataset", "cluster", "build", "solve", "sink"} <= stages, "every stage the run passed through left a span"
    assert {span.portfolio_id for span in manifest.timing if span.name == "solve"} == {"P1", "P2"}
    assert {span.name for span in manifest.timing if span.stage == "dataset"} == {f"dataset:{d.name}" for d in manifest.datasets}


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


def test_a_book_whose_rows_read_no_chain_is_told_so_before_any_build_and_still_matches_the_line(tmp_path: Path) -> None:
    """Without the ADV rows nothing reads the chain, and the run can see that from the data alone: nothing waits, and the answer is the line's."""
    book = uncoupled_book(tmp_path)
    free = execute(tmp_path, backend_factory=factory_for(LazyBackend()), data_root=book, run_id="free")
    line = execute(tmp_path, backend_factory=factory_for(LazyBackend()), data_root=book, dependencies="all", run_id="line")
    assert free.exit_code == EXIT_OK, [str(outcome) for outcome in free.outcomes]
    schedule = free.manifest.schedule
    assert schedule is not None and (schedule.coupling, schedule.edges, schedule.components, schedule.critical_path) == ("none", 0, 2, 1), (
        "both accounts can buy every name, but neither reads the chain"
    )
    assert [record.predecessors for record in free.manifest.portfolios] == [0, 0]
    assert line.manifest.schedule is not None and line.manifest.schedule.coupling == "none", "the diagnostic line is moot when nothing reads the chain"
    for left, right in zip(free.solved, line.solved, strict=True):
        assert left.orders.drop(columns=["run_id"]).equals(right.orders.drop(columns=["run_id"]))
        assert left.chain_state.content_hash() == right.chain_state.content_hash()


def test_a_worker_side_failure_carries_its_traceback_home_and_is_persisted(tmp_path: Path, scheduler_address: str) -> None:
    data_root = example_book(tmp_path, **CAPPED_P1)
    report = execute(tmp_path, scheduler_address=scheduler_address, on_error="continue", data_root=data_root)
    p1, p2 = report.outcomes
    assert isinstance(p1, PortfolioFailure) and p1.stage == "solve"
    assert p1.traceback is not None, "the solve ran in a worker process; without this the frames die with it"
    assert "InfeasibleError" in p1.traceback
    assert "portfolio_optimizer/engine/" in p1.traceback, "the engine frames, not just the message"
    report_path = tmp_path / "run-test" / "run-test" / "failures" / "P1.txt"
    assert report_path.read_text().endswith(p1.traceback)
    assert isinstance(p2, PortfolioFailure) and p2.stage == "skipped"
    assert p2.traceback is None
    assert {path.name for path in report_path.parent.iterdir()} == {"P1.txt"}, "a skipped portfolio has no traceback to write"


def test_the_failure_report_is_a_manifest_artifact(tmp_path: Path, scheduler_address: str) -> None:
    data_root = example_book(tmp_path, **CAPPED_P1)
    report = execute(tmp_path, scheduler_address=scheduler_address, on_error="continue", data_root=data_root)
    report_path = tmp_path / "run-test" / "run-test" / "failures" / "P1.txt"
    artifact = next(a for a in report.manifest.artifacts if a.path == str(report_path))
    assert artifact.sha256 == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert artifact.size_bytes == len(report_path.read_bytes())


def test_the_sink_failure_writes_its_own_traceback(tmp_path: Path, scheduler_address: str) -> None:
    run_dir = tmp_path / "run-test" / "run-test"
    run_dir.mkdir(parents=True)
    (run_dir / "orders").write_text("not a directory")  # the parquet sink cannot create its output directory
    report = execute(tmp_path, scheduler_address=scheduler_address)
    assert report.exit_code == EXIT_INFRASTRUCTURE
    report_path = run_dir / "failures" / "sink.txt"
    text = report_path.read_text()
    assert "portfolio_id: *" in text
    assert "stage: sink" in text
    assert "FileExistsError" in text
    assert "portfolio_optimizer/sinks.py" in text, "the sink's own frame, so the failure is placed in the step that raised it"
    assert str(report_path) in {a.path for a in report.manifest.artifacts}
