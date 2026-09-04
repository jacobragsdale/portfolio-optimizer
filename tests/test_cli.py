"""Tier 5: one smoke test per entry point over the shipped example, plus the exit-code contract."""

import io
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest

from portfolio_optimizer.cli import RetryError, parse_as_of, retry_of, run_cli
from portfolio_optimizer.config.models import InlinePortfolios, load_run_config
from portfolio_optimizer.engine.manifest import PortfolioRecord
from tests import steps
from tests.conftest import AS_OF, EXAMPLE_CONFIG, HANDOFF_CONFIG, example_body, instant
from tests.engine.support import BUY_ORDERS_P1, BUY_ORDERS_P2, details_csv, example_book, fixed_clock
from tests.engine.test_manifest import manifest

AS_OF_TEXT = "2026-08-28T00:00:00Z"


def cli(argv: Sequence[str], env: dict[str, str] | None = None, run_id: str = "run-smoke") -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = run_cli(argv, env=env or {}, clock=fixed_clock(), new_run_id=lambda: run_id, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def config(tmp_path: Path) -> Path:
    """The shipped config with the mock services' latency removed: the file a smoke test can afford to run."""
    path = tmp_path / "example_inflow.json"
    path.write_text(json.dumps(example_body()))
    return path


@pytest.fixture
def env(tmp_path: Path, scheduler_address: str) -> dict[str, str]:
    """Settings that point the run at the session cluster and the two-account book, so a CLI test does not pay a cluster start or the shipped hundred accounts."""
    return {
        "PORTFOLIO_OPTIMIZER_OUTPUT_DIR": str(tmp_path / "out"),
        "PORTFOLIO_OPTIMIZER_DATA_ROOT": str(example_book(tmp_path)),
        "PORTFOLIO_OPTIMIZER_LOG_LEVEL": "WARNING",
        "PORTFOLIO_OPTIMIZER_CLUSTER": scheduler_address,
        "PORTFOLIO_OPTIMIZER_MIN_WORKERS": "1",
        "PORTFOLIO_OPTIMIZER_MAX_WORKERS": "2",
        "PORTFOLIO_OPTIMIZER_CLUSTER_TIMEOUT_S": "120",
    }


def test_run_produces_the_golden_orders_and_a_manifest(tmp_path: Path, env: dict[str, str], config: Path) -> None:
    code, out, err = cli(["run", str(config), "--as-of", AS_OF_TEXT], env)
    assert code == 0, err
    assert "run run-smoke" in out
    assert "P1: solved, 2 order(s); binding: " in out and "ub" in out.split("P1: solved")[1].split("\n")[0], "the cap on A binds, and the run says so"
    assert "P2: solved, 1 order(s); binding: " in out and "adv/cumulative_participation" in out.split("P2: solved")[1], "why P2 did not buy C: P1 spent its budget"
    assert "  check restricted_never_traded: not_exercised, 0 examined, 0 violation(s)" in out and "  check wash_sale_window: not_exercised, 0 examined, 0 violation(s)" in out, (
        "every check's verdict, after the portfolios: this book proves neither"
    )
    run_dir = tmp_path / "out" / "run-smoke"
    orders = pd.read_parquet(run_dir / "orders" / "orders.parquet")
    assert orders[["portfolio_id", "security_id", "side", "quantity"]].to_dict("records") == [
        *({"portfolio_id": "P1", **order} for order in BUY_ORDERS_P1),
        *({"portfolio_id": "P2", **order} for order in BUY_ORDERS_P2),
    ]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["as_of_date"] == AS_OF.isoformat().replace("+00:00", "Z") and manifest["tags"] == {"desk": "template"}


def test_retry_of_reruns_the_failed_portfolios_under_whatever_config_the_desk_writes(tmp_path: Path, env: dict[str, str], config: Path) -> None:
    """P1 capped at a quarter holds A and B at 0.3 each, which the inflow cannot sell down; under ``fail_fast`` P2 is skipped behind it.

    Two retries of that run: the rebalance config over the failed solve alone, the default; and the
    inflow itself with the box held, over the failed solve and the portfolio skipped behind it.
    """
    capped = env | {"PORTFOLIO_OPTIMIZER_DATA_ROOT": str(example_book(tmp_path / "capped", **{"details.csv": details_csv(P1={"max_weight": "0.25"})}))}
    code, out, _ = cli(["run", str(config), "--as-of", AS_OF_TEXT], capped, run_id="failed")
    assert code == 1 and "P1: FAILED at solve: InfeasibleError" in out and "P2: FAILED at skipped" in out
    failed_manifest = tmp_path / "out" / "failed" / "manifest.json"
    rebalance = tmp_path / "example_rebalance.json"
    rebalance.write_text(json.dumps(example_body() | {"order_flow": "rebalance"}))
    code, out, err = cli(["run", str(rebalance), "--as-of", AS_OF_TEXT, "--retry-of", str(failed_manifest)], capped, run_id="retry")
    assert code == 0, err
    assert "P1: solved" in out and "P2" not in out, "the default retries the failed solves alone"
    manifest = json.loads((tmp_path / "out" / "retry" / "manifest.json").read_text())
    assert manifest["tags"] == {"desk": "template", "retry_of": "failed"}
    assert manifest["config"]["resolved"]["datasets"]["portfolios"]["ids"] == ["P1"], "the ids are the retry's book, written inline"
    orders = pd.read_parquet(tmp_path / "out" / "retry" / "orders" / "orders.parquet")
    assert set(orders["side"]) == {"BUY", "SELL"} and orders["portfolio_id"].unique().tolist() == ["P1"], "A and B are sold down to the cap and the proceeds go to C"
    held = tmp_path / "example_inflow_held.json"
    held.write_text(json.dumps(example_body() | {"build": {"name": "standard", "params": {"hold_breached_starts": True}}}))
    code, out, err = cli(["run", str(held), "--as-of", AS_OF_TEXT, "--retry-of", str(failed_manifest), "--retry-stages", "solve,skipped"], capped, run_id="held")
    assert code == 0, err
    assert "P1: solved" in out and "P2: solved" in out, "the same inflow, the box held, over the failed solve and the portfolio fail_fast skipped behind it"
    manifest = json.loads((tmp_path / "out" / "held" / "manifest.json").read_text())
    assert manifest["tags"] == {"desk": "template", "retry_of": "failed"} and manifest["config"]["resolved"]["datasets"]["portfolios"]["ids"] == ["P1", "P2"]
    orders = pd.read_parquet(tmp_path / "out" / "held" / "orders" / "orders.parquet")
    assert set(orders["side"]) == {"BUY"} and set(orders.loc[orders["portfolio_id"] == "P1", "security_id"]) == {"C"}, "P1 buys C with its cash and neither A nor B, which are held over the cap"
    code, _, err = cli(["run", str(rebalance), "--as-of", AS_OF_TEXT, "--retry-of", str(failed_manifest), "--retry-stages", "load"], capped, run_id="nothing-at-load")
    assert code == 2 and "no portfolio in run failed failed at load; the run recorded 1 at skipped (SkippedAfterFailure), 1 at solve (InfeasibleError)" in err
    code, _, err = cli(["run", str(rebalance), "--as-of", AS_OF_TEXT, "--retry-of", str(tmp_path / "out" / "retry" / "manifest.json")], capped, run_id="nothing")
    assert code == 2 and "no portfolio in run retry failed at solve; the run recorded every portfolio solved" in err
    code, _, err = cli(["run", str(rebalance), "--as-of", AS_OF_TEXT, "--retry-stages", "solve,skipped"], capped, run_id="no-manifest")
    assert code == 2 and "give it a manifest" in err
    code, _, err = cli(["run", str(rebalance), "--as-of", AS_OF_TEXT, "--retry-of", str(failed_manifest), "--retry-stages", "solved"], capped, run_id="bad-stage")
    assert code == 2 and "--retry-stages names ['solved']; a failure stage is one of" in err
    code, _, err = cli(["run", str(rebalance), "--as-of", AS_OF_TEXT, "--retry-of", str(tmp_path / "missing.json")], capped, run_id="missing")
    assert code == 3 and "cannot read manifest" in err


HANDOFF_ORDERS = (
    "portfolio_id,security_id,side,quantity,reference_price,notional,target_weight,unrounded_quantity,spec_hash,run_id,as_of_date\n"
    "P1,A,SELL,1000,100,100000,0.2,1000.0,outflow,run-outflow,2026-08-28T00:00:00Z\n"
    "P2,C,SELL,5000,10,50000,0.0,5000.0,outflow,run-outflow,2026-08-28T00:00:00Z\n"
)
"""What an earlier run sold: 1,000 A for P1, 5,000 C for P2 — a twentieth of C's day."""


def test_a_previous_runs_orders_feed_the_next_run_as_its_blotter_and_its_spent_volume(tmp_path: Path, env: dict[str, str]) -> None:
    """The inflow wired through ``load_run_orders`` over that blotter: P1 cannot rebuy A and finds its quarter of C's day less the 5,000 the outflow took of it, 20,000 shares, not 25,000; P2 cannot rebuy C and takes A to its cap."""
    fed = env | {"PORTFOLIO_OPTIMIZER_DATA_ROOT": str(example_book(tmp_path / "fed", **{"outflow_orders.csv": HANDOFF_ORDERS}))}
    body = json.loads(HANDOFF_CONFIG.read_text())
    body["datasets"] = {name: instant(spec) for name, spec in body["datasets"].items()}
    config = tmp_path / "example_inflow_after_outflow.json"
    config.write_text(json.dumps(body))
    code, _, err = cli(["run", str(config), "--as-of", AS_OF_TEXT], fed)
    assert code == 0, err
    orders = pd.read_parquet(tmp_path / "out" / "run-smoke" / "orders" / "orders.parquet")
    assert orders[["portfolio_id", "security_id", "side", "quantity"]].to_dict("records") == [
        {"portfolio_id": "P1", "security_id": "C", "side": "BUY", "quantity": 20000},
        {"portfolio_id": "P2", "security_id": "A", "side": "BUY", "quantity": 3000},
    ]
    manifest = json.loads((tmp_path / "out" / "run-smoke" / "manifest.json").read_text())
    assert {dataset["name"] for dataset in manifest["datasets"]} >= {"trades", "adv_consumed"}, "both shapes of the handoff are inputs the manifest records and hashes"
    assert [step["columns_added"] for step in manifest["assembly"]] == [{"universe": ["alpha", "tcost_bps"]}, {"universe": ["adv_consumed_quantity"]}, {}]


def test_the_default_settings_run_the_example_inline_with_no_environment_at_all(tmp_path: Path, config: Path) -> None:
    code, out, err = cli(["run", str(config), "--as-of", AS_OF_TEXT, "--data-root", str(example_book(tmp_path)), "--output", str(tmp_path / "out")])
    assert code == 0, err
    assert "P1: solved, 2 order(s)" in out
    manifest = json.loads((tmp_path / "out" / "run-smoke" / "manifest.json").read_text())
    assert manifest["cluster"]["kind"] == "inline" and manifest["settings"]["cluster"] == "inline"


def test_rerun_diffs_clean_and_verify_passes_without_cvxpy_objects(tmp_path: Path, env: dict[str, str], config: Path) -> None:
    assert cli(["run", str(config), "--as-of", AS_OF_TEXT], env, run_id="one")[0] == 0
    assert cli(["run", str(config), "--as-of", AS_OF_TEXT], env, run_id="two")[0] == 0
    left = tmp_path / "out" / "one" / "manifest.json"
    right = tmp_path / "out" / "two" / "manifest.json"
    code, out, _ = cli(["diff-manifests", str(left), str(right)])
    assert code == 0
    assert out.strip() == "no differences"
    code, out, err = cli(["verify", "--manifest", str(left), "--portfolio", "P1"])
    assert code == 0, err
    assert "VERIFIED P1" in out
    assert "ok   trade_balance" in out
    assert "ok   sector_floor/group_limit" in out and "ok   sector_cap/group_limit" in out, "two rows of one kind produce residuals of the same name; the label tells them apart"
    assert "ub " in out and "[binding]" in out.split("ub ")[1].split("\n")[0], "the verifier says where the answer sits against a limit"
    code, _, err = cli(["verify", "--manifest", str(left), "--portfolio", "P9"])
    assert code == 2
    assert "was not solved" in err


def test_run_writes_the_recorded_spans_as_a_chrome_trace(tmp_path: Path, env: dict[str, str], config: Path) -> None:
    assert cli(["run", str(config), "--as-of", AS_OF_TEXT], env)[0] == 0
    trace = tmp_path / "out" / "run-smoke" / "trace.json"
    assert trace.exists(), "the manifest's spans are written beside it in the Chrome trace format"
    assert "solve" in {str(event["name"]) for event in json.loads(trace.read_text())["traceEvents"]}


def test_validate_config_lists_every_resolved_step_and_term() -> None:
    code, out, _ = cli(["validate-config", str(EXAMPLE_CONFIG)])
    assert code == 0
    assert "config ok" in out
    assert "dependencies overlap" in out
    assert "rule                portfolio_optimizer.rules:restrict_low_liquidity" in out
    assert "build               portfolio_optimizer.engine.build:standard" in out
    assert "solve               portfolio_optimizer.solvers:cvxpy" in out
    assert "term                alpha (Linear)" in out
    assert "loader              portfolio_optimizer.loaders:load_constraints" in out
    assert "constraint          " not in out, "constraints are loaded data, so validate-config has none to list"


def test_validate_config_renders_every_term_before_saying_ok(tmp_path: Path) -> None:
    assert steps.Lying is not None  # imported for its side effect: the `lying` kind is registered in this process
    body = example_body() | {"objective": [{"kind": "lying", "name": "lie"}]}
    config = tmp_path / "lying.json"
    config.write_text(json.dumps(body))
    code, _, err = cli(["validate-config", str(config)])
    assert code == 2 and "objective[0]: lie: rendered ConstraintSet, expected ObjectiveTerm" in err


def test_validate_config_rejects_a_solver_the_adapter_does_not_know(tmp_path: Path) -> None:
    body = example_body() | {"solve": {"name": "cvxpy", "params": {"solver": "SCIPY"}}}  # cvxpy ships it; the adapter has no record for it, so its version could not be fingerprinted
    config = tmp_path / "scipy.json"
    config.write_text(json.dumps(body))
    code, out, err = cli(["validate-config", str(config)])
    assert code == 2 and out == ""
    assert "config rejected" in err and "solve: solver 'SCIPY' is not one the adapter knows" in err


def test_steps_lists_what_this_environment_can_name() -> None:
    code, out, _ = cli(["steps"])
    assert code == 0
    assert "rule (portfolio_optimizer.rules)" in out and "  restrict_low_liquidity (dataset, key)" in out
    assert "build (portfolio_optimizer.engine.build)" in out and "  standard" in out
    assert "  cvxpy (solver, options, time_limit_s, verbose)" in out
    assert "term kinds" in out and "  linear (" in out
    assert "constraint kinds" in out and "  participation_limit (" in out
    assert "check (portfolio_optimizer.checks)" in out and "  no_trades_inside_wash_window (dataset, window_days)" in out and "  restricted_never_traded" in out
    assert "parameter" not in out.split("rule (")[1].split("solve_order")[0], "a helper in a template module is not a step"


def test_the_as_of_argument_must_be_an_aware_instant(tmp_path: Path, config: Path) -> None:
    assert parse_as_of("2026-08-28T09:30:00-04:00").isoformat() == "2026-08-28T13:30:00+00:00", "normalized to UTC"
    code, _, err = cli(["run", str(config), "--as-of", "2026-08-28T00:00:00"], {"PORTFOLIO_OPTIMIZER_OUTPUT_DIR": str(tmp_path)})
    assert code == 2 and "must carry a time zone" in err
    code, _, err = cli(["run", str(config), "--as-of", "yesterday"], {"PORTFOLIO_OPTIMIZER_OUTPUT_DIR": str(tmp_path)})
    assert code == 2 and "ISO 8601" in err
    assert cli(["run", str(config)])[0] == 2, "the as-of date is required"


def test_exit_code_contract(tmp_path: Path, env: dict[str, str], config: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text('{"run": {}}')
    assert cli(["validate-config", str(bad_json)])[0] == 2
    assert cli(["validate-config", str(tmp_path / "missing.json")])[0] == 3
    assert cli(["run", str(config), "--as-of", AS_OF_TEXT], {"PORTFOLIO_OPTIMIZER_TYPO": "1"})[0] == 2  # an unknown setting
    code, _, err = cli(["run", str(config), "--as-of", AS_OF_TEXT], env | {"PORTFOLIO_OPTIMIZER_DATA_ROOT": str(tmp_path / "nowhere")})
    assert code == 3
    assert "infrastructure failure" in err
    assert cli(["no-such-command"])[0] == 2


def test_run_flags_override_settings(tmp_path: Path, env: dict[str, str], config: Path) -> None:
    code, out, _ = cli(["run", str(config), "--as-of", AS_OF_TEXT, "--output", str(tmp_path / "elsewhere"), "--max-workers", "1"], env)
    assert code == 0
    manifest = json.loads((tmp_path / "elsewhere" / "run-smoke" / "manifest.json").read_text())
    assert "elsewhere" in out
    assert manifest["settings"]["max_workers"] == "1"
    assert manifest["cluster"]["kind"] == "address"
    assert cli(["run", str(config), "--as-of", AS_OF_TEXT, "--max-workers", "0"], env)[0] == 2


def test_a_failed_run_points_at_the_traceback_it_wrote(tmp_path: Path, env: dict[str, str], config: Path) -> None:
    env["PORTFOLIO_OPTIMIZER_DATA_ROOT"] = str(example_book(tmp_path, **{"details.csv": details_csv(P1={"max_weight": "0.25"})}))
    code, out, err = cli(["run", str(config), "--as-of", AS_OF_TEXT], env)
    assert code == 1, err
    report_path = tmp_path / "out" / "run-smoke" / "failures" / "P1.txt"
    assert "P1: FAILED at solve: InfeasibleError" in out
    assert f"(traceback: {report_path})" in out
    assert "InfeasibleError" in report_path.read_text()
    assert "P2: FAILED at skipped" in out
    assert "traceback:" not in out.split("P2: FAILED")[1], "nothing was written for a skipped portfolio, so nothing is offered"


# --- retry_of: the pure function behind --retry-of ---


def _failed(portfolio_id: str, stage: str, solve_order: str | None = None, error: str = "InfeasibleError") -> PortfolioRecord:
    return PortfolioRecord(portfolio_id=portfolio_id, status="failed", solve_order=solve_order, failure_stage=stage, error=f"{error}: {stage} failed")


RECORDED = (
    _failed("P9", "solve", "2", error="VerificationError"),
    _failed("P2", "solve", "1"),
    _failed("P3", "load"),
    _failed("P4", "skipped", "3", error="SkippedAfterFailure"),
    _failed("P1", "solve", "1"),
    PortfolioRecord(portfolio_id="P5", status="solved", solve_order="0"),
)


def test_retry_of_takes_the_selected_failures_in_solve_order_and_tags_the_run() -> None:
    config = load_run_config(json.dumps(example_body()))
    recorded = manifest(run_id="run-failed", portfolios=RECORDED)
    retried = retry_of(config, recorded)
    assert retried.datasets["portfolios"] == InlinePortfolios(ids=("P1", "P2", "P9")), "the default: failed solves only, by their recorded key then id"
    assert retried.run.tags == {"desk": "template", "retry_of": "run-failed"} and retried.run.name == config.run.name
    assert {name: spec for name, spec in retried.datasets.items() if name != "portfolios"} == {name: spec for name, spec in config.datasets.items() if name != "portfolios"}
    assert retried.model_dump(exclude={"datasets", "run"}) == config.model_dump(exclude={"datasets", "run"}), (
        "nothing else in the wiring changes; the config is whatever the desk wrote, an inflow included"
    )
    assert retry_of(config, recorded, stages=frozenset({"solve", "skipped"})).datasets["portfolios"] == InlinePortfolios(ids=("P1", "P2", "P9", "P4")), "the stages select"
    assert retry_of(config, recorded, stages=frozenset({"load"})).datasets["portfolios"] == InlinePortfolios(ids=("P3",))
    assert retry_of(config, recorded, errors=frozenset({"VerificationError"})).datasets["portfolios"] == InlinePortfolios(ids=("P9",)), "the exception type selects within the stages"


def test_retry_of_refuses_a_run_where_nothing_matches_and_says_what_did_fail() -> None:
    config = load_run_config(json.dumps(example_body()))
    with pytest.raises(RetryError, match=r"no portfolio in run run-1 failed at solve; the run recorded every portfolio solved"):
        retry_of(config, manifest())
    with pytest.raises(RetryError, match=r"no portfolio in run run-1 failed at solve; the run recorded 1 at load \(InfeasibleError\), 2 at skipped \(SkippedAfterFailure\)"):
        retry_of(config, manifest(portfolios=(_failed("P1", "load"), _failed("P2", "skipped", error="SkippedAfterFailure"), _failed("P3", "skipped", error="SkippedAfterFailure"))))
    with pytest.raises(RetryError, match=r"no portfolio in run run-1 failed at skipped, solve with DriftError; the run recorded"):
        retry_of(config, manifest(portfolios=RECORDED), stages=frozenset({"solve", "skipped"}), errors=frozenset({"DriftError"}))


def test_run_id_names_the_output_directory_and_is_used_once(tmp_path: Path, config: Path) -> None:
    argv = ["run", str(config), "--as-of", AS_OF_TEXT, "--data-root", str(example_book(tmp_path)), "--output", str(tmp_path / "out")]
    code, out, err = cli([*argv, "--run-id", "qa-2026-09-03"], run_id="ignored")
    assert code == 0, err
    assert "run qa-2026-09-03: manifest" in out and (tmp_path / "out" / "qa-2026-09-03" / "manifest.json").exists(), (
        "the caller's id is the directory, so it knows where the run will land before it finishes"
    )
    code, _, err = cli([*argv, "--run-id", "qa-2026-09-03"])
    assert code == 2 and "already has a manifest" in err
    code, _, err = cli([*argv, "--run-id", "../escape"])
    assert code == 2 and "names the run's output directory" in err
