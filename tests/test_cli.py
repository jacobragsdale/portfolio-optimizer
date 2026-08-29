"""Tier 5: one smoke test per entry point over the shipped example, plus the exit-code contract."""

import io
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest

from portfolio_optimizer.cli import run_cli
from tests.conftest import EXAMPLE_CONFIG, EXAMPLE_DATA, FixedClock, FixedIds


def cli(argv: Sequence[str], env: dict[str, str] | None = None, run_id: str = "run-smoke") -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = run_cli(argv, env=env or {}, clock=FixedClock(), ids=FixedIds(run_id), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def env(tmp_path: Path, scheduler_address: str) -> dict[str, str]:
    """Settings that point the run at the session cluster, so a CLI test does not pay a cluster start."""
    return {
        "PORTFOLIO_OPTIMIZER_OUTPUT_DIR": str(tmp_path / "out"),
        "PORTFOLIO_OPTIMIZER_DATA_ROOT": str(EXAMPLE_DATA),
        "PORTFOLIO_OPTIMIZER_LOG_LEVEL": "WARNING",
        "PORTFOLIO_OPTIMIZER_CLUSTER": scheduler_address,
        "PORTFOLIO_OPTIMIZER_MIN_WORKERS": "1",
        "PORTFOLIO_OPTIMIZER_MAX_WORKERS": "2",
        "PORTFOLIO_OPTIMIZER_CLUSTER_TIMEOUT_S": "120",
    }


def test_run_produces_the_golden_orders_and_a_manifest(tmp_path: Path, env: dict[str, str]) -> None:
    code, out, err = cli(["run", str(EXAMPLE_CONFIG)], env)
    assert code == 0, err
    assert "run run-smoke" in out
    assert "P1: solved, 3 order(s)" in out
    assert "P2: solved, 0 order(s)" in out
    run_dir = tmp_path / "out" / "run-smoke"
    orders = pd.read_parquet(run_dir / "orders" / "orders.parquet")
    assert orders[["portfolio_id", "security_id", "side", "quantity"]].to_dict("records") == [
        {"portfolio_id": "P1", "security_id": "A", "side": "SELL", "quantity": 1250},
        {"portfolio_id": "P1", "security_id": "B", "side": "SELL", "quantity": 2500},
        {"portfolio_id": "P1", "security_id": "C", "side": "BUY", "quantity": 25000},
    ]
    assert (run_dir / "manifest.json").exists()


def test_rerun_diffs_clean_and_verify_passes_without_cvxpy_objects(tmp_path: Path, env: dict[str, str]) -> None:
    assert cli(["run", str(EXAMPLE_CONFIG)], env, run_id="one")[0] == 0
    assert cli(["run", str(EXAMPLE_CONFIG)], env, run_id="two")[0] == 0
    left = tmp_path / "out" / "one" / "manifest.json"
    right = tmp_path / "out" / "two" / "manifest.json"
    code, out, _ = cli(["diff-manifests", str(left), str(right)])
    assert code == 0
    assert out.strip() == "no differences"
    code, out, err = cli(["verify", "--manifest", str(left), "--portfolio", "P1"])
    assert code == 0, err
    assert "VERIFIED P1" in out
    assert "ok   trade_balance" in out
    code, _, err = cli(["verify", "--manifest", str(left), "--portfolio", "P9"])
    assert code == 2
    assert "was not solved" in err


def test_validate_config_lists_every_resolved_step() -> None:
    code, out, _ = cli(["validate-config", str(EXAMPLE_CONFIG)])
    assert code == 0
    assert "config ok" in out
    assert "rule                portfolio_optimizer.rules:restrict_low_liquidity" in out
    assert "constraint          portfolio_optimizer.terms:cumulative_adv_participation [chain]" in out


def test_exit_code_contract(tmp_path: Path, env: dict[str, str]) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text('{"run": {}}')
    assert cli(["validate-config", str(bad_json)])[0] == 2
    assert cli(["validate-config", str(tmp_path / "missing.json")])[0] == 3
    assert cli(["run", str(EXAMPLE_CONFIG)], {})[0] == 2  # settings missing
    code, _, err = cli(["run", str(EXAMPLE_CONFIG)], env | {"PORTFOLIO_OPTIMIZER_DATA_ROOT": str(tmp_path / "nowhere")})
    assert code == 3
    assert "infrastructure failure" in err
    assert cli(["no-such-command"])[0] == 2


def test_run_flags_override_settings(tmp_path: Path, env: dict[str, str]) -> None:
    code, out, _ = cli(["run", str(EXAMPLE_CONFIG), "--output", str(tmp_path / "elsewhere"), "--max-workers", "1"], env)
    assert code == 0
    manifest = json.loads((tmp_path / "elsewhere" / "run-smoke" / "manifest.json").read_text())
    assert "elsewhere" in out
    assert manifest["settings"]["max_workers"] == "1"
    assert manifest["cluster"]["kind"] == "address"
    assert cli(["run", str(EXAMPLE_CONFIG), "--max-workers", "0"], env)[0] == 2
