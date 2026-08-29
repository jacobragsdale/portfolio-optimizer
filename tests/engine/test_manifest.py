"""Tier 1/2: the manifest round-trips with an integrity hash, is written atomically, and localizes drift."""

import json
from pathlib import Path

import pytest

from portfolio_optimizer.engine.manifest import ConfigInfo, OrdersRecord, PortfolioRecord, RunManifest, VersionInfo, diff_manifests, finalize, load_manifest, read_git_info, write_manifest
from tests.conftest import AS_OF


def manifest(**overrides: object) -> RunManifest:
    base: dict[str, object] = {
        "run_id": "run-1",
        "run_name": "r",
        "created_at_utc": AS_OF,
        "as_of": AS_OF,
        "git_sha": "abc",
        "git_dirty": False,
        "execution_mode": "sequential",
        "versions": VersionInfo(python="3.13", cvxpy="1.9", numpy="2.5", pandas="3.0", solver="CLARABEL", solver_version="0.11"),
        "config": ConfigInfo(path="c.json", sha256="cfg", resolved={}),
        "settings": {},
        "terms": (),
        "constraints": (),
        "datasets": (),
        "portfolios": (
            PortfolioRecord(portfolio_id="P1", status="solved", problem_spec_sha256="spec1", chain_inputs_sha256="chain1", orders=OrdersRecord(count=1, sha256="orders1", gross_notional="1")),
        ),
        "artifacts": (),
        "exit_code": 0,
    }
    return finalize(RunManifest.model_validate(base | overrides))


def test_finalize_stamps_a_hash_that_load_verifies() -> None:
    stamped = manifest()
    assert len(stamped.manifest_sha256) == 64
    assert load_manifest(stamped.model_dump_json()).manifest_sha256 == stamped.manifest_sha256


def test_tampered_manifest_is_rejected() -> None:
    body = json.loads(manifest().model_dump_json())
    body["git_sha"] = "def"
    with pytest.raises(ValueError, match="does not match its content"):
        load_manifest(json.dumps(body))


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = write_manifest(manifest(), tmp_path / "run")
    assert path.name == "manifest.json"
    assert sorted(p.name for p in path.parent.iterdir()) == ["manifest.json"]
    assert load_manifest(path.read_text()).run_id == "run-1"


def test_identical_manifests_do_not_differ() -> None:
    assert diff_manifests(manifest(), manifest(run_id="run-2")) == []


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"config": ConfigInfo(path="c.json", sha256="other", resolved={})}, "config: resolved config differs"),
        ({"git_sha": "zzz"}, "code: git sha abc vs zzz"),
        (
            {
                "portfolios": (
                    PortfolioRecord(portfolio_id="P1", status="solved", problem_spec_sha256="spec2", chain_inputs_sha256="chain1", orders=OrdersRecord(count=1, sha256="orders1", gross_notional="1")),
                )
            },
            "P1: first divergence at spec",
        ),
        (
            {
                "portfolios": (
                    PortfolioRecord(portfolio_id="P1", status="solved", problem_spec_sha256="spec1", chain_inputs_sha256="chain1", orders=OrdersRecord(count=2, sha256="orders2", gross_notional="2")),
                )
            },
            "P1: first divergence at orders",
        ),
        ({"portfolios": (PortfolioRecord(portfolio_id="P1", status="failed", failure_stage="solve", error="x"),)}, "P1: first divergence at status (solved vs failed)"),
        ({"portfolios": ()}, "P1: missing from the second manifest"),
    ],
)
def test_diff_names_the_first_divergence(overrides: dict[str, object], expected: str) -> None:
    assert expected in diff_manifests(manifest(), manifest(**overrides))


def test_git_info_outside_a_repository_is_unknown(tmp_path: Path) -> None:
    info = read_git_info(tmp_path)
    assert info.sha == "unknown"
    assert not info.dirty
