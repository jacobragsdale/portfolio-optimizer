"""Tier 1/2: the manifest round-trips with an integrity hash, is written atomically, and localizes drift."""

import hashlib
import json
from pathlib import Path

import pytest

from portfolio_optimizer.domain.results import PortfolioFailure
from portfolio_optimizer.engine.environment import WorkerEnvironment
from portfolio_optimizer.engine.manifest import (
    ConfigInfo,
    OrdersRecord,
    PortfolioRecord,
    RunManifest,
    VersionInfo,
    WorkerRecord,
    diff_manifests,
    failure_report_path,
    finalize,
    load_manifest,
    write_failure_reports,
    write_manifest,
)
from portfolio_optimizer.engine.schedule import ScheduleSummary
from tests.conftest import AS_OF


def manifest(**overrides: object) -> RunManifest:
    base: dict[str, object] = {
        "run_id": "run-1",
        "run_name": "r",
        "created_at_utc": AS_OF,
        "as_of_date": AS_OF,
        "git_sha": "abc",
        "git_dirty": False,
        "schedule": ScheduleSummary(coupling="overlap", portfolios=1, edges=0, components=1, largest_component=1, critical_path=1),
        "versions": VersionInfo(python="3.13", cvxpy="1.9", numpy="2.5", pandas="3.0", solver="CLARABEL", solver_version="0.11"),
        "config": ConfigInfo(path="c.json", sha256="cfg", resolved={}),
        "settings": {},
        "terms": (),
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
        ({"versions": manifest().versions.model_copy(update={"packages": {"my-firm-quant": "1.5.0"}})}, "versions: library, solver, or step-package versions differ"),
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


def _worker(host: str, **overrides: object) -> WorkerRecord:
    base: dict[str, object] = {
        "python": "3.13",
        "cvxpy": "1.9",
        "numpy": "2.5",
        "pandas": "3.0",
        "solver": "CLARABEL",
        "solver_version": "0.11",
        "packages": (),
        "git_sha": "abc",
        "image_digest": None,
    }
    return WorkerRecord(environment=WorkerEnvironment.model_validate(base | overrides), hosts=(host,), portfolios=1)


def test_worker_hosts_do_not_differ_but_worker_environments_do() -> None:
    left = manifest(versions=manifest().versions.model_copy(update={"workers": (_worker("laptop"),)}))
    same_environment_elsewhere = manifest(versions=manifest().versions.model_copy(update={"workers": (_worker("pod-1"), _worker("pod-2"))}))
    stale_worker = manifest(versions=manifest().versions.model_copy(update={"workers": (_worker("pod-1"), _worker("pod-2", git_sha="old"))}))
    assert diff_manifests(left, same_environment_elsewhere) == []
    assert "versions: library, solver, or step-package versions differ" in diff_manifests(left, stale_worker)


def failed(portfolio_id: str = "P1", stage: str = "solve", traceback: str | None = "Traceback (most recent call last):\n  boom\n") -> PortfolioFailure:
    return PortfolioFailure(portfolio_id=portfolio_id, stage=stage, error_type="KeyError", message="no such column 'oas'", traceback=traceback)


def test_a_failure_report_names_the_portfolio_and_carries_the_traceback(tmp_path: Path) -> None:
    (artifact,) = write_failure_reports([failed()], tmp_path, run_id="run-1")
    path = Path(artifact.path)
    assert path == tmp_path / "failures" / "P1.txt"
    text = path.read_text()
    assert "run_id: run-1" in text
    assert "portfolio_id: P1" in text
    assert "stage: solve" in text
    assert "error: KeyError: no such column 'oas'" in text
    assert text.rstrip().endswith("boom")


def test_a_failure_with_no_traceback_writes_nothing(tmp_path: Path) -> None:
    assert write_failure_reports([failed("P2", "skipped", traceback=None)], tmp_path, run_id="run-1") == ()
    assert not (tmp_path / "failures").exists(), "a skipped portfolio has no where to record"


def test_the_run_s_own_failure_is_named_for_its_stage(tmp_path: Path) -> None:
    (artifact,) = write_failure_reports([failed("*", "sink")], tmp_path, run_id="run-1")
    assert Path(artifact.path) == tmp_path / "failures" / "sink.txt"


def test_reports_are_written_atomically_and_hashed(tmp_path: Path) -> None:
    (artifact,) = write_failure_reports([failed()], tmp_path, run_id="run-1")
    path = Path(artifact.path)
    assert sorted(p.name for p in path.parent.iterdir()) == ["P1.txt"], "no temp file survives"
    assert artifact.size_bytes == len(path.read_bytes())
    assert artifact.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_two_failures_that_map_to_one_path_are_written_once(tmp_path: Path) -> None:
    written = write_failure_reports([failed("sink", "solve"), failed("*", "sink")], tmp_path, run_id="run-1")
    assert [Path(a.path).name for a in written] == ["sink.txt"]
    assert (tmp_path / "failures" / "sink.txt").read_text().count("portfolio_id: sink") == 1, "the first failure keeps the path"


def test_the_report_path_is_the_one_the_writer_used(tmp_path: Path) -> None:
    (artifact,) = write_failure_reports([failed()], tmp_path, run_id="run-1")
    assert Path(artifact.path) == failure_report_path(tmp_path, failed())
