"""The run manifest: everything needed to reproduce a run and localize any drift between two runs.

The manifest carries the engine's own audit records — dataset, assembly, rule, step, artifact,
schedule — as they are; the models here are the per-portfolio summaries and the envelope.
"""

import json
import platform
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, Field

from portfolio_optimizer.domain.results import Artifact, AssemblyAuditRecord, PortfolioFailure, PortfolioResult, RuleAuditRecord, StepRef
from portfolio_optimizer.domain.types import StrictModel
from portfolio_optimizer.engine.environment import WorkerEnvironment, distribution_version
from portfolio_optimizer.engine.hashing import frame_sha256, json_sha256
from portfolio_optimizer.engine.load import DatasetAudit
from portfolio_optimizer.engine.schedule import ScheduleSummary

MANIFEST_FILENAME = "manifest.json"


class WorkerRecord(StrictModel):
    """One environment that executed tasks for the run, the hosts it ran on, and how many portfolios it handled."""

    environment: WorkerEnvironment
    hosts: tuple[str, ...]
    portfolios: int


class VersionInfo(StrictModel):
    """Library versions that determine numerical results, and the version of every package that supplied a step.

    ``packages`` maps the distribution behind each step named outside the template modules to its
    installed version (``{"my-firm-quant": "1.4.2"}``); a module that no distribution claims is
    recorded under its own name as ``unknown``. Steps from the template modules are covered by ``git_sha``.
    ``workers`` lists every distinct environment that executed a task — normally exactly one, equal to
    the run's own — with the hosts it ran on; a worker whose fingerprint differed failed its portfolio
    at stage ``worker`` and still appears here.
    """

    python: str
    cvxpy: str
    numpy: str
    pandas: str
    solver: str
    solver_version: str
    packages: dict[str, str] = Field(default_factory=dict)
    workers: tuple[WorkerRecord, ...] = ()


class ClusterRecord(StrictModel):
    """The cluster's lifetime: what the run asked for, when it answered, and when it was released.

    ``provision_started_at`` is before the load stage and ``first_worker_ready_at`` after assembly, so
    their difference is the start-up the load stage hid; ``workers_ready`` is how many workers had
    joined when the first task could run.
    """

    kind: str
    min_workers: int
    max_workers: int
    workers_ready: int | None
    scheduler_address: str | None
    provision_started_at: AwareDatetime | None
    first_worker_ready_at: AwareDatetime | None
    closed_at: AwareDatetime | None


class ConfigInfo(StrictModel):
    """The resolved run config and its hash."""

    path: str
    sha256: str
    resolved: dict[str, object]


class SolveRecord(StrictModel):
    """Solver identity and outcome; the cvxpy version behind the shipped step is in ``versions``."""

    solver: str
    solver_version: str
    status: str
    iterations: int | None
    objective_value: float | None
    solve_time_s: float


class CheckRecord(StrictModel):
    """Outcome of the independent verification; ``tolerance`` is the violation every residual was held to."""

    tolerance: float
    max_violation: float
    violated: tuple[str, ...]
    objective_gap: float
    objective_passed: bool
    unverified: tuple[str, ...]
    passed: bool


class DriftRecord(StrictModel):
    """Rounding drift between solved and executed weights."""

    max_weight_error: float
    tolerance: float
    dropped_orders: int
    passed: bool


class OrdersRecord(StrictModel):
    """Summary of a portfolio's orders."""

    count: int
    sha256: str
    gross_notional: str


class PortfolioRecord(StrictModel):
    """Everything recorded about one portfolio."""

    portfolio_id: str
    status: Literal["solved", "failed"]
    solve_order: str | None = None
    predecessors: int | None = None
    rules: tuple[RuleAuditRecord, ...] = ()
    constraints: tuple[StepRef, ...] = ()
    """What the solve step made of this portfolio's constraint rows, after its rules.

    Per portfolio, because constraints are loaded data and a rule may change them; the run-level block
    that used to hold them could not say what any one account solved.
    """

    problem_spec_sha256: str | None = None
    chain_inputs_sha256: str | None = None
    solve: SolveRecord | None = None
    check: CheckRecord | None = None
    drift: DriftRecord | None = None
    orders: OrdersRecord | None = None
    failure_stage: str | None = None
    error: str | None = None


class RunManifest(StrictModel):
    """The audit record of one run."""

    run_id: str
    run_name: str
    created_at_utc: AwareDatetime
    as_of_date: AwareDatetime
    git_sha: str
    git_dirty: bool
    schedule: ScheduleSummary | None = None
    cluster: ClusterRecord | None = None
    versions: VersionInfo
    config: ConfigInfo
    settings: dict[str, str]
    terms: tuple[StepRef, ...]
    datasets: tuple[DatasetAudit, ...]
    assembly: tuple[AssemblyAuditRecord, ...] = ()
    portfolios: tuple[PortfolioRecord, ...]
    artifacts: tuple[Artifact, ...]
    exit_code: int
    manifest_sha256: str = Field(default="")


def versions(solver: str, solver_version: str, packages: Mapping[str, str], workers: Sequence[WorkerRecord] = ()) -> VersionInfo:
    """Collect the versions that matter for reproducibility."""
    return VersionInfo(
        python=platform.python_version(),
        cvxpy=distribution_version("cvxpy"),
        numpy=distribution_version("numpy"),
        pandas=distribution_version("pandas"),
        solver=solver,
        solver_version=solver_version,
        packages=dict(packages),
        workers=tuple(workers),
    )


def solved_record(result: PortfolioResult, violation_tol: float, *, solve_order: str | None = None) -> PortfolioRecord:
    """The manifest record for a portfolio that produced orders; ``violation_tol`` is what its verification was held to."""
    solution, report, drift = result.solution, result.report, result.drift
    gross = sum((notional for notional in result.orders["notional"]), start=0)
    return PortfolioRecord(
        portfolio_id=result.portfolio_id,
        status="solved",
        solve_order=solve_order,
        predecessors=len(result.chain_state.predecessors),
        rules=result.rule_audit,
        constraints=solution.constraints,
        problem_spec_sha256=result.spec.content_hash(),
        chain_inputs_sha256=result.chain_state.content_hash(),
        solve=SolveRecord(
            solver=solution.solver,
            solver_version=solution.solver_version,
            status=str(solution.status),
            iterations=solution.iterations,
            objective_value=solution.objective,
            solve_time_s=solution.solve_time_s,
        ),
        check=CheckRecord(
            tolerance=violation_tol,
            max_violation=report.max_violation,
            violated=report.violated,
            objective_gap=report.objective_gap,
            objective_passed=report.objective_passed,
            unverified=report.unverified,
            passed=report.passed,
        ),
        drift=DriftRecord(max_weight_error=drift.max_weight_error, tolerance=drift.tolerance, dropped_orders=drift.dropped_orders, passed=drift.passed),
        orders=OrdersRecord(count=len(result.orders), sha256=frame_sha256(result.orders.drop(columns=["run_id"]), ("portfolio_id", "security_id")), gross_notional=str(gross)),
    )


def failed_record(failure: PortfolioFailure, rules: tuple[RuleAuditRecord, ...] = (), *, solve_order: str | None = None, predecessors: int | None = None) -> PortfolioRecord:
    """The manifest record for a portfolio that did not produce orders."""
    return PortfolioRecord(
        portfolio_id=failure.portfolio_id,
        status="failed",
        solve_order=solve_order,
        predecessors=predecessors,
        rules=rules,
        failure_stage=failure.stage,
        error=f"{failure.error_type}: {failure.message}",
    )


def finalize(manifest: RunManifest) -> RunManifest:
    """Stamp the manifest with the hash of everything else in it."""
    without_hash = manifest.model_dump(mode="json")
    without_hash["manifest_sha256"] = ""
    return RunManifest.model_validate_json(json.dumps({**without_hash, "manifest_sha256": json_sha256(without_hash)}))


def write_manifest(manifest: RunManifest, directory: Path) -> Path:
    """Write the manifest atomically and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / MANIFEST_FILENAME
    temp = target.with_name(f".{MANIFEST_FILENAME}.tmp")
    try:
        temp.write_text(manifest.model_dump_json(indent=2) + "\n")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def load_manifest(text: str) -> RunManifest:
    """Parse and validate a manifest; the stored hash must match the content."""
    manifest = RunManifest.model_validate_json(text)
    expected = finalize(manifest).manifest_sha256
    if manifest.manifest_sha256 != expected:
        msg = f"manifest hash {manifest.manifest_sha256[:12]} does not match its content ({expected[:12]})"
        raise ValueError(msg)
    return manifest


def diff_manifests(left: RunManifest, right: RunManifest) -> list[str]:
    """Name the first stage at which two runs diverge, overall and per portfolio."""
    lines: list[str] = []
    if left.config.sha256 != right.config.sha256:
        lines.append("config: resolved config differs")
    if left.git_sha != right.git_sha:
        lines.append(f"code: git sha {left.git_sha[:12]} vs {right.git_sha[:12]}")
    if _versions_identity(left) != _versions_identity(right):
        lines.append("versions: library, solver, or step-package versions differ")
    left_datasets = {d.name: d.content_sha256 for d in left.datasets}
    right_datasets = {d.name: d.content_sha256 for d in right.datasets}
    lines.extend(f"datasets: {name} content differs" for name in sorted(set(left_datasets) | set(right_datasets)) if left_datasets.get(name) != right_datasets.get(name))
    if _assembly_identity(left) != _assembly_identity(right):
        lines.append("assembly: steps, their parameters, or their effect on the datasets differ")
    right_portfolios = {p.portfolio_id: p for p in right.portfolios}
    for portfolio in left.portfolios:
        other = right_portfolios.get(portfolio.portfolio_id)
        if other is None:
            lines.append(f"{portfolio.portfolio_id}: missing from the second manifest")
            continue
        stage = _first_divergence(portfolio, other)
        if stage is not None:
            lines.append(f"{portfolio.portfolio_id}: first divergence at {stage}")
    lines.extend(f"{portfolio_id}: missing from the first manifest" for portfolio_id in sorted(set(right_portfolios) - {p.portfolio_id for p in left.portfolios}))
    return lines


def _versions_identity(manifest: RunManifest) -> tuple[dict[str, object], frozenset[WorkerEnvironment]]:
    """Versions compare on what determines results: the libraries and every worker environment, not the hosts that happened to run them."""
    libraries: dict[str, object] = {str(key): value for key, value in manifest.versions.model_dump(exclude={"workers"}).items()}
    return (libraries, frozenset(worker.environment for worker in manifest.versions.workers))


def _assembly_identity(manifest: RunManifest) -> list[tuple[str, str, str, dict[str, int], dict[str, tuple[str, ...]]]]:
    return [(a.qualname, a.source_sha256, a.params_sha256, a.rows_out, a.columns_added) for a in manifest.assembly]


def _first_divergence(left: PortfolioRecord, right: PortfolioRecord) -> str | None:
    if left.status != right.status:
        return f"status ({left.status} vs {right.status})"
    if [(r.qualname, r.source_sha256, r.params_sha256, r.rows_out) for r in left.rules] != [(r.qualname, r.source_sha256, r.params_sha256, r.rows_out) for r in right.rules]:
        return "rules"
    if left.problem_spec_sha256 != right.problem_spec_sha256 or left.chain_inputs_sha256 != right.chain_inputs_sha256:
        return "spec"
    if (left.solve is None) != (right.solve is None) or (left.solve is not None and right.solve is not None and left.solve.objective_value != right.solve.objective_value):
        return "solve"
    if (left.orders is None) != (right.orders is None) or (left.orders is not None and right.orders is not None and left.orders.sha256 != right.orders.sha256):
        return "orders"
    return None


def created_at(now: datetime) -> AwareDatetime:
    """Normalize the injected clock's reading for the manifest."""
    if now.tzinfo is None:
        msg = "the clock must return timezone-aware datetimes"
        raise ValueError(msg)
    return now
