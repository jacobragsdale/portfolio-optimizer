"""Schedule portfolios through build and solve according to the execution mode, then publish and record.

The three modes differ only in *where* build and solve happen; every portfolio goes through the
same ``build_portfolio`` → ``finish_portfolio`` functions, results are consumed in configured
solve order regardless of completion order, and the manifest is written whatever happens.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

from portfolio_optimizer.config.models import RunConfig
from portfolio_optimizer.config.resolve import ResolvedConfig, ResolvedStep, resolve_config
from portfolio_optimizer.cvx.adapter import solver_version
from portfolio_optimizer.domain.data import IoContext, PortfolioData, PortfolioDataError
from portfolio_optimizer.domain.results import (
    Artifact,
    AssemblyAuditRecord,
    ConstraintReport,
    DriftReport,
    OrderInputs,
    PortfolioFailure,
    PortfolioResult,
    ProblemSpec,
    RuleAuditRecord,
    SolveContext,
    StepRef,
    Tolerances,
    derive_chain_state,
)
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.build import build_problem_spec
from portfolio_optimizer.engine.check import verify
from portfolio_optimizer.engine.hashing import file_sha256
from portfolio_optimizer.engine.load import AssembledDatasets, DatasetAudit, assemble, load_datasets, slice_portfolio
from portfolio_optimizer.engine.manifest import (
    ConfigInfo,
    GitInfo,
    RunManifest,
    artifact_records,
    assembly_records,
    created_at,
    dataset_records,
    failed_record,
    finalize,
    package_versions,
    solved_record,
    step_records,
    versions,
    write_manifest,
)
from portfolio_optimizer.engine.orders import rounding_drift, solution_to_orders
from portfolio_optimizer.engine.pipeline import apply_rules
from portfolio_optimizer.engine.solve import solve

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_PORTFOLIO_FAILED = 1
EXIT_INPUT_REJECTED = 2
EXIT_INFRASTRUCTURE = 3

type Outcome = PortfolioResult | PortfolioFailure


class VerificationError(RuntimeError):
    """The independent check disagreed with the solver."""

    def __init__(self, report: ConstraintReport) -> None:
        self.report = report
        super().__init__(f"verification failed: violated {list(report.violated)}, objective gap {report.objective_gap:.3e}")


class DriftError(RuntimeError):
    """Rounding to whole shares moved the portfolio further than lot sizes can explain."""

    def __init__(self, report: DriftReport) -> None:
        self.report = report
        super().__init__(f"rounding drift {report.max_weight_error:.3e} exceeds tolerance {report.tolerance:.3e}")


@dataclass(frozen=True, slots=True)
class BuildPayload:
    """Everything a worker needs to build one portfolio; picklable."""

    portfolio_id: PortfolioId
    data: PortfolioData
    config: RunConfig
    config_sha256: str
    run_id: str


@dataclass(frozen=True, slots=True, eq=False)
class BuildResult:
    """A built portfolio: pure data, safe to send back from a worker."""

    portfolio_id: PortfolioId
    spec: ProblemSpec
    order_inputs: OrderInputs
    rule_audit: tuple[RuleAuditRecord, ...]


@dataclass(frozen=True, slots=True)
class RunReport:
    """What a run produced."""

    run_id: str
    outcomes: tuple[Outcome, ...]
    manifest: RunManifest
    manifest_path: Path
    artifacts: tuple[Artifact, ...]
    exit_code: int

    @property
    def solved(self) -> tuple[PortfolioResult, ...]:
        """Portfolios that produced orders."""
        return tuple(outcome for outcome in self.outcomes if isinstance(outcome, PortfolioResult))

    @property
    def failed(self) -> tuple[PortfolioFailure, ...]:
        """Portfolios that did not."""
        return tuple(outcome for outcome in self.outcomes if isinstance(outcome, PortfolioFailure))


# --- the per-portfolio pipeline, shared by every mode and by workers ---


def build_portfolio(data: PortfolioData, resolved: ResolvedConfig, ctx: SolveContext | None) -> BuildResult:
    """Apply rules and build the spec."""
    ruled, audit = apply_rules(data, resolved.rules, ctx)
    output = build_problem_spec(ruled)
    return BuildResult(portfolio_id=data.portfolio_id, spec=output.spec, order_inputs=output.order_inputs, rule_audit=audit)


def finish_portfolio(built: BuildResult, resolved: ResolvedConfig, ctx: SolveContext, run_id: str) -> PortfolioResult:
    """Solve, verify independently, round to orders, and bound the rounding drift."""
    chain = derive_chain_state(ctx, built.spec.security_ids)
    solution = solve(built.spec, chain, resolved)
    post = resolved.config.post_solve
    report = verify(
        built.spec,
        solution,
        chain,
        step_refs(resolved.terms),
        step_refs(resolved.constraints),
        Tolerances(eq=post.violation_tol, ineq=post.violation_tol, obj_rel=post.objective_rel_tol, obj_abs=post.objective_abs_tol),
    )
    if not report.passed:
        raise VerificationError(report)
    orders = solution_to_orders(built.spec, solution, built.order_inputs, run_id=run_id)
    drift = rounding_drift(built.spec, solution, orders, built.order_inputs)
    if not drift.passed:
        raise DriftError(drift)
    return PortfolioResult(portfolio_id=built.portfolio_id, spec=built.spec, solution=solution, report=report, orders=orders, rule_audit=built.rule_audit, chain_state=chain, drift=drift)


def step_refs(steps: Sequence[ResolvedStep]) -> tuple[StepRef, ...]:
    """Reduce resolved steps to the data the verifier and manifest need."""
    return tuple(StepRef(step.qualname, step.params.model_dump(mode="json") if step.params is not None else {}) for step in steps)


def _failure(portfolio_id: str, stage: str, error: BaseException) -> PortfolioFailure:
    return PortfolioFailure(portfolio_id=portfolio_id, stage=stage, error_type=type(error).__name__, message=str(error))


def build_task(payload: BuildPayload) -> BuildResult | PortfolioFailure:
    """Worker entry for ``parallel_build_sequential_solve``: rules and build, no context."""
    try:
        resolved = resolve_config(payload.config, payload.config_sha256)
        return build_portfolio(payload.data, resolved, ctx=None)
    except Exception as error:  # noqa: BLE001  # a worker must report, not raise, so the scheduler can apply on_error
        return _failure(payload.portfolio_id, "build", error)


def full_task(payload: BuildPayload) -> PortfolioResult | PortfolioFailure:
    """Worker entry for ``parallel``: the whole pipeline with an empty context."""
    try:
        resolved = resolve_config(payload.config, payload.config_sha256)
        built = build_portfolio(payload.data, resolved, ctx=None)
    except Exception as error:  # noqa: BLE001  # see build_task
        return _failure(payload.portfolio_id, "build", error)
    try:
        return finish_portfolio(built, resolved, SolveContext(), payload.run_id)
    except Exception as error:  # noqa: BLE001  # see build_task
        return _failure(payload.portfolio_id, "solve", error)


# --- the run ---


class InputRejectedError(ValueError):
    """The configured inputs could not be loaded or assembled; nothing was solved."""


def run(resolved: ResolvedConfig, io: IoContext, *, git: GitInfo, config_path: str, settings: Mapping[str, str]) -> RunReport:
    """Execute the run end to end and write its manifest. Raises only when nothing could start."""
    config = resolved.config
    log.info("run starting", extra={"run_id": io.run_id, "mode": config.execution.mode, "stage": "load"})
    try:
        loaded = load_datasets(resolved, data_root=io.data_root, run_id=io.run_id)
        assembled = assemble(loaded, resolved)
    except (ValueError, KeyError) as error:
        msg = f"inputs rejected: {error}"
        raise InputRejectedError(msg) from error
    bundles, outcomes = _slice_all(assembled)
    outcomes.update(_execute(config, resolved, bundles, assembled.portfolio_ids, io))
    ordered = tuple(outcomes[portfolio_id] for portfolio_id in assembled.portfolio_ids)
    artifacts, publish_error = _persist_and_publish(ordered, resolved, io)
    exit_code = _exit_code(ordered, publish_error)
    manifest = _manifest(resolved, io, git, config_path, settings, loaded.audits, assembled.audits, ordered, artifacts, exit_code, publish_error)
    manifest_path = write_manifest(manifest, io.output_dir / io.run_id)
    log.info("run finished", extra={"run_id": io.run_id, "stage": "manifest", "exit_code": exit_code})
    return RunReport(run_id=io.run_id, outcomes=ordered, manifest=manifest, manifest_path=manifest_path, artifacts=artifacts, exit_code=exit_code)


def _slice_all(assembled: AssembledDatasets) -> tuple[dict[PortfolioId, PortfolioData], dict[PortfolioId, Outcome]]:
    bundles: dict[PortfolioId, PortfolioData] = {}
    failures: dict[PortfolioId, Outcome] = {}
    for portfolio_id in assembled.portfolio_ids:
        try:
            bundles[portfolio_id] = slice_portfolio(assembled, portfolio_id)
        except (PortfolioDataError, ValueError) as error:
            failures[portfolio_id] = _failure(portfolio_id, "slice", error)
    return bundles, failures


def _execute(config: RunConfig, resolved: ResolvedConfig, bundles: Mapping[PortfolioId, PortfolioData], order: Sequence[PortfolioId], io: IoContext) -> dict[PortfolioId, Outcome]:
    ids = [portfolio_id for portfolio_id in order if portfolio_id in bundles]
    mode = config.execution.mode
    if mode == "sequential":
        return _run_sequential(ids, bundles, resolved, io)
    payloads = {portfolio_id: BuildPayload(portfolio_id, bundles[portfolio_id], config, resolved.config_sha256, io.run_id) for portfolio_id in ids}
    if mode == "parallel":
        return _collect(ids, payloads, full_task, config)
    built = _collect(ids, payloads, build_task, config)
    return _solve_sequentially(ids, built, resolved, io)


def _run_sequential(ids: Sequence[PortfolioId], bundles: Mapping[PortfolioId, PortfolioData], resolved: ResolvedConfig, io: IoContext) -> dict[PortfolioId, Outcome]:
    outcomes: dict[PortfolioId, Outcome] = {}
    ctx = SolveContext()
    for portfolio_id in ids:
        if _should_skip(outcomes, resolved.config):
            outcomes[portfolio_id] = PortfolioFailure(portfolio_id, "skipped", "SkippedAfterFailure", "not processed because an earlier portfolio failed and on_error is fail_fast")
            continue
        try:
            built = build_portfolio(bundles[portfolio_id], resolved, ctx)
        except Exception as error:  # noqa: BLE001  # recorded per portfolio; on_error decides what happens next
            outcomes[portfolio_id] = _failure(portfolio_id, "build", error)
            continue
        outcome = _finish_or_fail(built, resolved, ctx, io.run_id)
        outcomes[portfolio_id] = outcome
        if isinstance(outcome, PortfolioResult):
            ctx = ctx.with_result(outcome)
    return outcomes


def _solve_sequentially(ids: Sequence[PortfolioId], built: Mapping[PortfolioId, BuildResult | PortfolioFailure], resolved: ResolvedConfig, io: IoContext) -> dict[PortfolioId, Outcome]:
    outcomes: dict[PortfolioId, Outcome] = {}
    ctx = SolveContext()
    for portfolio_id in ids:
        build_outcome = built[portfolio_id]
        if isinstance(build_outcome, PortfolioFailure):
            outcomes[portfolio_id] = build_outcome
            continue
        if _should_skip(outcomes, resolved.config):
            outcomes[portfolio_id] = PortfolioFailure(portfolio_id, "skipped", "SkippedAfterFailure", "not solved because an earlier portfolio failed and on_error is fail_fast")
            continue
        outcome = _finish_or_fail(build_outcome, resolved, ctx, io.run_id)
        outcomes[portfolio_id] = outcome
        if isinstance(outcome, PortfolioResult):
            ctx = ctx.with_result(outcome)
    return outcomes


def _finish_or_fail(built: BuildResult, resolved: ResolvedConfig, ctx: SolveContext, run_id: str) -> Outcome:
    try:
        result = finish_portfolio(built, resolved, ctx, run_id)
    except Exception as error:  # noqa: BLE001  # recorded per portfolio; on_error decides what happens next
        log.error("portfolio failed", extra={"run_id": run_id, "portfolio_id": built.portfolio_id, "stage": "solve", "error": type(error).__name__})
        return _failure(built.portfolio_id, "solve", error)
    log.info("portfolio solved", extra={"run_id": run_id, "portfolio_id": built.portfolio_id, "stage": "solve", "orders": len(result.orders)})
    return result


def _should_skip(outcomes: Mapping[PortfolioId, Outcome], config: RunConfig) -> bool:
    return config.execution.on_error == "fail_fast" and any(isinstance(outcome, PortfolioFailure) for outcome in outcomes.values())


def _collect[T](
    ids: Sequence[PortfolioId], payloads: Mapping[PortfolioId, BuildPayload], task: Callable[[BuildPayload], T | PortfolioFailure], config: RunConfig
) -> dict[PortfolioId, T | PortfolioFailure]:
    """Submit every task, then consume results in configured order so completion order never matters."""
    outcomes: dict[PortfolioId, T | PortfolioFailure] = {}
    with _executor(config) as executor:
        futures: dict[PortfolioId, Future[T | PortfolioFailure]] = {portfolio_id: executor.submit(task, payloads[portfolio_id]) for portfolio_id in ids}
        for portfolio_id in ids:
            if config.execution.on_error == "fail_fast" and any(isinstance(outcome, PortfolioFailure) for outcome in outcomes.values()):
                futures[portfolio_id].cancel()
                outcomes[portfolio_id] = PortfolioFailure(portfolio_id, "skipped", "SkippedAfterFailure", "not processed because an earlier portfolio failed and on_error is fail_fast")
                continue
            try:
                outcomes[portfolio_id] = futures[portfolio_id].result()
            except Exception as error:  # noqa: BLE001  # a worker that died (e.g. unpicklable result) is a per-portfolio failure
                outcomes[portfolio_id] = _failure(portfolio_id, "worker", error)
    return outcomes


def _executor(config: RunConfig) -> Executor:
    if config.execution.executor == "thread":
        return ThreadPoolExecutor(max_workers=config.execution.max_workers)
    return ProcessPoolExecutor(max_workers=config.execution.max_workers, mp_context=get_context("spawn"))


# --- persistence, publication, manifest ---


def _persist_and_publish(outcomes: Sequence[Outcome], resolved: ResolvedConfig, io: IoContext) -> tuple[tuple[Artifact, ...], PortfolioFailure | None]:
    artifacts: list[Artifact] = []
    solved = [outcome for outcome in outcomes if isinstance(outcome, PortfolioResult)]
    run_dir = io.output_dir / io.run_id
    try:
        for result in solved:
            artifacts.extend(_persist_result(result, run_dir))
        if not solved:
            log.error("no portfolio solved; nothing published", extra={"run_id": io.run_id, "stage": "sink"})
            return tuple(artifacts), None
        orders = pd.concat([result.orders for result in solved], ignore_index=True).sort_values(["portfolio_id", "security_id"], kind="stable").reset_index(drop=True)
        published = resolved.sink.invoke(orders=orders, io=io)
        if not isinstance(published, tuple):
            msg = f"sink {resolved.sink.qualname!r} returned {type(published).__name__}, expected a tuple of Artifact"
            raise TypeError(msg)  # the sink contract violation is reported like any other sink failure
        for item in published:
            if not isinstance(item, Artifact):
                msg = f"sink {resolved.sink.qualname!r} returned a {type(item).__name__}, expected Artifact"
                raise TypeError(msg)  # see above
            artifacts.append(item)
    except Exception as error:  # noqa: BLE001  # the manifest must still be written; the exit code carries the failure
        log.error("publishing failed", extra={"run_id": io.run_id, "stage": "sink", "error": type(error).__name__})
        return tuple(artifacts), _failure("*", "sink", error)
    return tuple(artifacts), None


def _persist_result(result: PortfolioResult, run_dir: Path) -> list[Artifact]:
    written: list[Artifact] = []
    for subdir, writer in (("problem_specs", result.spec.to_npz), ("solutions", result.solution.to_npz), ("chain", result.chain_state.to_npz)):
        directory = run_dir / subdir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{result.portfolio_id}.npz"
        writer(path)
        written.append(Artifact(path=str(path), sha256=file_sha256(path), size_bytes=path.stat().st_size))
    return written


def _exit_code(outcomes: Sequence[Outcome], publish_error: PortfolioFailure | None) -> int:
    if publish_error is not None:
        return EXIT_INFRASTRUCTURE
    if not outcomes or any(isinstance(outcome, PortfolioFailure) for outcome in outcomes):
        return EXIT_PORTFOLIO_FAILED
    return EXIT_OK


def _manifest(
    resolved: ResolvedConfig,
    io: IoContext,
    git: GitInfo,
    config_path: str,
    settings: Mapping[str, str],
    audits: Sequence[DatasetAudit],
    assembly_audits: Sequence[AssemblyAuditRecord],
    outcomes: Sequence[Outcome],
    artifacts: Sequence[Artifact],
    exit_code: int,
    publish_error: PortfolioFailure | None,
) -> RunManifest:
    config = resolved.config
    post = config.post_solve
    records = [solved_record(o, o.report, o.drift, post.violation_tol, post.violation_tol) if isinstance(o, PortfolioResult) else failed_record(o) for o in outcomes]
    if publish_error is not None:
        records.append(failed_record(publish_error))
    solved = [o for o in outcomes if isinstance(o, PortfolioResult)]
    solver_ver = solved[0].solution.solver_version if solved else solver_version(config.solver.name)
    manifest = RunManifest(
        run_id=io.run_id,
        run_name=config.run.name,
        created_at_utc=created_at(io.clock.now()),
        as_of=config.run.as_of,
        git_sha=git.sha,
        git_dirty=git.dirty,
        execution_mode=config.execution.mode,
        versions=versions(config.solver.name, solver_ver, package_versions(step.qualname.partition(":")[0] for step in resolved.all_steps if step.is_external)),
        config=ConfigInfo(path=config_path, sha256=resolved.config_sha256, resolved=config.model_dump(mode="json")),
        settings=dict(settings),
        terms=step_records(step_refs(resolved.terms)),
        constraints=step_records(step_refs(resolved.constraints)),
        datasets=dataset_records(audits),
        assembly=assembly_records(assembly_audits),
        portfolios=tuple(records),
        artifacts=artifact_records(artifacts),
        exit_code=exit_code,
    )
    return finalize(manifest)
