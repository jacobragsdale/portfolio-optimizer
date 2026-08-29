"""The per-portfolio pipeline every backend runs: slice, rules, build, and — in ``parallel`` mode — solve, verify, orders.

:func:`build_task` and :func:`full_task` are the worker entry points. Each receives the run's shared
data (the backend resolves its handle on the worker), resolves the config by name once per process,
runs the pipeline, and returns a
:class:`TaskOutput` stamped with the fingerprint of the process that did the work. A task never raises:
every failure becomes a :class:`PortfolioFailure` naming its stage, so the scheduler can apply
``on_error``. The sequential mode calls the same pipeline functions in the main process with a live
:class:`SolveContext`.
"""

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from portfolio_optimizer.config.resolve import ResolvedConfig, ResolvedStep, resolve_config
from portfolio_optimizer.domain.data import PortfolioData, PortfolioDataError
from portfolio_optimizer.domain.results import (
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
from portfolio_optimizer.engine.backends import SharedRunData, TaskOutput
from portfolio_optimizer.engine.build import build_problem_spec
from portfolio_optimizer.engine.check import verify
from portfolio_optimizer.engine.environment import IMAGE_DIGEST_VARIABLE, WorkerEnvironment, environment_for, host_name
from portfolio_optimizer.engine.load import slice_portfolio
from portfolio_optimizer.engine.orders import rounding_drift, solution_to_orders
from portfolio_optimizer.engine.pipeline import apply_rules
from portfolio_optimizer.engine.solve import solve

log = logging.getLogger(__name__)

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


@dataclass(frozen=True, slots=True, eq=False)
class BuildResult:
    """A built portfolio: pure data, safe to send back from a worker."""

    portfolio_id: PortfolioId
    spec: ProblemSpec
    order_inputs: OrderInputs
    rule_audit: tuple[RuleAuditRecord, ...]


# --- the pipeline, shared by every mode ---


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


def failure(portfolio_id: str, stage: str, error: BaseException) -> PortfolioFailure:
    """Record ``error`` as the failure of ``portfolio_id`` at ``stage``."""
    return PortfolioFailure(portfolio_id=portfolio_id, stage=stage, error_type=type(error).__name__, message=str(error))


def slice_and_build(shared: SharedRunData, resolved: ResolvedConfig, portfolio_id: PortfolioId, ctx: SolveContext | None) -> BuildResult | PortfolioFailure:
    """Slice the portfolio's bundle from the shared data, apply rules, and build its spec; a failure names its stage."""
    try:
        data = slice_portfolio(shared.assembled, portfolio_id)
    except (PortfolioDataError, ValueError) as error:
        return failure(portfolio_id, "slice", error)
    try:
        return build_portfolio(data, resolved, ctx)
    except Exception as error:  # noqa: BLE001  # recorded per portfolio; on_error decides what happens next
        return failure(portfolio_id, "build", error)


def finish_or_fail(built: BuildResult, resolved: ResolvedConfig, ctx: SolveContext, run_id: str) -> Outcome:
    """Solve and finish, recording any failure at stage ``solve``."""
    try:
        result = finish_portfolio(built, resolved, ctx, run_id)
    except Exception as error:  # noqa: BLE001  # recorded per portfolio; on_error decides what happens next
        log.error("portfolio failed", extra={"run_id": run_id, "portfolio_id": built.portfolio_id, "stage": "solve", "error": type(error).__name__})
        return failure(built.portfolio_id, "solve", error)
    log.info("portfolio solved", extra={"run_id": run_id, "portfolio_id": built.portfolio_id, "stage": "solve", "orders": len(result.orders)})
    return result


# --- worker entry points ---

_RESOLVED: dict[str, ResolvedConfig] = {}
"""Per process: the resolved config of every run this process has worked on, by config hash."""


def resolved_for(shared: SharedRunData) -> ResolvedConfig:
    """Resolve the run's step names in this process, once per config; function objects never cross a process boundary.

    The cache is keyed by the config hash but trusts only a cached entry whose config is equal to the
    one asked for, so a caller that lies about the hash gets a fresh resolution rather than another run's.
    """
    resolved = _RESOLVED.get(shared.config_sha256)
    if resolved is None or resolved.config != shared.config:
        resolved = _RESOLVED[shared.config_sha256] = resolve_config(shared.config, shared.config_sha256)
    return resolved


def worker_environment(shared: SharedRunData) -> WorkerEnvironment:
    """This process's fingerprint for the run; the image digest is read from this process's own environment, which is the point."""
    return environment_for(shared.config, cwd=Path.cwd(), image_digest=os.environ.get(IMAGE_DIGEST_VARIABLE))


def build_task(shared: SharedRunData, portfolio_id: PortfolioId) -> TaskOutput[BuildResult]:
    """Worker entry for ``parallel_build_sequential_solve``: slice, rules, and build with no context."""
    return _task(shared, portfolio_id, lambda data, resolved: slice_and_build(data, resolved, portfolio_id, ctx=None))


def full_task(shared: SharedRunData, portfolio_id: PortfolioId) -> TaskOutput[PortfolioResult]:
    """Worker entry for ``parallel``: the whole pipeline with an empty context."""

    def pipeline(data: SharedRunData, resolved: ResolvedConfig) -> Outcome:
        built = slice_and_build(data, resolved, portfolio_id, ctx=None)
        if isinstance(built, PortfolioFailure):
            return built
        return finish_or_fail(built, resolved, SolveContext(), data.run_id)

    return _task(shared, portfolio_id, pipeline)


def _task[T](data: SharedRunData, portfolio_id: PortfolioId, pipeline: Callable[[SharedRunData, ResolvedConfig], T | PortfolioFailure]) -> TaskOutput[T]:
    environment = worker_environment(data)
    try:
        resolved = resolved_for(data)
    except Exception as error:  # noqa: BLE001  # a step package missing on this worker is a worker failure, not a crash
        return TaskOutput(outcome=failure(portfolio_id, "worker", error), environment=environment, host=host_name())
    return TaskOutput(outcome=pipeline(data, resolved), environment=environment, host=host_name())
