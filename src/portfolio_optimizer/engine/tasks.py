"""The tasks every run submits to its cluster: a probe of each worker, then per portfolio build, summarize, solve, contribute.

:func:`probe_task` runs on every worker the run starts with, before any data is shared: it resolves
the config there — every step importable, the solver installed — and reports the fingerprint, so a
worker that cannot do the run's work stops the run before it does any. :func:`build_task` then runs
for every portfolio, chain-free and in parallel: slice, rules, the
solve-order key, and the spec. Its result stays on the worker; :func:`summarize` sends the main
process only what it needs to derive the schedule — the key, the tradable securities, the spec hash,
the rule audit — stamped with the build's environment. :func:`solve_task` then runs where the build
lives, once the portfolio's predecessors have contributed: it folds their trades on the side the run
couples through into a :class:`ChainState`, solves, verifies, rounds, and verifies the rounded book
against the same constraints. :func:`contribution`
reduces a result to the order rows on that side a dependent needs. Each task receives the run's shared data (the backend resolves its handle on
the worker) and resolves the config by name once per process. A task never raises for a portfolio's
own failure: it returns a :class:`PortfolioFailure` naming the stage, so ``on_error`` can be applied.
"""

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_optimizer.config.models import RunConfig
from portfolio_optimizer.config.resolve import ResolvedConfig, resolve_config
from portfolio_optimizer.config.steps import ResolvedStep
from portfolio_optimizer.domain.constraints import ConstraintSpecError, check_against_spec, consumed_securities, parse_constraints
from portfolio_optimizer.domain.data import PortfolioData, PortfolioDataError
from portfolio_optimizer.domain.objective import TermSpecError
from portfolio_optimizer.domain.order_flow import OrderFlowProfile
from portfolio_optimizer.domain.results import (
    RUN_SCOPED,
    ChainState,
    ConstraintReport,
    Contribution,
    DriftReport,
    Flags,
    OrderInputs,
    PortfolioFailure,
    PortfolioResult,
    ProblemSpec,
    RuleAuditRecord,
    Tolerances,
)
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.backends import SharedRunData, TaskOutput
from portfolio_optimizer.engine.build import BuildError, order_inputs
from portfolio_optimizer.engine.check import constraints_of, verify
from portfolio_optimizer.engine.environment import IMAGE_DIGEST_VARIABLE, WorkerEnvironment, environment_for, host_name
from portfolio_optimizer.engine.load import slice_portfolio
from portfolio_optimizer.engine.orders import executed_solution, rounding_drift, solution_to_orders
from portfolio_optimizer.engine.pipeline import apply_rules
from portfolio_optimizer.engine.solve import solve
from portfolio_optimizer.engine.timing import SpanRecorder

log = logging.getLogger(__name__)

type Outcome = PortfolioResult | PortfolioFailure

SKIPPED_ERROR = "SkippedAfterFailure"


class VerificationError(RuntimeError):
    """The independent check disagreed with the solver, or the executed orders breach a constraint the solved weights kept."""

    def __init__(self, report: ConstraintReport, *, what: str = "solution") -> None:
        self.report = report
        super().__init__(f"{what} failed verification: violated {list(report.violated)}, objective gap {report.objective_gap:.3e}")


class DriftError(RuntimeError):
    """Rounding to whole shares moved the portfolio further than lot sizes can explain."""

    def __init__(self, report: DriftReport) -> None:
        self.report = report
        super().__init__(f"rounding drift {report.max_weight_error:.3e} exceeds tolerance {report.tolerance:.3e}")


class SolveOrderError(ValueError):
    """The solve-order step returned something other than a finite ``Decimal``."""


class ChainInvariantError(RuntimeError):
    """A portfolio traded a security outside its tradable set, which the dependency graph could not have seen."""


class SideInvariantError(RuntimeError):
    """A portfolio produced an order on a side the run does not trade."""


@dataclass(frozen=True, slots=True, eq=False)
class BuildResult:
    """A built portfolio: pure data that stays on the worker that built it."""

    portfolio_id: PortfolioId
    spec: ProblemSpec
    order_inputs: OrderInputs
    rule_audit: tuple[RuleAuditRecord, ...]
    solve_order: Decimal
    tradable: tuple[str, ...]
    consumes: tuple[str, ...]
    """The securities predecessors' trades can reach this portfolio through — the schedule's consume side, at most ``tradable`` and empty when nothing here reads the chain."""

    constraints: pd.DataFrame
    """This portfolio's constraint rows as the rules left them, carried to the solve step that interprets them."""

    extras: Mapping[str, pd.DataFrame]
    """The run's extra datasets as the rules left them, carried past the spec to the solve step for the same reason: the engine does not know what they mean."""


@dataclass(frozen=True, slots=True)
class BuildSummary:
    """What the main process learns from a build: enough to place the portfolio in the schedule."""

    portfolio_id: PortfolioId
    solve_order: Decimal
    tradable: tuple[str, ...]
    consumes: tuple[str, ...]
    spec_sha256: str
    rule_audit: tuple[RuleAuditRecord, ...]


# --- the pipeline ---


def build_portfolio(data: PortfolioData, resolved: ResolvedConfig, fallback_solve_order: Decimal, recorder: SpanRecorder) -> BuildResult:
    """Apply rules, compute the solve-order key, run the build step, and read the constraint declarations, timing each phase onto ``recorder``.

    Typed constraint rows are parsed here — a malformed one, or one naming a column the spec does not
    carry, fails the portfolio at stage ``build``, before any solve is scheduled on it, as does a term
    that reads a column this spec lacks — and the
    consume set the schedule needs is derived from them: empty when nothing reads the chain, their
    scopes when chain-reading constraints are the only readers, the whole tradable set when anything
    opaque (a chain-aware term, a solve step that is not the shipped one) might.
    """
    with recorder.span("build:rules"):
        ruled, audit = apply_rules(data, resolved.rules)
    key = fallback_solve_order if resolved.solve_order is None else solve_order_key(resolved.solve_order, ruled)
    with recorder.span("build:spec"):
        spec = build_spec(resolved.build, ruled)
        inputs = order_inputs(ruled, spec)
    parsed = parse_constraints(ruled.constraints)
    if parsed is None and not ruled.constraints.empty and resolved.shipped_solve:
        msg = f"the {len(ruled.constraints)} constraint row(s) carry no `kind` column; the shipped cvxpy step interprets typed rows only"
        raise ConstraintSpecError(msg)
    if parsed is not None:
        problems = list(check_against_spec(parsed.typed, spec))
        if problems:
            msg = "typed constraint(s) cannot apply to this problem: " + "; ".join(problems)
            raise ConstraintSpecError(msg)
    wanting = [f"{term.name}: {problem}" for term in resolved.terms for problem in term.requirements(spec)]
    if wanting:
        msg = "objective term(s) cannot apply to this problem: " + "; ".join(wanting)
        raise TermSpecError(msg)
    consumes = consumed_securities(parsed, spec, resolved.profile, chain_aware_terms=bool(resolved.chain_aware_terms), opaque_solve=not resolved.shipped_solve)
    return BuildResult(
        portfolio_id=data.portfolio_id,
        spec=spec,
        order_inputs=inputs,
        rule_audit=audit,
        solve_order=key,
        tradable=tradable_ids(resolved.profile, spec),
        consumes=consumes,
        constraints=ruled.constraints,
        extras=ruled.extras,
    )


def build_spec(step: ResolvedStep, data: PortfolioData) -> ProblemSpec:
    """Run the build step and insist on a ``ProblemSpec`` for this portfolio."""
    result = step.invoke(data=data)
    if not isinstance(result, ProblemSpec):
        msg = f"build step {step.qualname!r} returned {type(result).__name__}, expected ProblemSpec"
        raise BuildError(msg)
    if result.portfolio_id != data.portfolio_id:
        msg = f"build step {step.qualname!r} returned a spec for {result.portfolio_id!r} while building {data.portfolio_id!r}"
        raise BuildError(msg)
    return result


def solve_order_key(step: ResolvedStep, data: PortfolioData) -> Decimal:
    """Run the solve-order step and insist on a finite ``Decimal``: the key is sorted and recorded, so it must be exact."""
    value = step.invoke(data=data)
    if isinstance(value, bool) or not isinstance(value, Decimal | int):
        msg = f"solve-order step {step.qualname!r} returned {type(value).__name__}, expected Decimal"
        raise SolveOrderError(msg)
    key = Decimal(value)
    if not key.is_finite():
        msg = f"solve-order step {step.qualname!r} returned {key}, expected a finite value"
        raise SolveOrderError(msg)
    return key


def finish_portfolio(built: BuildResult, resolved: ResolvedConfig, chain: ChainState, run_id: str, recorder: SpanRecorder) -> PortfolioResult:
    """Solve, verify independently, round to orders, bound the rounding drift, and verify the executed book the same way, timing each phase onto ``recorder``."""
    with recorder.span("solve:solve"):
        solution = solve(built.spec, chain, resolved, built.constraints, built.extras)
    post = resolved.config.post_solve
    tolerances = Tolerances(violation=post.violation_tol, obj_rel=post.objective_rel_tol, obj_abs=post.objective_abs_tol)
    constraints = constraints_of(solution)
    with recorder.span("solve:verify"):
        report = verify(built.spec, solution, chain, resolved.terms, constraints, profile=resolved.profile, tolerances=tolerances)
    if not report.passed:
        raise VerificationError(report)
    with recorder.span("solve:orders"):
        orders = solution_to_orders(built.spec, solution, built.order_inputs, run_id=run_id)
        foreign = sorted({str(side) for side in orders["side"]} - resolved.profile.order_sides)
        if foreign:
            msg = f"orders on a side order flow {resolved.profile.order_flow!r} does not trade: {foreign}"
            raise SideInvariantError(msg)
        contribution = resolved.profile.contribution(built.portfolio_id, orders)
        outside = sorted(set(contribution.security_ids) - set(built.tradable))
        if outside:
            msg = f"traded securities outside the tradable set: {outside}"
            raise ChainInvariantError(msg)
        drift = rounding_drift(built.spec, solution, orders, built.order_inputs, violation_tol=post.violation_tol)
    if not drift.passed:
        raise DriftError(drift)
    with recorder.span("solve:verify_orders"):
        executed = executed_solution(built.spec, solution, orders, resolved.profile)
        executed_report = verify(built.spec, executed, chain, resolved.terms, constraints, profile=resolved.profile, tolerances=tolerances)
    if not executed_report.passed:
        raise VerificationError(executed_report, what="executed orders")
    return PortfolioResult(
        portfolio_id=built.portfolio_id,
        spec=built.spec,
        solution=solution,
        report=report,
        orders=orders,
        rule_audit=built.rule_audit,
        chain_state=chain,
        drift=drift,
        contribution=contribution,
        executed=executed,
        executed_report=executed_report,
    )


def tradable_ids(profile: OrderFlowProfile, spec: ProblemSpec) -> tuple[str, ...]:
    """The securities the profile lets this spec trade on the side it couples through, sorted."""
    return tuple(sorted(security for security, allowed in zip(spec.security_ids, profile.tradable(spec), strict=True) if allowed))


def consume_flags(spec: ProblemSpec, consumes: Sequence[str]) -> Flags:
    """The build's consume set as a mask aligned to the spec, for the chain-state fold."""
    wanted = frozenset(consumes)
    return np.fromiter((security in wanted for security in spec.security_ids), dtype=np.bool_, count=spec.n)


def skipped(portfolio_id: str, message: str) -> PortfolioFailure:
    """A portfolio that was never solved because of another portfolio's failure."""
    return PortfolioFailure(portfolio_id=portfolio_id, stage="skipped", error_type=SKIPPED_ERROR, message=message)


def slice_and_build(shared: SharedRunData, resolved: ResolvedConfig, portfolio_id: PortfolioId, recorder: SpanRecorder) -> BuildResult | PortfolioFailure:
    """Slice the portfolio's bundle from the shared data, apply rules, and build its spec; a failure names its stage."""
    try:
        with recorder.span("build:slice"):
            data = slice_portfolio(shared.assembled, portfolio_id)
    except (PortfolioDataError, ValueError) as error:
        return PortfolioFailure.from_exception(portfolio_id, "slice", error)
    try:
        return build_portfolio(data, resolved, Decimal(shared.assembled.solve_orders[portfolio_id]), recorder)
    except Exception as error:  # noqa: BLE001  # recorded per portfolio; on_error decides what happens next
        return PortfolioFailure.from_exception(portfolio_id, "build", error)


def finish_or_fail(built: BuildResult, resolved: ResolvedConfig, chain: ChainState, run_id: str, recorder: SpanRecorder) -> Outcome:
    """Solve and finish, recording any failure at stage ``solve``."""
    try:
        result = finish_portfolio(built, resolved, chain, run_id, recorder)
    except Exception as error:  # noqa: BLE001  # recorded per portfolio; on_error decides what happens next
        log.error("portfolio failed", extra={"run_id": run_id, "portfolio_id": built.portfolio_id, "stage": "solve", "error": type(error).__name__})
        return PortfolioFailure.from_exception(built.portfolio_id, "solve", error)
    log.info("portfolio solved", extra={"run_id": run_id, "portfolio_id": built.portfolio_id, "stage": "solve", "orders": len(result.orders), "predecessors": len(chain.predecessors)})
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
        resolved = _RESOLVED[shared.config_sha256] = resolve_config(shared.config, packages=shared.packages)
    return resolved


def worker_environment(shared: SharedRunData) -> WorkerEnvironment:
    """This process's fingerprint for the run; the image digest is read from this process's own environment, which is the point."""
    return environment_for(shared.config, cwd=Path.cwd(), image_digest=os.environ.get(IMAGE_DIGEST_VARIABLE))


def probe_task(config: RunConfig, packages: tuple[str, ...] | None = None) -> TaskOutput[None]:
    """Resolve the config in this process and report the fingerprint; a resolution failure is returned, never raised.

    The resolution is kept, so the first build on this worker does not repeat it.
    """
    environment = environment_for(config, cwd=Path.cwd(), image_digest=os.environ.get(IMAGE_DIGEST_VARIABLE))
    try:
        resolved = resolve_config(config, packages=packages)
    except Exception as error:  # noqa: BLE001  # a solver or step package missing on this worker is what the probe exists to report
        failed: TaskOutput[None] = TaskOutput(outcome=PortfolioFailure.from_exception(RUN_SCOPED, "worker", error), environment=environment, host=host_name())
        return failed
    _RESOLVED[resolved.config_sha256] = resolved
    passed: TaskOutput[None] = TaskOutput(outcome=None, environment=environment, host=host_name())
    return passed


def build_task(shared: SharedRunData, portfolio_id: PortfolioId) -> TaskOutput[BuildResult]:
    """Slice, rules, solve-order key, and build; chain-free, so every portfolio's build runs at once."""
    return _task(shared, portfolio_id, "build", lambda data, resolved, recorder: slice_and_build(data, resolved, portfolio_id, recorder))


def summarize(build: TaskOutput[BuildResult]) -> TaskOutput[BuildSummary]:
    """Reduce a build to what the main process needs, keeping the build's environment stamp so a stale worker is caught before it solves.

    The build's spans travel on the summary — the result itself never leaves its worker, and this is
    the one message from that build the main process is guaranteed to read.
    """
    outcome = build.outcome
    if isinstance(outcome, PortfolioFailure):
        passed_on: TaskOutput[BuildSummary] = TaskOutput(outcome=outcome, environment=build.environment, host=build.host, spans=build.spans)
        return passed_on
    summary = BuildSummary(
        portfolio_id=outcome.portfolio_id, solve_order=outcome.solve_order, tradable=outcome.tradable, consumes=outcome.consumes, spec_sha256=outcome.spec.content_hash(), rule_audit=outcome.rule_audit
    )
    return TaskOutput(outcome=summary, environment=build.environment, host=build.host, spans=build.spans)


def solve_task(shared: SharedRunData, build: TaskOutput[BuildResult], *contributions: Contribution | PortfolioFailure) -> TaskOutput[PortfolioResult]:
    """Fold the predecessors' contributions and finish the portfolio; a failed build or predecessor is passed on, never solved around."""
    built = build.outcome

    def pipeline(data: SharedRunData, resolved: ResolvedConfig, recorder: SpanRecorder) -> Outcome:
        if isinstance(built, PortfolioFailure):
            return built
        failed = next((contribution for contribution in contributions if isinstance(contribution, PortfolioFailure)), None)
        if failed is not None:
            return skipped(built.portfolio_id, f"not solved because predecessor {failed.portfolio_id!r} failed at stage {failed.stage!r}")
        folded = [contribution for contribution in contributions if isinstance(contribution, Contribution)]
        with recorder.span("solve:chain"):
            chain = resolved.profile.chain_state(built.spec, folded, consume_flags(built.spec, built.consumes))
        return finish_or_fail(built, resolved, chain, data.run_id, recorder)

    return _task(shared, built.portfolio_id, "solve", pipeline)


def contribution(solved: TaskOutput[PortfolioResult]) -> Contribution | PortfolioFailure:
    """What a dependent solve receives: the portfolio's trades on the side the run couples through, or the failure that stops the dependent."""
    outcome = solved.outcome
    if isinstance(outcome, PortfolioFailure):
        return outcome
    return outcome.contribution


def _task[T](data: SharedRunData, portfolio_id: str, stage: str, pipeline: Callable[[SharedRunData, ResolvedConfig, SpanRecorder], T | PortfolioFailure]) -> TaskOutput[T]:
    environment = worker_environment(data)
    try:
        resolved = resolved_for(data)
    except Exception as error:  # noqa: BLE001  # a step package missing on this worker is a worker failure, not a crash
        return TaskOutput(outcome=PortfolioFailure.from_exception(portfolio_id, "worker", error), environment=environment, host=host_name())
    recorder = SpanRecorder(portfolio_id)
    with recorder.span(stage):
        outcome = pipeline(data, resolved, recorder)
    return TaskOutput(outcome=outcome, environment=environment, host=host_name(), spans=recorder.spans)
