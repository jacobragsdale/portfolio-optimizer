"""Hand one portfolio to the configured solve step and classify what comes back.

The step — the shipped cvxpy one, a firm's library, a pure function — returns weights; this module
turns them into a :class:`~portfolio_optimizer.domain.results.Solution` through the order-flow profile's
split, stamps it with the portfolio's typed constraint rows as records — the engine's reading of the
rows, not the step's, so what the verifier checks cannot be narrowed by the step — explains an
infeasibility with arithmetic, and raises for everything else.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_optimizer.config.resolve import ResolvedConfig
from portfolio_optimizer.config.steps import ResolvedStep
from portfolio_optimizer.domain.constraints import parse_constraints
from portfolio_optimizer.domain.order_flow import OrderFlowProfile
from portfolio_optimizer.domain.results import ChainState, ConstraintRecord, ProblemSpec, Solution, SolveStatus
from portfolio_optimizer.engine.environment import package_versions
from portfolio_optimizer.solving import SolveRequest, SolveResult, SolveSetupError


@dataclass(frozen=True, slots=True)
class InfeasibilityReport:
    """Cheap arithmetic explanations of why no feasible portfolio exists."""

    findings: tuple[str, ...]


class InfeasibleError(RuntimeError):
    """The solver proved there is no feasible portfolio."""

    def __init__(self, spec_hash: str, report: InfeasibilityReport) -> None:
        self.spec_hash = spec_hash
        self.report = report
        detail = "; ".join(report.findings) if report.findings else "no arithmetic cause found; inspect the persisted spec"
        super().__init__(f"infeasible problem (spec {spec_hash[:12]}): {detail}")


class UnboundedError(RuntimeError):
    """Every variable is bounded, so this indicates a bug in a custom term or constraint."""


class SolverFailureError(RuntimeError):
    """The solve step raised, or returned a status the engine cannot act on."""


def solve(spec: ProblemSpec, chain: ChainState, resolved: ResolvedConfig, constraints: pd.DataFrame, extras: Mapping[str, pd.DataFrame] | None = None) -> Solution:
    """Solve one portfolio with the configured step. Raises on infeasible, unbounded, or failure; never returns ``w0`` as a default."""
    spec_hash = spec.content_hash()
    step = resolved.solve
    if spec.n == 0:
        empty = np.zeros(0)
        return Solution(
            w=empty, buy=empty, sell=empty, objective=0.0, status=SolveStatus.OPTIMAL, solver=step.qualname, solver_version=_step_version(step), solve_time_s=0.0, iterations=0, spec_hash=spec_hash
        )
    request = SolveRequest(spec=spec, chain=chain, profile=resolved.profile, terms=resolved.terms, constraints=constraints, extras=extras or {})
    result = step.invoke(request=request)
    if not isinstance(result, SolveResult):
        msg = f"solve step {step.qualname!r} returned {type(result).__name__}, expected SolveResult"
        raise SolveSetupError(msg)
    parsed = parse_constraints(constraints)
    return _classify(result, spec, chain, spec_hash, resolved, tuple(model.record() for model in parsed.typed) if parsed is not None else ())


def _classify(result: SolveResult, spec: ProblemSpec, chain: ChainState, spec_hash: str, resolved: ResolvedConfig, records: tuple[ConstraintRecord, ...]) -> Solution:
    step = resolved.solve
    solver = result.solver if result.solver is not None else step.qualname
    if result.status in (SolveStatus.OPTIMAL, SolveStatus.OPTIMAL_INACCURATE):
        if result.w is None:
            msg = f"solve step {step.qualname!r} reported {result.status} but returned no weights ({result.detail})"
            raise SolverFailureError(msg)
        if result.w.shape != (spec.n,):
            msg = f"solve step {step.qualname!r} returned weights of shape {result.w.shape}, expected {(spec.n,)}"
            raise SolveSetupError(msg)
        # The profile decides the split the engine reports for the step's weights; verification
        # then re-checks the identity and, when the step minimized one, the objective against it.
        buy, sell = resolved.profile.split(result.w, spec.w0)
        return Solution(
            constraints=records,
            duals=result.duals,
            w=result.w,
            buy=buy,
            sell=sell,
            objective=result.objective,
            status=result.status,
            solver=solver,
            solver_version=result.solver_version if result.solver_version is not None else _step_version(step),
            solve_time_s=result.solve_time_s,
            iterations=result.iterations,
            spec_hash=spec_hash,
        )
    if result.status is SolveStatus.INFEASIBLE:
        raise InfeasibleError(spec_hash, diagnose_infeasibility(spec, chain, profile=resolved.profile))
    if result.status is SolveStatus.UNBOUNDED:
        msg = f"unbounded problem (spec {spec_hash[:12]}): a custom term or constraint removed a bound"
        raise UnboundedError(msg)
    msg = f"solver {solver} failed (spec {spec_hash[:12]}): {result.detail}"
    raise SolverFailureError(msg)


def _step_version(step: ResolvedStep) -> str:
    """The version of the distribution behind a solve step, for the manifest when the step names none itself."""
    return next(iter(package_versions([step.qualname.partition(":")[0]]).values()), "unknown")


def diagnose_infeasibility(spec: ProblemSpec, chain: ChainState, *, profile: OrderFlowProfile) -> InfeasibilityReport:
    """Arithmetic checks that explain the common infeasibilities without another solve; the profile adds the ones its side creates.

    Each check reads what the spec carries — the cash bounds, the turnover cap, the ADV capacity —
    and says nothing about a limit the spec does not have.
    """
    findings: list[str] = profile.infeasible_starts(spec)
    cash_lb, cash_ub, max_turnover = spec.scalars.get("cash_lb"), spec.scalars.get("cash_ub"), spec.scalars.get("max_turnover")
    if cash_ub is not None and spec.ub.sum() < 1.0 - cash_ub - 1e-12:
        findings.append(f"upper bounds sum to {spec.ub.sum():.6f} < required investment {1.0 - cash_ub:.6f}")
    if cash_lb is not None and spec.lb.sum() > 1.0 - cash_lb + 1e-12:
        findings.append(f"lower bounds sum to {spec.lb.sum():.6f} > allowed investment {1.0 - cash_lb:.6f}")
    clamped = np.clip(spec.w0, spec.lb, spec.ub)
    needed = float(np.abs(clamped - spec.w0).sum())
    if max_turnover is not None and needed > max_turnover + 1e-12:
        findings.append(f"moving w0 inside its bounds needs turnover {needed:.6f} > max_turnover {max_turnover:.6f}")
    capacity = spec.columns.get("adv_capacity")
    if capacity is not None:
        consumed = chain.traded_shares * spec.price / spec.nav if chain.security_ids == spec.security_ids else np.zeros(spec.n)
        remaining = np.maximum(0.0, capacity - consumed)
        required = clamped - spec.w0
        blocked = [spec.security_ids[i] for i in range(spec.n) if abs(required[i]) > capacity[i] + 1e-12 or required[i] > remaining[i] + 1e-12]
        if blocked:
            findings.append(f"names that must trade but have no ADV budget left: {blocked}")
    return InfeasibilityReport(tuple(findings))
