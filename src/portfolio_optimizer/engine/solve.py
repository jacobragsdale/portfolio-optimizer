"""Construct the cvxpy problem from the configured terms and constraints, solve it, and classify the outcome."""

from dataclasses import dataclass

import numpy as np

from portfolio_optimizer.config.resolve import ResolvedConfig, ResolvedStep
from portfolio_optimizer.cvx.adapter import ConstraintSet, ObjectiveTerm, RawSolve, solve_problem, solver_version, variables
from portfolio_optimizer.cvx.sides import identity_constraints
from portfolio_optimizer.domain.results import ChainState, ProblemSpec, Solution, SolveStatus
from portfolio_optimizer.domain.sides import SideProfile


class SolveSetupError(ValueError):
    """A term or constraint did not produce what its contract promises."""


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
    """The solver raised or returned an unrecognized status."""


def solve(spec: ProblemSpec, chain: ChainState, resolved: ResolvedConfig) -> Solution:
    """Solve one portfolio. Raises on infeasible, unbounded, or solver error; never returns ``w0`` as a default."""
    spec_hash = spec.content_hash()
    solver = resolved.config.solver
    if spec.n == 0:
        empty = np.zeros(0)
        return Solution(
            w=empty,
            buy=empty,
            sell=empty,
            objective=0.0,
            status=SolveStatus.OPTIMAL,
            solver=solver.name,
            solver_version=solver_version(solver.name),
            cvxpy_version="n/a",
            solve_time_s=0.0,
            iterations=0,
            spec_hash=spec_hash,
        )
    x = variables(spec.n)
    terms = [_term(step, x, spec, chain) for step in resolved.terms]
    constraints = [identity_constraints(resolved.profile.sides, x, spec.w0), *(_constraint_set(constraint.step, x, spec, chain) for constraint in resolved.constraints)]
    raw = solve_problem(x, terms, constraints, solver=solver.name, options=solver.options, time_limit_s=solver.time_limit_s, verbose=solver.verbose)
    return _classify(raw, spec, chain, spec_hash, resolved.profile)


def _term(step: ResolvedStep, x: object, spec: ProblemSpec, chain: ChainState) -> ObjectiveTerm:
    result = step.invoke(x=x, spec=spec, context=chain if step.needs_context else None)
    if not isinstance(result, ObjectiveTerm):
        msg = f"term {step.qualname!r} returned {type(result).__name__}, expected ObjectiveTerm"
        raise SolveSetupError(msg)
    return result


def _constraint_set(step: ResolvedStep, x: object, spec: ProblemSpec, chain: ChainState) -> ConstraintSet:
    result = step.invoke(x=x, spec=spec, context=chain if step.needs_context else None)
    if not isinstance(result, ConstraintSet):
        msg = f"constraint {step.qualname!r} returned {type(result).__name__}, expected ConstraintSet"
        raise SolveSetupError(msg)
    return result


def _classify(raw: RawSolve, spec: ProblemSpec, chain: ChainState, spec_hash: str, profile: SideProfile) -> Solution:
    if raw.status in (SolveStatus.OPTIMAL, SolveStatus.OPTIMAL_INACCURATE):
        if raw.w is None or raw.buy is None or raw.sell is None or raw.objective is None:
            msg = f"solver reported {raw.status} but returned no values ({raw.detail})"
            raise SolverFailureError(msg)
        # The profile decides the split the engine reports for the solver's weights; verification
        # then re-checks the identity and the objective against it.
        buy, sell = profile.split(raw.w, spec.w0)
        return Solution(
            w=raw.w,
            buy=buy,
            sell=sell,
            objective=raw.objective,
            status=raw.status,
            solver=raw.solver,
            solver_version=raw.solver_version,
            cvxpy_version=raw.cvxpy_version,
            solve_time_s=raw.solve_time_s,
            iterations=raw.iterations,
            spec_hash=spec_hash,
        )
    if raw.status is SolveStatus.INFEASIBLE:
        raise InfeasibleError(spec_hash, diagnose_infeasibility(spec, chain))
    if raw.status is SolveStatus.UNBOUNDED:
        msg = f"unbounded problem (spec {spec_hash[:12]}): a custom term or constraint removed a bound"
        raise UnboundedError(msg)
    msg = f"solver {raw.solver} failed (spec {spec_hash[:12]}): {raw.detail}"
    raise SolverFailureError(msg)


def diagnose_infeasibility(spec: ProblemSpec, chain: ChainState) -> InfeasibilityReport:
    """Arithmetic checks that explain the common infeasibilities without another solve."""
    findings: list[str] = []
    invested_lb = 1.0 - spec.cash_ub
    invested_ub = 1.0 - spec.cash_lb
    if spec.ub.sum() < invested_lb - 1e-12:
        findings.append(f"upper bounds sum to {spec.ub.sum():.6f} < required investment {invested_lb:.6f}")
    if spec.lb.sum() > invested_ub + 1e-12:
        findings.append(f"lower bounds sum to {spec.lb.sum():.6f} > allowed investment {invested_ub:.6f}")
    if len(spec.sector_names) and spec.sector_lb.sum() > invested_ub + 1e-12:
        findings.append(f"sector lower bounds sum to {spec.sector_lb.sum():.6f} > allowed investment {invested_ub:.6f}")
    capacities = np.asarray(spec.sector_matrix @ spec.ub, dtype=np.float64)
    for index, name in enumerate(spec.sector_names):
        capacity = float(capacities[index])
        if capacity < spec.sector_lb[index] - 1e-12:
            findings.append(f"sector {name!r} can hold at most {capacity:.6f} < its lower bound {spec.sector_lb[index]:.6f}")
    clamped = np.clip(spec.w0, spec.lb, spec.ub)
    needed = float(np.abs(clamped - spec.w0).sum())
    if needed > spec.max_turnover + 1e-12:
        findings.append(f"moving w0 inside its bounds needs turnover {needed:.6f} > max_turnover {spec.max_turnover:.6f}")
    consumed = chain.bought_shares * spec.price / spec.nav if chain.security_ids == spec.security_ids else np.zeros(spec.n)
    remaining = np.maximum(0.0, spec.adv_capacity - consumed)
    required = clamped - spec.w0
    blocked = [spec.security_ids[i] for i in range(spec.n) if abs(required[i]) > spec.adv_capacity[i] + 1e-12 or required[i] > remaining[i] + 1e-12]
    if blocked:
        findings.append(f"names that must trade but have no ADV budget left: {blocked}")
    return InfeasibilityReport(tuple(findings))
