"""Tier 1/2: the shipped solve steps — the cvxpy step through the seam, and a pro-rata fill that is verified like any solve."""

from decimal import Decimal

import numpy as np
import pytest

from portfolio_optimizer.domain.results import ChainState, Contribution, PortfolioResult, ProblemSpec
from portfolio_optimizer.domain.sides import TWO_SIDED
from portfolio_optimizer.engine.build import build_problem_spec
from portfolio_optimizer.engine.check import verify
from portfolio_optimizer.engine.runner import EXIT_OK
from portfolio_optimizer.engine.tasks import constraint_refs, step_refs
from portfolio_optimizer.solvers import pro_rata_fill
from portfolio_optimizer.solving import SolveRequest
from tests.conftest import Factories, resolved_example_real
from tests.engine.test_backends import LazyBackend, execute


def _request(spec: ProblemSpec, chain: ChainState | None = None) -> SolveRequest:
    resolved = resolved_example_real()
    return SolveRequest(
        spec=spec, chain=chain if chain is not None else ChainState.empty(spec.security_ids), profile=TWO_SIDED, terms=resolved.terms, constraints=resolved.constraints, solver=resolved.config.solver
    )


def test_pro_rata_fill_spends_the_cash_on_the_underweights_in_proportion(make: Factories) -> None:
    spec = make.spec(w0=np.array([0.3, 0.3, 0.3]), w_target=np.array([0.5, 0.35, 0.15]))
    result = pro_rata_fill(_request(spec))
    assert result.w is not None and result.objective is None
    np.testing.assert_allclose(result.w - spec.w0, [0.08, 0.02, 0.0], atol=1e-12, err_msg="0.1 of cash split 0.2:0.05 between the two underweights; the overweight name is untouched")
    assert result.w.sum() == pytest.approx(1.0)


def test_pro_rata_fill_respects_a_cap_and_gives_the_excess_to_the_rest(make: Factories) -> None:
    spec = make.spec(w0=np.array([0.3, 0.3, 0.3]), w_target=np.array([0.5, 0.35, 0.15]), ub=np.array([0.32, 1.0, 1.0]))
    result = pro_rata_fill(_request(spec))
    assert result.w is not None
    np.testing.assert_allclose(result.w - spec.w0, [0.02, 0.08, 0.0], atol=1e-12, err_msg="S0 is capped at 0.02 of room; the remaining 0.08 goes to the other underweight")


def test_pro_rata_fill_reads_the_chain(make: Factories) -> None:
    spec = make.spec(w0=np.array([0.3, 0.3, 0.3]), w_target=np.array([0.5, 0.35, 0.15]), adv_capacity=np.array([0.05, 1.0, 1.0]))
    consumed = Contribution("P0", ("S0",), np.array([300.0]))  # 300 shares at 100 on NAV 1e6 is 0.03 of NAV
    chain = TWO_SIDED.chain_state(spec, [consumed])
    result = pro_rata_fill(_request(spec, chain))
    assert result.w is not None
    np.testing.assert_allclose(result.w - spec.w0, [0.02, 0.08, 0.0], atol=1e-12, err_msg="S0 has 0.05 - 0.03 of ADV budget left after its predecessor")


def test_pro_rata_fill_refuses_a_book_below_its_cash_floor(make: Factories) -> None:
    spec = make.spec(w0=np.array([0.4, 0.4, 0.3]), cash_lb=0.0)
    with pytest.raises(ValueError, match="below the floor"):
        pro_rata_fill(_request(spec))


def test_pro_rata_fill_verifies_like_a_solve(make: Factories) -> None:
    details = make.details(nav=Decimal(1_250_000), cash=Decimal(250_000))
    output = build_problem_spec(make.portfolio_data(details=details, style=make.style(max_weight=Decimal("0.6"))))
    resolved = resolved_example_real(solve="pro_rata_fill")
    chain = ChainState.empty(output.spec.security_ids)
    result = pro_rata_fill(SolveRequest(spec=output.spec, chain=chain, profile=TWO_SIDED, terms=resolved.terms, constraints=resolved.constraints, solver=resolved.config.solver))
    assert result.w is not None
    from portfolio_optimizer.domain.results import Solution, SolveStatus  # local: the assertion is about verification, not the record

    solution = Solution(
        w=result.w,
        buy=result.w - output.spec.w0,
        sell=np.zeros(3),
        objective=None,
        status=SolveStatus.OPTIMAL,
        solver="f",
        solver_version="0",
        cvxpy_version="n/a",
        solve_time_s=0.0,
        iterations=None,
        spec_hash=output.spec.content_hash(),
    )
    report = verify(output.spec, solution, chain, step_refs(resolved.terms), constraint_refs(resolved.constraints))
    assert report.passed, report.violated
    assert result.w.sum() == pytest.approx(1.0), "cash_bounds [0, 0] means every dollar is invested"


def test_the_example_runs_end_to_end_on_the_pro_rata_fill(tmp_path: object) -> None:
    from pathlib import Path

    report = execute(Path(str(tmp_path)), LazyBackend(), solve="pro_rata_fill")
    assert report.exit_code == EXIT_OK
    for outcome in report.outcomes:
        assert isinstance(outcome, PortfolioResult) and outcome.report.passed and outcome.solution.solver == "portfolio_optimizer.solvers:pro_rata_fill"
        assert set(outcome.orders["side"]) <= {"BUY"}
