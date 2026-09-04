"""Tier 1/2: the shipped solve steps — the cvxpy step through the seam, every shipped kind rendered, and a pro-rata fill that is verified like any solve."""

from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from portfolio_optimizer.cvx.adapter import ConstraintSet
from portfolio_optimizer.cvx.order_flow import decision_variables
from portfolio_optimizer.domain.constraints import CashLimit, ExposureLimit, GroupLimit, ParticipationLimit, ScalarRef, TurnoverLimit, WeightLimit
from portfolio_optimizer.domain.order_flow import INFLOW
from portfolio_optimizer.domain.results import ChainState, Contribution, PortfolioResult, ProblemSpec
from portfolio_optimizer.engine.build import standard
from portfolio_optimizer.engine.check import constraints_of, verify
from portfolio_optimizer.engine.runner import EXIT_OK
from portfolio_optimizer.solvers import CvxpyParams, cvxpy, pro_rata_fill
from portfolio_optimizer.solving import SolveRequest, SolveSetupError
from tests.conftest import Factories, constraint_frame, resolved_example_real
from tests.engine.fakes import LazyBackend, factory_for
from tests.engine.support import execute


def _request(spec: ProblemSpec, chain: ChainState | None = None) -> SolveRequest:
    resolved = resolved_example_real()
    return SolveRequest(spec=spec, chain=chain if chain is not None else ChainState.empty(spec.security_ids), profile=INFLOW, terms=resolved.terms, constraints=constraint_frame())


def test_pro_rata_fill_spreads_the_cash_evenly_over_the_names_it_may_buy(make: Factories) -> None:
    spec = make.spec(w0=np.array([0.3, 0.3, 0.3]))
    result = pro_rata_fill(_request(spec))
    assert result.w is not None and result.objective is None
    np.testing.assert_allclose(result.w - spec.w0, [0.1 / 3] * 3, atol=1e-12, err_msg="0.1 of cash split three ways; the fill has no view on which name is better")
    assert result.w.sum() == pytest.approx(1.0)


def test_pro_rata_fill_respects_a_cap_and_gives_the_excess_to_the_rest(make: Factories) -> None:
    spec = make.spec(w0=np.array([0.3, 0.3, 0.3]), ub=np.array([0.32, 1.0, 1.0]))
    result = pro_rata_fill(_request(spec))
    assert result.w is not None
    np.testing.assert_allclose(result.w - spec.w0, [0.02, 0.04, 0.04], atol=1e-12, err_msg="S0 fills at 0.02 of room; the 0.08 it could not take is split between the two still open")


def test_pro_rata_fill_reads_the_chain(make: Factories) -> None:
    spec = make.spec(w0=np.array([0.3, 0.3, 0.3]), adv_capacity=np.array([0.05, 1.0, 1.0]))
    consumed = Contribution("P0", ("S0",), np.array([300.0]))  # 300 shares at 100 on NAV 1e6 is 0.03 of NAV
    chain = INFLOW.chain_state(spec, [consumed], np.ones(spec.n, dtype=np.bool_))
    result = pro_rata_fill(_request(spec, chain))
    assert result.w is not None
    np.testing.assert_allclose(result.w - spec.w0, [0.02, 0.04, 0.04], atol=1e-12, err_msg="S0 has 0.05 - 0.03 of ADV budget left after its predecessor")


def test_pro_rata_fill_without_an_adv_column_is_bounded_by_the_caps_alone(make: Factories) -> None:
    spec = make.spec(w0=np.array([0.3, 0.3, 0.3]), columns={"alpha": np.zeros(3)})
    result = pro_rata_fill(_request(spec))
    assert result.w is not None and result.w.sum() == pytest.approx(1.0)


def test_pro_rata_fill_refuses_a_book_below_its_cash_floor(make: Factories) -> None:
    spec = make.spec(w0=np.array([0.4, 0.4, 0.3]), cash_lb=0.0)
    with pytest.raises(ValueError, match="below the floor"):
        pro_rata_fill(_request(spec))


def test_pro_rata_fill_verifies_like_a_solve(make: Factories) -> None:
    details = make.details(nav=Decimal(1_250_000), cash=Decimal(250_000), max_weight=Decimal("0.6"))
    spec = standard(make.portfolio_data(details=details))
    resolved = resolved_example_real(solve="pro_rata_fill")
    chain = ChainState.empty(spec.security_ids)
    result = pro_rata_fill(SolveRequest(spec=spec, chain=chain, profile=INFLOW, terms=resolved.terms, constraints=constraint_frame()))
    assert result.w is not None
    solution = make.solution(spec, w=result.w, buy=result.w - spec.w0, objective=None, solver="f", iterations=None)
    report = verify(spec, solution, chain, resolved.terms, constraints_of(solution), profile=INFLOW)
    assert report.passed, report.violated
    assert result.w.sum() == pytest.approx(1.0), "cash_lb = cash_ub = 0 means every dollar is invested"


def test_the_cvxpy_step_refuses_rows_it_cannot_interpret(make: Factories) -> None:
    spec = make.spec()
    request = _request(spec)
    foreign = request.constraints.rename(columns={"kind": "rule"})
    with pytest.raises(SolveSetupError, match="carry no `kind` column"):
        cvxpy(SolveRequest(spec=spec, chain=request.chain, profile=INFLOW, terms=request.terms, constraints=foreign), CvxpyParams())


def test_the_cvxpy_step_reports_the_shadow_price_of_what_bound(make: Factories) -> None:
    spec = make.spec()
    result = cvxpy(_request(spec), CvxpyParams())
    assert set(result.duals) == {"no_sells", "cash_floor", "cash_cap", "turnover", "adv"}
    assert result.duals["cash_floor"] > 0.0 and result.duals["turnover"] == pytest.approx(0.0, abs=1e-6), "the cash floor binds a fully invested book; a turnover cap of two never does"


@pytest.mark.parametrize(
    "model",
    [
        WeightLimit(name="cap", direction="<=", bounds=Decimal("0.4")),
        WeightLimit(name="floor", direction=">=", bounds=Decimal("0.1"), scope="is_thin"),
        GroupLimit(name="bands", direction="<=", column="sector", bounds={"TECH": Decimal("0.9")}),
        ExposureLimit(name="beta", direction="<=", column="alpha", bounds=ScalarRef(scalar="max_weight")),
        CashLimit(name="cash", direction=">=", bounds=Decimal(0)),
        TurnoverLimit(name="turnover", direction="<=", bounds=ScalarRef(scalar="max_turnover")),
        ParticipationLimit(name="adv", direction="<=", scope="is_thin"),
    ],
    ids=lambda model: type(model).__name__,
)
def test_every_shipped_kind_renders_a_constraint_set_under_its_own_name(make: Factories, model: object) -> None:
    spec = make.spec(flags={"is_thin": np.array([False, False, True])})
    x = decision_variables("inflow", spec)
    rendered = model.to_cvxpy(x, spec, ChainState.empty(spec.security_ids))  # ty: ignore[unresolved-attribute]  # every model in the table is a TypedConstraint
    assert isinstance(rendered, ConstraintSet) and rendered.name == model.name  # ty: ignore[unresolved-attribute]  # see above
    assert all(constraint.is_dcp() for constraint in rendered.constraints)


def test_the_runs_parameter_datasets_reach_the_solve_step_through_the_build(tmp_path: Path) -> None:
    report = execute(tmp_path, backend_factory=factory_for(LazyBackend()), solve="tests.steps:cvxpy_reporting_a_runtime_parameter")
    assert report.exit_code == EXIT_OK
    for outcome in report.outcomes:
        assert isinstance(outcome, PortfolioResult) and outcome.report.passed
        assert outcome.solution.solver == "risk_aversion=2.5", "the example's global_parameters frame reached the step untouched, past the spec the build produced"
    assert report.manifest.portfolios[0].solve is not None
    assert report.manifest.portfolios[0].solve.solver == "risk_aversion=2.5", "and what the step read is recorded with the rest of the provenance"


def test_the_example_runs_end_to_end_on_the_pro_rata_fill(tmp_path: Path) -> None:
    report = execute(tmp_path, backend_factory=factory_for(LazyBackend()), solve="pro_rata_fill", objective=[])
    assert report.exit_code == EXIT_OK, [str(outcome) for outcome in report.outcomes]
    for outcome in report.outcomes:
        assert isinstance(outcome, PortfolioResult) and outcome.report.passed and outcome.solution.solver == "portfolio_optimizer.solvers:pro_rata_fill"
        assert set(outcome.orders["side"]) <= {"BUY"}
