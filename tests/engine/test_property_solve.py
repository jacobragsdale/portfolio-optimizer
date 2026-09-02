"""Property: for any feasible spec, the solver's answer passes independent verification and beats resting."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from portfolio_optimizer.domain.results import ChainState, ProblemSpec
from portfolio_optimizer.domain.sides import BUY_ONLY
from portfolio_optimizer.engine.check import constraints_of, verify
from portfolio_optimizer.engine.solve import solve
from tests.conftest import ALPHA, SHIPPED_CONSTRAINTS, constraint_frame, make_spec, resolved_example


@st.composite
def feasible_specs(draw: st.DrawFn) -> ProblemSpec:
    n = draw(st.integers(min_value=2, max_value=5))
    w0_raw = np.array(draw(st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=n, max_size=n)))
    alpha = np.array(draw(st.lists(st.floats(min_value=-0.1, max_value=0.1), min_size=n, max_size=n)))
    turnover = draw(st.floats(min_value=0.05, max_value=2.0))
    invested = 1.0 - draw(st.floats(min_value=0.0, max_value=0.5))  # the rest is cash the buy program may put to work, or not
    w0 = w0_raw / w0_raw.sum() * invested
    return make_spec(n=n, w0=w0, columns={"alpha": alpha}, shares_held=w0 * 1e6 / 100.0, max_turnover=turnover, cash_ub=1.0)


@given(spec=feasible_specs())
@settings(deadline=None, max_examples=15)
def test_solutions_verify_and_never_do_worse_than_resting(spec: ProblemSpec) -> None:
    resolved = resolved_example(objective=[ALPHA])
    chain = ChainState.empty(spec.security_ids)
    solution = solve(spec, chain, resolved, constraint_frame(SHIPPED_CONSTRAINTS))
    report = verify(spec, solution, chain, resolved.terms, constraints_of(solution), profile=BUY_ONLY)
    assert report.passed, (report.violated, report.objective_gap)
    resting = -float((spec.column("alpha") * spec.w0).sum())
    assert solution.objective is not None and solution.objective <= resting + 1e-7
