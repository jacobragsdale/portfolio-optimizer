"""Property: for any feasible spec, the solver's answer passes independent verification and beats resting."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from portfolio_optimizer.domain.results import ChainState, ProblemSpec, StepRef
from portfolio_optimizer.engine.check import verify
from portfolio_optimizer.engine.solve import solve
from tests.conftest import make_spec, resolved_example

CONSTRAINT_NAMES = ["long_only", "max_weight", "cash_bounds", "turnover_cap", "cumulative_adv_participation"]


@st.composite
def feasible_specs(draw: st.DrawFn) -> ProblemSpec:
    n = draw(st.integers(min_value=2, max_value=5))
    w0_raw = np.array(draw(st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=n, max_size=n)))
    target_raw = np.array(draw(st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=n, max_size=n)))
    turnover = draw(st.floats(min_value=0.05, max_value=2.0))
    return make_spec(n=n, w0=w0_raw / w0_raw.sum(), w_target=target_raw / target_raw.sum(), shares_held=w0_raw / w0_raw.sum() * 1e6 / 100.0, max_turnover=turnover)


@given(spec=feasible_specs())
@settings(deadline=None, max_examples=15)
def test_solutions_verify_and_never_do_worse_than_resting(spec: ProblemSpec) -> None:
    resolved = resolved_example(objective={"terms": [{"name": "tracking_error", "params": {"weight": "1"}}]}, constraints=CONSTRAINT_NAMES)
    chain = ChainState.empty(spec.security_ids)
    solution = solve(spec, chain, resolved)
    terms = [StepRef("portfolio_optimizer.terms:tracking_error", {"weight": "1"})]
    constraints = [StepRef(f"portfolio_optimizer.terms:{name}", {}) for name in CONSTRAINT_NAMES]
    report = verify(spec, solution, chain, terms, constraints)
    assert report.passed, (report.violated, report.objective_gap)
    resting = float(((spec.w0 - spec.w_target) ** 2).sum())
    assert solution.objective <= resting + 1e-7
