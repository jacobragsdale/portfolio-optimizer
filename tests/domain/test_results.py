"""Tier 1: the problem spec's invariants, hashing, persistence, and chain-state derivation."""

from pathlib import Path

import numpy as np
import pytest

from portfolio_optimizer.domain.results import ChainState, ConstraintReport, MissingSpecColumnError, PortfolioResult, ProblemSpecError, Solution, SolveContext, SolveStatus, derive_chain_state
from tests.conftest import Factories, Frames


def test_spec_arrays_are_read_only_float64(make: Factories) -> None:
    spec = make.spec()
    assert spec.w0.dtype == np.float64
    assert not spec.w0.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        spec.w0[0] = 1.0


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"price": np.array([1.0, 2.0])}, "price has shape \\(2,\\)"),
        ({"w0": np.array([np.nan, 0.5, 0.5])}, "non-finite"),
        ({"lb": np.array([0.5, 0.0, 0.0]), "ub": np.array([0.4, 1.0, 1.0])}, "lb > ub"),
        ({"w0": np.array([0.9, 0.05, 0.05]), "ub": np.array([0.5, 1.0, 1.0])}, "w0 lies outside"),
        ({"security_ids": ("S1", "S0", "S2")}, "not sorted"),
        ({"security_ids": ("S0", "S0", "S2")}, "not unique"),
        ({"cash_lb": 0.5, "cash_ub": 0.1}, "cash_lb > cash_ub"),
        ({"sigma_factor": np.ones((2, 4))}, "sigma_factor has shape"),
        ({"columns": {"alpha": np.ones(2)}}, "column 'alpha' has shape"),
        ({"nav": float("inf")}, "nav is not finite"),
    ],
)
def test_malformed_specs_are_rejected(make: Factories, overrides: dict[str, object], fragment: str) -> None:
    with pytest.raises(ProblemSpecError, match=fragment):
        make.spec(**overrides)


def test_empty_universe_spec_is_allowed(make: Factories) -> None:
    assert make.spec(n=0).n == 0


def test_hash_is_equal_on_a_copy_and_differs_on_one_ulp(make: Factories) -> None:
    spec = make.spec()
    same = make.spec()
    nudged = make.spec(price=np.nextafter(spec.price, np.inf))
    assert spec.content_hash() == same.content_hash()
    assert spec.content_hash() != nudged.content_hash()


def test_hash_normalizes_negative_zero(make: Factories) -> None:
    assert make.spec(tax_per_dollar=np.array([0.0, -0.0, 0.0])).content_hash() == make.spec().content_hash()


def test_hash_covers_metadata_and_extra_columns(make: Factories) -> None:
    spec = make.spec()
    assert make.spec(portfolio_id="P2").content_hash() != spec.content_hash()
    assert make.spec(columns={"alpha": np.zeros(3)}).content_hash() != spec.content_hash()


def test_npz_round_trip_preserves_hash(make: Factories, tmp_path: Path) -> None:
    spec = make.spec(sigma_factor=np.eye(3) * 0.2, columns={"alpha": np.array([0.1, 0.2, 0.3])}, psd_shift=1e-10)
    path = tmp_path / "spec.npz"
    spec.to_npz(path)
    loaded = spec.from_npz(path)
    assert loaded.content_hash() == spec.content_hash()
    assert loaded.security_ids == spec.security_ids
    assert loaded.as_of == spec.as_of


def test_missing_column_names_what_is_available(make: Factories) -> None:
    spec = make.spec(columns={"alpha": np.zeros(3)})
    np.testing.assert_array_equal(spec.column("alpha"), np.zeros(3))
    with pytest.raises(MissingSpecColumnError, match="available: \\['alpha'\\]"):
        spec.column("momentum")


def test_solution_round_trips_through_npz(tmp_path: Path) -> None:
    solution = Solution(
        w=np.array([0.5, 0.5]),
        buy=np.zeros(2),
        sell=np.zeros(2),
        objective=1.5,
        status=SolveStatus.OPTIMAL,
        solver="CLARABEL",
        solver_version="0.11",
        cvxpy_version="1.9",
        solve_time_s=0.01,
        iterations=7,
        spec_hash="ab",
    )
    path = tmp_path / "solution.npz"
    solution.to_npz(path)
    loaded = Solution.from_npz(path)
    assert loaded.status is SolveStatus.OPTIMAL
    assert loaded.iterations == 7
    np.testing.assert_array_equal(loaded.w, solution.w)


def test_chain_state_shape_must_match_ids() -> None:
    with pytest.raises(ValueError, match="cumulative_shares has shape"):
        ChainState(security_ids=("A", "B"), cumulative_shares=np.zeros(3), portfolios_done=0)


def test_derive_chain_state_sums_prior_orders_by_security(prior_results: SolveContext) -> None:
    state = derive_chain_state(prior_results, ("A", "C", "Z"))
    np.testing.assert_array_equal(state.cumulative_shares, np.array([1250.0, 25000.0, 0.0]))
    assert state.portfolios_done == 2
    assert derive_chain_state(SolveContext(), ("A",)).portfolios_done == 0


def test_solve_context_with_result_does_not_mutate_the_original(prior_results: SolveContext) -> None:
    extended = prior_results.with_result(prior_results.results[0])
    assert prior_results.portfolios_done == 2
    assert extended.portfolios_done == 3


@pytest.fixture
def prior_results(make: Factories, frames: Frames) -> SolveContext:
    spec = make.spec()
    solution = Solution(
        w=spec.w0,
        buy=np.zeros(3),
        sell=np.zeros(3),
        objective=0.0,
        status=SolveStatus.OPTIMAL,
        solver="CLARABEL",
        solver_version="0",
        cvxpy_version="0",
        solve_time_s=0.0,
        iterations=0,
        spec_hash=spec.content_hash(),
    )
    report = ConstraintReport(checks=(), objective_terms=(), recomputed_objective=0.0, solver_objective=0.0, objective_gap=0.0, objective_passed=True, unverified=())

    def result(portfolio_id: str, orders_rows: list[dict[str, object]]) -> PortfolioResult:
        return PortfolioResult(
            portfolio_id=portfolio_id, spec=spec, solution=solution, report=report, orders=frames.orders(*orders_rows), rule_audit=(), chain_state=ChainState.empty(spec.security_ids)
        )

    first = result(
        "P1", [{"security_id": "A", "side": "SELL", "quantity": 1250, "notional": 125000}, {"security_id": "C", "side": "BUY", "quantity": 20000, "reference_price": 10, "notional": 200000}]
    )
    second = result("P2", [{"security_id": "C", "side": "BUY", "quantity": 5000, "reference_price": 10, "notional": 50000}])
    return SolveContext().with_result(first).with_result(second)
