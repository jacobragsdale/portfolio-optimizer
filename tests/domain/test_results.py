"""Tier 1: the problem spec's invariants, hashing, persistence, the buyable set, and chain-state derivation from predecessors' buys."""

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_array

from portfolio_optimizer.domain.results import TRACEBACK_LIMIT, ChainState, Contribution, MissingSpecColumnError, PortfolioFailure, ProblemSpecError, Solution, SolveStatus, derive_chain_state
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
        ({"security_ids": ("S1", "S0", "S2")}, "not sorted"),
        ({"security_ids": ("S0", "S0", "S2")}, "not unique"),
        ({"cash_lb": 0.5, "cash_ub": 0.1}, "cash_lb > cash_ub"),
        ({"columns": {"alpha": np.ones(2)}}, "column 'alpha' has shape"),
        ({"flags": {"esg": np.ones(2, dtype=bool)}}, "flag 'esg' has shape"),
        ({"columns": {"esg": np.ones(3)}, "flags": {"esg": np.ones(3, dtype=bool)}}, "both a column and a flag"),
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
    flagged = make.spec(flags={"esg": np.array([True, False, True])})
    assert flagged.content_hash() != spec.content_hash()
    assert flagged.content_hash() != make.spec(flags={"esg": np.array([True, True, True])}).content_hash()


def test_npz_round_trip_preserves_hash(make: Factories, tmp_path: Path) -> None:
    spec = make.spec(columns={"alpha": np.array([0.1, 0.2, 0.3])}, flags={"esg": np.array([True, False, True])}, cash_ub=0.05)
    path = tmp_path / "spec.npz"
    spec.to_npz(path)
    loaded = spec.from_npz(path)
    assert loaded.content_hash() == spec.content_hash()
    assert loaded.flag("esg").dtype == np.bool_
    assert not loaded.flag("esg").flags.writeable
    assert isinstance(loaded.sector_matrix, csr_array) and not loaded.sector_matrix.data.flags.writeable
    assert loaded.security_ids == spec.security_ids
    assert loaded.as_of_date == spec.as_of_date


def test_sector_matrix_is_stored_sparse_whatever_form_it_arrives_in(make: Factories) -> None:
    membership = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    dense = make.spec(sector_names=("A", "B"), sector_matrix=membership)
    sparse = make.spec(sector_names=("A", "B"), sector_matrix=csr_array(membership))
    assert isinstance(dense.sector_matrix, csr_array) and dense.sector_matrix.nnz == 3
    assert dense.content_hash() == sparse.content_hash()
    assert dense.content_hash() != make.spec(sector_names=("A", "B"), sector_matrix=membership[::-1]).content_hash()
    with pytest.raises(ProblemSpecError, match="sector_matrix has shape"):
        make.spec(sector_names=("A", "B"), sector_matrix=membership[:1])


def test_one_sector_row_comes_back_sparse_and_an_unknown_name_is_refused(make: Factories) -> None:
    spec = make.spec(sector_names=("A", "B"), sector_matrix=np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
    row = spec.sector("B")
    assert isinstance(row, csr_array) and row.shape == (1, 3) and row.nnz == 1, "a dense row per sector is what made the old spec enormous"
    np.testing.assert_array_equal(row.toarray(), [[0.0, 1.0, 0.0]])
    with pytest.raises(MissingSpecColumnError, match=r"spec has no sector 'ENERGY'; available: \['A', 'B'\]"):
        spec.sector("ENERGY")


def test_missing_column_names_what_is_available(make: Factories) -> None:
    spec = make.spec(columns={"alpha": np.zeros(3)}, flags={"esg": np.ones(3, dtype=bool)})
    np.testing.assert_array_equal(spec.column("alpha"), np.zeros(3))
    with pytest.raises(MissingSpecColumnError, match="spec has no flag 'liquid'; available: \\['esg'\\]"):
        spec.flag("liquid")
    with pytest.raises(MissingSpecColumnError, match="available: \\['alpha'\\]"):
        spec.column("momentum")


def test_solution_round_trips_through_npz(make: Factories, tmp_path: Path) -> None:
    solution = make.solution(make.spec(n=2), objective=1.5, solver="CLARABEL", solver_version="0.11", solve_time_s=0.01, iterations=7)
    path = tmp_path / "solution.npz"
    solution.to_npz(path)
    loaded = Solution.from_npz(path)
    assert loaded.status is SolveStatus.OPTIMAL
    assert loaded.iterations == 7
    np.testing.assert_array_equal(loaded.w, solution.w)


def test_buyable_is_where_a_positive_buy_is_allowed(make: Factories) -> None:
    spec = make.spec(ub=np.array([1.0, 1.0 / 3.0, 0.5]))
    assert spec.buyable.tolist() == [True, False, True], "a name capped at its current weight is not buyable"


def test_sellable_is_where_a_positive_sell_is_allowed(make: Factories) -> None:
    spec = make.spec(w0=np.array([0.5, 0.5, 0.0]), lb=np.array([0.0, 0.5, 0.0]))
    assert spec.sellable.tolist() == [True, False, False], "a name floored at its current weight, or not held, is not sellable"


def test_chain_state_shape_must_match_ids() -> None:
    with pytest.raises(ValueError, match="traded_shares has shape"):
        ChainState(security_ids=("A", "B"), traded_shares=np.zeros(3))


def test_contribution_keeps_only_the_side_it_is_asked_for(contributions: tuple[Contribution, Contribution], frames: Frames) -> None:
    first, second = contributions
    assert (first.portfolio_id, first.security_ids, first.traded_shares.tolist()) == ("P1", ("C",), [20000.0])
    assert (second.security_ids, second.traded_shares.tolist()) == (("C",), [5000.0])
    orders = frames.orders(
        {"security_id": "A", "side": "SELL", "quantity": 1250, "notional": 125000}, {"security_id": "C", "side": "BUY", "quantity": 20000, "reference_price": 10, "notional": 200000}
    )
    sells = Contribution.from_orders("P1", orders, "SELL")
    assert (sells.security_ids, sells.traded_shares.tolist()) == (("A",), [1250.0])


def test_derive_chain_state_folds_predecessors_buys_onto_the_spec(contributions: tuple[Contribution, Contribution]) -> None:
    state = derive_chain_state(("A", "C", "Z"), np.array([True, True, True]), contributions)
    np.testing.assert_array_equal(state.traded_shares, np.array([0.0, 25000.0, 0.0]))
    assert state.predecessors == ("P1", "P2")
    assert derive_chain_state(("A",), np.array([True]), ()).predecessors == ()


def test_derive_chain_state_zeroes_what_this_portfolio_cannot_buy(contributions: tuple[Contribution, Contribution]) -> None:
    state = derive_chain_state(("A", "C", "Z"), np.array([True, False, True]), contributions)
    np.testing.assert_array_equal(state.traded_shares, np.zeros(3))


def test_chain_hash_covers_the_shares_and_not_who_bought_them(contributions: tuple[Contribution, Contribution]) -> None:
    first, _ = contributions
    both = derive_chain_state(("C",), np.array([True]), contributions)
    anonymous = ChainState(security_ids=("C",), traded_shares=np.array([25000.0]))
    assert both.content_hash() == anonymous.content_hash()
    assert derive_chain_state(("C",), np.array([True]), (first,)).content_hash() != both.content_hash()


def test_chain_hash_is_the_one_bought_shares_produced() -> None:
    # The field was renamed from bought_shares on 2026-08-29; the hash covers the ids and the values only, so recorded manifests still match.
    assert ChainState(security_ids=("C",), traded_shares=np.array([25000.0])).content_hash() == "7606b5f9997880f2d4a3c939ce7e280e110d7411d0e3c7af44968e9af8ee6f6b"
    assert ChainState(security_ids=("A", "C", "Z"), traded_shares=np.array([0.0, 25000.0, 0.0])).content_hash() == "4a6f72f066b1694ea5e186f2662b1afbe84b262a2cce3f43946ba506009c9658"


def test_chain_state_round_trips_through_npz(tmp_path: Path, contributions: tuple[Contribution, Contribution]) -> None:
    state = derive_chain_state(("A", "C"), np.array([True, True]), contributions)
    state.to_npz(tmp_path / "chain.npz")
    loaded = ChainState.from_npz(tmp_path / "chain.npz")
    assert loaded.content_hash() == state.content_hash()
    assert loaded.predecessors == ("P1", "P2")


@pytest.fixture
def contributions(frames: Frames) -> tuple[Contribution, Contribution]:
    first = frames.orders({"security_id": "A", "side": "SELL", "quantity": 1250, "notional": 125000}, {"security_id": "C", "side": "BUY", "quantity": 20000, "reference_price": 10, "notional": 200000})
    second = frames.orders({"security_id": "C", "side": "BUY", "quantity": 5000, "reference_price": 10, "notional": 50000})
    return Contribution.from_orders("P1", first, "BUY"), Contribution.from_orders("P2", second, "BUY")


def raise_from_a_named_frame() -> None:
    """A frame with a name a traceback can be searched for."""
    msg = "no such column 'oas'"
    raise KeyError(msg)


def raise_through_a_cause() -> None:
    """The same failure wrapped, so the traceback has a chain to keep."""
    try:
        raise_from_a_named_frame()
    except KeyError as cause:
        msg = "could not build the universe"
        raise ValueError(msg) from cause


def test_from_exception_keeps_the_frame_the_failure_happened_in() -> None:
    with pytest.raises(KeyError) as caught:
        raise_from_a_named_frame()
    failure = PortfolioFailure.from_exception("P1", "build", caught.value)
    assert (failure.portfolio_id, failure.stage, failure.error_type) == ("P1", "build", "KeyError")
    assert failure.traceback is not None
    assert "raise_from_a_named_frame" in failure.traceback, "the frame is the whole point: the message alone never says where"
    assert "KeyError" in failure.traceback


def test_from_exception_keeps_the_cause_chain() -> None:
    with pytest.raises(ValueError) as caught:
        raise_through_a_cause()
    failure = PortfolioFailure.from_exception("P1", "build", caught.value)
    assert failure.traceback is not None
    assert "raise_from_a_named_frame" in failure.traceback
    assert "The above exception was the direct cause" in failure.traceback


def test_message_override_leaves_the_traceback_alone() -> None:
    with pytest.raises(KeyError) as caught:
        raise_from_a_named_frame()
    failure = PortfolioFailure.from_exception("P1", "worker", caught.value, message="task 'solve-P1' killed its worker")
    assert failure.message == "task 'solve-P1' killed its worker"
    assert failure.traceback is not None and "raise_from_a_named_frame" in failure.traceback


def test_an_enormous_traceback_is_capped_at_both_ends() -> None:
    # Deep recursion is not the case that needs the cap — Python collapses repeated frames itself.
    # A validation error that names every offending row is, and a book has a hundred thousand of them.
    with pytest.raises(ValueError) as caught:
        msg = "rejected rows: " + ", ".join(f"SEC{index:06d}" for index in range(6000))
        raise ValueError(msg)
    failure = PortfolioFailure.from_exception("P1", "build", caught.value)
    assert failure.traceback is not None
    assert len(failure.traceback) < TRACEBACK_LIMIT + 100, "the cap plus its elision notice, not the whole message"
    assert failure.traceback.startswith("Traceback"), "the origin survives"
    assert failure.traceback.rstrip().endswith("SEC005999"), "the tail survives"
    assert "character(s) elided" in failure.traceback


def test_an_ordinary_traceback_is_passed_through_whole() -> None:
    with pytest.raises(KeyError) as caught:
        raise_from_a_named_frame()
    failure = PortfolioFailure.from_exception("P1", "build", caught.value)
    assert failure.traceback is not None and "elided" not in failure.traceback


def test_a_failure_no_exception_produced_has_no_traceback() -> None:
    assert PortfolioFailure("P2", "skipped", "SkippedAfterFailure", "predecessor 'P1' failed").traceback is None
