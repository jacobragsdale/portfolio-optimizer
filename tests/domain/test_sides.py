"""Tier 1: the two-sided profile — the split, the identity residuals, the tradable set, and what a dependent receives."""

from decimal import Decimal
from typing import get_args

import numpy as np
import pytest

from portfolio_optimizer.config.models import RunConfig
from portfolio_optimizer.cvx.sides import IDENTITIES
from portfolio_optimizer.domain.results import ChainState, Contribution, Solution, SolveStatus
from portfolio_optimizer.domain.sides import PROFILES, TWO_SIDED, profile_for
from tests.conftest import Factories, Frames


def _solution(w: np.ndarray, buy: np.ndarray, sell: np.ndarray) -> Solution:
    return Solution(w=w, buy=buy, sell=sell, objective=0.0, status=SolveStatus.OPTIMAL, solver="X", solver_version="0", cvxpy_version="0", solve_time_s=0.0, iterations=1, spec_hash="h")


def test_every_profile_has_its_identity_and_the_config_can_select_it() -> None:
    assert set(PROFILES) == set(IDENTITIES) == set(get_args(RunConfig.model_fields["sides"].annotation)) == {"both"}
    assert profile_for("both") is TWO_SIDED
    with pytest.raises(ValueError, match="sides 'short' is not one the engine knows"):
        profile_for("short")


def test_two_sided_split_is_the_minimal_one() -> None:
    buy, sell = TWO_SIDED.split(np.array([0.5, 0.2, 0.3]), np.array([0.4, 0.3, 0.3]))
    np.testing.assert_allclose(buy, [0.1, 0.0, 0.0])
    np.testing.assert_allclose(sell, [0.0, 0.1, 0.0])


@pytest.mark.parametrize(
    ("buy", "sell", "violated"),
    [
        (np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.1, 0.0]), set()),
        (np.array([0.2, 0.0, 0.0]), np.array([0.0, 0.1, 0.0]), {"trade_balance"}),
        (np.array([0.1, 0.0, -0.01]), np.array([0.0, 0.1, -0.01]), {"nonneg_buy", "nonneg_sell"}),
        (np.array([0.1, 0.0, 0.5]), np.array([0.0, 0.1, 0.5]), {"sell_le_w0", "complementarity"}),
    ],
    ids=["minimal", "balance", "negative", "round-trip-beyond-holding"],
)
def test_two_sided_identity_residuals_name_what_is_violated(make: Factories, buy: np.ndarray, sell: np.ndarray, violated: set[str]) -> None:
    spec = make.spec(w0=np.array([0.4, 0.3, 0.3]))
    solution = _solution(np.array([0.5, 0.2, 0.3]), buy, sell)
    residuals = dict(TWO_SIDED.identity_residuals(spec, solution))
    assert set(residuals) == {"trade_balance", "nonneg_buy", "nonneg_sell", "sell_le_w0", "complementarity"}
    assert {name for name, residual in residuals.items() if residual.max() > 1e-9} == violated


def test_two_sided_couples_through_buys_only(make: Factories, frames: Frames) -> None:
    spec = make.spec(w0=np.array([0.5, 0.5, 0.0]), ub=np.array([0.5, 1.0, 1.0]))
    assert TWO_SIDED.tradable(spec).tolist() == [False, True, True], "S0 is capped at its holding, so it is not buyable"
    orders = frames.orders({"security_id": "S1", "side": "SELL", "quantity": 10, "notional": Decimal(1000)}, {"security_id": "S2", "side": "BUY", "quantity": 7, "notional": Decimal(700)})
    contribution = TWO_SIDED.contribution("P1", orders)
    assert (contribution.security_ids, contribution.bought_shares.tolist()) == (("S2",), [7.0])
    state = TWO_SIDED.chain_state(spec, [Contribution("P0", ("S0", "S2"), np.array([5.0, 3.0]))])
    assert isinstance(state, ChainState) and state.bought_shares.tolist() == [0.0, 0.0, 3.0], "a predecessor's buy of a name this portfolio cannot buy is masked out"
    assert TWO_SIDED.order_sides == {"BUY", "SELL"}
