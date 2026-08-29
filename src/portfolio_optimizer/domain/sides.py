"""Which side a run trades, as the one object that knows: the side profile.

A run is two-sided — the problem the engine grew up with — or, once they land, buy-only or
sell-only; ``sides`` in the run config selects one. Everything side-dependent lives here or in its
cvxpy half, ``cvx/sides.py``: how a solver's weights become a trade, which securities the dependency
graph and the chain state are built from, what a dependent solve receives, and which invariants the
verifier adds. Nothing else in the engine asks which side a run is; it consumes what the profile
hands it. This module is cvxpy-free so the verifier and the ``verify`` command can use it without the
solver stack.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from portfolio_optimizer.domain.results import F64, ChainState, Contribution, Flags, ProblemSpec, Solution, derive_chain_state

type Sides = Literal["both"]
"""The sides a run may trade. ``buy`` and ``sell`` are decided and arrive next; ``both`` is the two-sided problem."""


class SideProfile(Protocol):
    """What the engine needs to know about the side a run trades."""

    @property
    def sides(self) -> Sides:
        """The config value this profile answers to."""
        ...

    @property
    def order_sides(self) -> frozenset[str]:
        """The order sides a solved portfolio may produce."""
        ...

    def tradable(self, spec: ProblemSpec) -> Flags:
        """The securities this portfolio can trade on the side it couples through; the dependency graph and the chain are built from it."""
        ...

    def split(self, w: F64, w0: F64) -> tuple[F64, F64]:
        """The ``(buy, sell)`` the engine reports for solved weights ``w``."""
        ...

    def identity_residuals(self, spec: ProblemSpec, solution: Solution) -> list[tuple[str, F64]]:
        """Violations of the trade identity, one named residual vector per check; the verifier's twin of ``cvx/sides.py``."""
        ...

    def contribution(self, portfolio_id: str, orders: pd.DataFrame) -> Contribution:
        """What a dependent solve receives from this portfolio's orders."""
        ...

    def chain_state(self, spec: ProblemSpec, contributions: Sequence[Contribution]) -> ChainState:
        """Predecessors' contributions folded onto this spec and masked to what it can trade."""
        ...


@dataclass(frozen=True, slots=True)
class TwoSided:
    """Buys and sells in one problem, ``w = w0 + buy - sell``; portfolios couple through buys only.

    A sell reaches no other portfolio, so the tradable set is the buyable set and a contribution is the
    BUY rows. The reported split is the minimal one — the solver's own pair may carry slack on both
    sides, or a round trip where a term rewards one (see IDEAS).
    """

    sides: Sides = "both"
    order_sides: frozenset[str] = frozenset({"BUY", "SELL"})

    def tradable(self, spec: ProblemSpec) -> Flags:
        """The buyable set: ``ub > w0``."""
        return spec.buyable

    def split(self, w: F64, w0: F64) -> tuple[F64, F64]:
        """``buy = max(w - w0, 0)``, ``sell = max(w0 - w, 0)``."""
        delta = w - w0
        return np.maximum(delta, 0.0), np.maximum(-delta, 0.0)

    def identity_residuals(self, spec: ProblemSpec, solution: Solution) -> list[tuple[str, F64]]:
        """``w - w0 = buy - sell`` (an equality), both non-negative, ``sell ≤ w0``, and no name both bought and sold."""
        return [
            ("trade_balance", np.abs(solution.w - spec.w0 - solution.buy + solution.sell)),
            ("nonneg_buy", -solution.buy),
            ("nonneg_sell", -solution.sell),
            ("sell_le_w0", solution.sell - spec.w0),
            ("complementarity", np.minimum(solution.buy, solution.sell)),
        ]

    def contribution(self, portfolio_id: str, orders: pd.DataFrame) -> Contribution:
        """The BUY rows; sells never reach a later portfolio."""
        return Contribution.from_orders(portfolio_id, orders)

    def chain_state(self, spec: ProblemSpec, contributions: Sequence[Contribution]) -> ChainState:
        """Predecessors' buys, masked to this portfolio's buyable set."""
        return derive_chain_state(spec.security_ids, self.tradable(spec), contributions)


TWO_SIDED: SideProfile = TwoSided()

PROFILES: Mapping[str, SideProfile] = {TWO_SIDED.sides: TWO_SIDED}
"""Every profile a config may select, by its ``sides`` value; ``cvx/sides.py`` carries the matching identities."""


def profile_for(sides: str) -> SideProfile:
    """The profile ``sides`` selects; a value outside :data:`PROFILES` is a config error."""
    profile = PROFILES.get(sides)
    if profile is None:
        msg = f"sides {sides!r} is not one the engine knows; known: {sorted(PROFILES)}"
        raise ValueError(msg)
    return profile
