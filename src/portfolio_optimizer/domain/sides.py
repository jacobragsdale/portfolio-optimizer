"""Which side a run trades, as the one object that knows: the side profile.

A run is buy-only, sell-only, or two-sided; ``sides`` in the run config selects one. Everything
side-dependent lives here or in its cvxpy half, ``cvx/sides.py``: how a solver's weights become a
trade, which securities the dependency graph and the chain state are built from, what a dependent
solve receives, which starting books the side cannot trade out of, and which invariants the verifier
adds. Nothing else in the engine asks which side a run is; it consumes what the profile hands it.
This module is cvxpy-free so the verifier and the ``verify`` command can use it without the solver
stack.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from portfolio_optimizer.domain.results import F64, ChainState, Contribution, Flags, ProblemSpec, Solution, derive_chain_state

type Sides = Literal["both", "buy", "sell"]
"""The sides a run may trade: buy-only, sell-only, or the two-sided problem."""

TOLERANCE = 1e-12
"""Float noise, not policy: how far past a bound a starting weight may sit before the start is called infeasible."""


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

    def coupled(self, solution: Solution) -> F64:
        """The amount traded on the side the run couples through, per security; the numpy twin of ``x.coupled``."""
        ...

    def identity_residuals(self, spec: ProblemSpec, solution: Solution) -> list[tuple[str, F64]]:
        """Violations of the trade identity, one named residual vector per check; the verifier's twin of ``cvx/sides.py``."""
        ...

    def infeasible_starts(self, spec: ProblemSpec) -> list[str]:
        """Why this side cannot trade out of where the book starts, in words; empty when the start is not the problem."""
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

    def coupled(self, solution: Solution) -> F64:
        """The buys."""
        return solution.buy

    def identity_residuals(self, spec: ProblemSpec, solution: Solution) -> list[tuple[str, F64]]:
        """``w - w0 = buy - sell`` (an equality), both non-negative, ``sell ≤ w0``, and no name both bought and sold."""
        return [
            ("trade_balance", np.abs(solution.w - spec.w0 - solution.buy + solution.sell)),
            ("nonneg_buy", -solution.buy),
            ("nonneg_sell", -solution.sell),
            ("sell_le_w0", solution.sell - spec.w0),
            ("complementarity", np.minimum(solution.buy, solution.sell)),
        ]

    def infeasible_starts(self, spec: ProblemSpec) -> list[str]:
        """Nothing side-specific: a two-sided run can move any weight either way."""
        del spec
        return []

    def contribution(self, portfolio_id: str, orders: pd.DataFrame) -> Contribution:
        """The BUY rows; sells never reach a later portfolio."""
        return Contribution.from_orders(portfolio_id, orders, "BUY")

    def chain_state(self, spec: ProblemSpec, contributions: Sequence[Contribution]) -> ChainState:
        """Predecessors' buys, masked to this portfolio's buyable set."""
        return derive_chain_state(spec.security_ids, self.tradable(spec), contributions)


@dataclass(frozen=True, slots=True)
class BuyOnly:
    """Buys alone: ``w ≥ w0`` and ``buy = w - w0``; there is no ``sell``, so no wash trade is possible.

    A buy-only run can only lower cash, and a name whose cap sits below its holding cannot be traded
    out of: both are named as the infeasibilities they are rather than solved around.
    """

    sides: Sides = "buy"
    order_sides: frozenset[str] = frozenset({"BUY"})

    def tradable(self, spec: ProblemSpec) -> Flags:
        """The buyable set: ``ub > w0``."""
        return spec.buyable

    def split(self, w: F64, w0: F64) -> tuple[F64, F64]:
        """``buy = max(w - w0, 0)``, ``sell = 0``: the clip keeps solver noise below ``w0`` from becoming a sell of a few shares at a large NAV."""
        return np.maximum(w - w0, 0.0), np.zeros_like(w)

    def coupled(self, solution: Solution) -> F64:
        """The buys."""
        return solution.buy

    def identity_residuals(self, spec: ProblemSpec, solution: Solution) -> list[tuple[str, F64]]:
        """``w ≥ w0``, the reported buy is ``w - w0`` (an equality), it is non-negative, and the sell vector is zero."""
        return [
            ("no_sells", spec.w0 - solution.w),
            ("trade_balance", np.abs(solution.w - spec.w0 - solution.buy + solution.sell)),
            ("nonneg_buy", -solution.buy),
            ("sell_absent", np.abs(solution.sell)),
        ]

    def infeasible_starts(self, spec: ProblemSpec) -> list[str]:
        """Cash already below its floor (buys only lower it), and names held above their cap (buys cannot lower them)."""
        return [*_cash_start(spec, "buy-only", floor=True), *_bound_starts(spec, spec.w0 > spec.ub + TOLERANCE, "cap is below their holding")]

    def contribution(self, portfolio_id: str, orders: pd.DataFrame) -> Contribution:
        """The BUY rows, which is every row."""
        return Contribution.from_orders(portfolio_id, orders, "BUY")

    def chain_state(self, spec: ProblemSpec, contributions: Sequence[Contribution]) -> ChainState:
        """Predecessors' buys, masked to this portfolio's buyable set."""
        return derive_chain_state(spec.security_ids, self.tradable(spec), contributions)


@dataclass(frozen=True, slots=True)
class SellOnly:
    """Sells alone: ``w ≤ w0`` and ``sell = w0 - w``; there is no ``buy``. The mirror of :class:`BuyOnly`.

    Portfolios couple through sells: the tradable set is the sellable set, a contribution is the SELL
    rows, and the chain carries shares predecessors sold. A sell-only run can only raise cash, and a
    name held below its floor cannot be traded out of.
    """

    sides: Sides = "sell"
    order_sides: frozenset[str] = frozenset({"SELL"})

    def tradable(self, spec: ProblemSpec) -> Flags:
        """The sellable set: held, and ``lb < w0``."""
        return spec.sellable

    def split(self, w: F64, w0: F64) -> tuple[F64, F64]:
        """``buy = 0``, ``sell = max(w0 - w, 0)``: the clip keeps solver noise above ``w0`` from becoming a buy of a few shares at a large NAV."""
        return np.zeros_like(w), np.maximum(w0 - w, 0.0)

    def coupled(self, solution: Solution) -> F64:
        """The sells."""
        return solution.sell

    def identity_residuals(self, spec: ProblemSpec, solution: Solution) -> list[tuple[str, F64]]:
        """``w ≤ w0``, the reported sell is ``w0 - w`` (an equality), it is non-negative, and the buy vector is zero."""
        return [
            ("no_buys", solution.w - spec.w0),
            ("trade_balance", np.abs(solution.w - spec.w0 - solution.buy + solution.sell)),
            ("nonneg_sell", -solution.sell),
            ("buy_absent", np.abs(solution.buy)),
        ]

    def infeasible_starts(self, spec: ProblemSpec) -> list[str]:
        """Cash already above its cap (sells only raise it), and names held below their floor (sells cannot raise them)."""
        return [*_cash_start(spec, "sell-only", floor=False), *_bound_starts(spec, spec.w0 < spec.lb - TOLERANCE, "floor is above their holding")]

    def contribution(self, portfolio_id: str, orders: pd.DataFrame) -> Contribution:
        """The SELL rows, which is every row."""
        return Contribution.from_orders(portfolio_id, orders, "SELL")

    def chain_state(self, spec: ProblemSpec, contributions: Sequence[Contribution]) -> ChainState:
        """Predecessors' sells, masked to this portfolio's sellable set."""
        return derive_chain_state(spec.security_ids, self.tradable(spec), contributions)


def _cash_start(spec: ProblemSpec, run: str, *, floor: bool) -> list[str]:
    """The starting cash against the bound the side can only move away from."""
    cash = 1.0 - float(spec.w0.sum())
    if floor and cash < spec.cash_lb - TOLERANCE:
        return [f"the book starts with cash {cash:.6f} below cash_lb {spec.cash_lb:.6f}, and a {run} run can only lower cash"]
    if not floor and cash > spec.cash_ub + TOLERANCE:
        return [f"the book starts with cash {cash:.6f} above cash_ub {spec.cash_ub:.6f}, and a {run} run can only raise cash"]
    return []


def _bound_starts(spec: ProblemSpec, violated: Flags, how: str) -> list[str]:
    """Names whose starting weight already sits past a bound the side cannot move it back inside; ``how`` says which bound."""
    names = [security for security, flag in zip(spec.security_ids, violated, strict=True) if flag]
    if not names:
        return []
    return [f"names whose {how}, which this side cannot trade out of: {names}"]


TWO_SIDED: SideProfile = TwoSided()
BUY_ONLY: SideProfile = BuyOnly()
SELL_ONLY: SideProfile = SellOnly()

PROFILES: Mapping[Sides, SideProfile] = {profile.sides: profile for profile in (TWO_SIDED, BUY_ONLY, SELL_ONLY)}
"""Every profile a config may select, by its ``sides`` value; ``cvx/sides.py`` carries the matching variables and identities."""


def profile_for(sides: str) -> SideProfile:
    """The profile ``sides`` selects; ``sides`` is text because it may come from a manifest, and a value outside :data:`PROFILES` is an error."""
    for profile in PROFILES.values():
        if profile.sides == sides:
            return profile
    msg = f"sides {sides!r} is not one the engine knows; known: {sorted(PROFILES)}"
    raise ValueError(msg)
