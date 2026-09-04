"""The run's order flow, as the one object that knows what it means: the order-flow profile.

A run is an inflow, an outflow, or a rebalance; ``order_flow`` in the run config says which. An
inflow puts cash into the book, so it buys; an outflow takes cash out, so it sells; a rebalance moves
no cash on purpose and may do either, which is what a book that starts outside its bounds needs. A
desk's order flows are separate runs over one snapshot, each a pure function of its inputs with its
own manifest, and nothing crosses between them inside the engine. Everything that depends on the order flow lives here or in
its cvxpy half, ``cvx/order_flow.py``: how a solver's weights become a trade, which securities the
dependency graph and the chain state are built from, what a dependent solve receives, which starting
books the order flow cannot trade out of, and which invariants the verifier adds. Nothing else in the
engine asks which order flow a run is; it consumes what the profile hands it. This module is
cvxpy-free so the verifier and the ``verify`` command can use it without the solver stack.

One variable per name is the whole of the design, under every order flow: the trade is a function
of the weight alone — affine under an inflow or an outflow, ``pos(w - w0)`` and ``pos(w0 - w)``
under a rebalance — so no name can be bought and sold in one solve. Under an inflow or an outflow
a term that rewards selling — a harvestable loss — is therefore exact; under a rebalance the same
term is a reward on a convex quantity, which is not convex, and the shipped ``linear`` kind refuses
it by name rather than letting the solver fail. The two-sided problem, with ``buy`` and ``sell`` as
independent variables, was what made a rewarded round trip possible; it was removed on 2026-09-01
and the rebalance profile is not it.

The identity every profile adds includes the spec's own box, ``lb ≤ w ≤ ub``: the bounds the build
derived from the style cap, the restricted flags, and any per-security floor or cap column. They are
structural rather than a constraint row because the schedule and the order rounding already assume
them — the buyable set is ``ub > w0``, and a buy is clamped to the room under ``ub``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from portfolio_optimizer.domain.results import F64, ChainState, Contribution, Flags, ProblemSpec, Solution, derive_chain_state

type OrderFlow = Literal["inflow", "outflow", "rebalance"]
"""The run's order flow: an inflow buys, an outflow sells, a rebalance may do either."""

TOLERANCE = 1e-12
"""Float noise, not policy: how far past a bound a starting weight may sit before the start is called infeasible."""


class OrderFlowProfile(Protocol):
    """What the engine needs to know about the run's order flow."""

    @property
    def order_flow(self) -> OrderFlow:
        """The config value this profile answers to."""
        ...

    @property
    def order_sides(self) -> frozenset[str]:
        """The order sides a solved portfolio may produce."""
        ...

    def tradable(self, spec: ProblemSpec) -> Flags:
        """The securities this portfolio can trade under this order flow; the dependency graph and the chain are built from it."""
        ...

    def split(self, w: F64, w0: F64) -> tuple[F64, F64]:
        """The ``(buy, sell)`` the engine reports for solved weights ``w``."""
        ...

    def coupled(self, solution: Solution) -> F64:
        """The amount traded on the side the run couples through, per security; the numpy twin of ``x.coupled``."""
        ...

    def identity_residuals(self, spec: ProblemSpec, solution: Solution) -> list[tuple[str, F64]]:
        """Violations of the trade identity and the spec's box, one named residual vector per check; the verifier's twin of ``cvx/order_flow.py``."""
        ...

    def infeasible_starts(self, spec: ProblemSpec) -> list[str]:
        """Why this order flow cannot trade out of where the book starts, in words; empty when the start is not the problem."""
        ...

    def contribution(self, portfolio_id: str, orders: pd.DataFrame) -> Contribution:
        """What a dependent solve receives from this portfolio's orders."""
        ...

    def chain_state(self, spec: ProblemSpec, contributions: Sequence[Contribution], consumes: Flags) -> ChainState:
        """Predecessors' contributions folded onto this spec, masked to its tradable set *and* to ``consumes`` — the securities its own chain readers can see.

        The build derives ``consumes`` from the portfolio's typed constraints (the whole tradable set
        when anything opaque might read the chain); masking to it is what keeps the chain state — and
        its hash — identical whether every earlier portfolio was folded or only the overlapping ones.
        """
        ...


def _box_residuals(spec: ProblemSpec, solution: Solution) -> list[tuple[str, F64]]:
    """``lb ≤ w ≤ ub``: the spec's own bounds, which every profile holds."""
    return [("lb", spec.lb - solution.w), ("ub", solution.w - spec.ub)]


@dataclass(frozen=True, slots=True)
class Inflow:
    """An inflow buys alone: ``w ≥ w0`` and ``buy = w - w0``; there is no ``sell``.

    Portfolios couple through buys: the tradable set is the buyable set, a contribution is the BUY
    rows, and the chain carries the quantity predecessors bought. An inflow can only lower cash, and a name
    whose cap sits below its holding cannot be traded out of: both are named as the infeasibilities
    they are rather than solved around.
    """

    order_flow: OrderFlow = "inflow"
    order_sides: frozenset[str] = frozenset({"BUY"})

    def tradable(self, spec: ProblemSpec) -> Flags:
        """The buyable set: ``ub > w0``."""
        return spec.buyable

    def split(self, w: F64, w0: F64) -> tuple[F64, F64]:
        """``buy = max(w - w0, 0)``, ``sell = 0``: the clip keeps solver noise below ``w0`` from becoming a sell of a few units at a large NAV."""
        return np.maximum(w - w0, 0.0), np.zeros_like(w)

    def coupled(self, solution: Solution) -> F64:
        """The buys."""
        return solution.buy

    def identity_residuals(self, spec: ProblemSpec, solution: Solution) -> list[tuple[str, F64]]:
        """``w ≥ w0``, the reported buy is ``w - w0`` (an equality), it is non-negative, the sell vector is zero, and the box."""
        return [
            ("no_sells", spec.w0 - solution.w),
            ("trade_balance", np.abs(solution.w - spec.w0 - solution.buy + solution.sell)),
            ("nonneg_buy", -solution.buy),
            ("sell_absent", np.abs(solution.sell)),
            *_box_residuals(spec, solution),
        ]

    def infeasible_starts(self, spec: ProblemSpec) -> list[str]:
        """Cash already below its floor (buys only lower it), and names held above their cap (buys cannot lower them)."""
        return [*_cash_start(spec, "an inflow", floor=True), *_bound_starts(spec, spec.w0 > spec.ub + TOLERANCE, "cap is below their holding")]

    def contribution(self, portfolio_id: str, orders: pd.DataFrame) -> Contribution:
        """The BUY rows, which is every row."""
        return Contribution.from_orders(portfolio_id, orders, "BUY")

    def chain_state(self, spec: ProblemSpec, contributions: Sequence[Contribution], consumes: Flags) -> ChainState:
        """Predecessors' buys, masked to this portfolio's buyable set and to what its chain readers consume."""
        return derive_chain_state(spec.security_ids, self.tradable(spec) & consumes, contributions)


@dataclass(frozen=True, slots=True)
class Outflow:
    """An outflow sells alone: ``w ≤ w0`` and ``sell = w0 - w``; there is no ``buy``. The mirror of :class:`Inflow`.

    Portfolios couple through sells: the tradable set is the sellable set, a contribution is the SELL
    rows, and the chain carries the quantity predecessors sold. An outflow can only raise cash, and a name
    held below its floor cannot be traded out of.
    """

    order_flow: OrderFlow = "outflow"
    order_sides: frozenset[str] = frozenset({"SELL"})

    def tradable(self, spec: ProblemSpec) -> Flags:
        """The sellable set: held, and ``lb < w0``."""
        return spec.sellable

    def split(self, w: F64, w0: F64) -> tuple[F64, F64]:
        """``buy = 0``, ``sell = max(w0 - w, 0)``: the clip keeps solver noise above ``w0`` from becoming a buy of a few units at a large NAV."""
        return np.zeros_like(w), np.maximum(w0 - w, 0.0)

    def coupled(self, solution: Solution) -> F64:
        """The sells."""
        return solution.sell

    def identity_residuals(self, spec: ProblemSpec, solution: Solution) -> list[tuple[str, F64]]:
        """``w ≤ w0``, the reported sell is ``w0 - w`` (an equality), it is non-negative, the buy vector is zero, and the box."""
        return [
            ("no_buys", solution.w - spec.w0),
            ("trade_balance", np.abs(solution.w - spec.w0 - solution.buy + solution.sell)),
            ("nonneg_sell", -solution.sell),
            ("buy_absent", np.abs(solution.buy)),
            *_box_residuals(spec, solution),
        ]

    def infeasible_starts(self, spec: ProblemSpec) -> list[str]:
        """Cash already above its cap (sells only raise it), and names held below their floor (sells cannot raise them)."""
        return [*_cash_start(spec, "an outflow", floor=False), *_bound_starts(spec, spec.w0 < spec.lb - TOLERANCE, "floor is above their holding")]

    def contribution(self, portfolio_id: str, orders: pd.DataFrame) -> Contribution:
        """The SELL rows, which is every row."""
        return Contribution.from_orders(portfolio_id, orders, "SELL")

    def chain_state(self, spec: ProblemSpec, contributions: Sequence[Contribution], consumes: Flags) -> ChainState:
        """Predecessors' sells, masked to this portfolio's sellable set and to what its chain readers consume."""
        return derive_chain_state(spec.security_ids, self.tradable(spec) & consumes, contributions)


@dataclass(frozen=True, slots=True)
class Rebalance:
    """A rebalance may buy and sell: ``w`` moves anywhere in the box, ``buy = max(w - w0, 0)`` and ``sell = max(w0 - w, 0)``.

    Still one variable per name, so a name is bought or sold, never both. Portfolios couple through
    every trade: the tradable set is the buyable set and the sellable set together, a contribution is
    every order row, and the chain carries the quantity predecessors traded on either side — a buy and a
    sell both spend a name's daily volume. No start is one a rebalance cannot trade out of: a name over
    its cap is sold down, a book below its cash floor raises cash, which is why a failed inflow or
    outflow is retried as one.
    """

    order_flow: OrderFlow = "rebalance"
    order_sides: frozenset[str] = frozenset({"BUY", "SELL"})

    def tradable(self, spec: ProblemSpec) -> Flags:
        """The buyable set and the sellable set together."""
        return spec.buyable | spec.sellable

    def split(self, w: F64, w0: F64) -> tuple[F64, F64]:
        """``buy = max(w - w0, 0)``, ``sell = max(w0 - w, 0)``."""
        delta = w - w0
        return np.maximum(delta, 0.0), np.maximum(-delta, 0.0)

    def coupled(self, solution: Solution) -> F64:
        """The trade, on whichever side: ``buy + sell``."""
        return solution.buy + solution.sell

    def identity_residuals(self, spec: ProblemSpec, solution: Solution) -> list[tuple[str, F64]]:
        """``w - w0 = buy - sell`` (an equality), both non-negative, no name on both sides, and the box."""
        return [
            ("trade_balance", np.abs(solution.w - spec.w0 - solution.buy + solution.sell)),
            ("nonneg_buy", -solution.buy),
            ("nonneg_sell", -solution.sell),
            ("no_round_trip", np.minimum(solution.buy, solution.sell)),
            *_box_residuals(spec, solution),
        ]

    def infeasible_starts(self, spec: ProblemSpec) -> list[str]:
        """Nothing: a rebalance can move any weight either way, so no start is its own infeasibility."""
        del spec
        return []

    def contribution(self, portfolio_id: str, orders: pd.DataFrame) -> Contribution:
        """Every row, BUY and SELL alike."""
        return Contribution.from_orders(portfolio_id, orders)

    def chain_state(self, spec: ProblemSpec, contributions: Sequence[Contribution], consumes: Flags) -> ChainState:
        """Predecessors' trades on either side, masked to this portfolio's tradable set and to what its chain readers consume."""
        return derive_chain_state(spec.security_ids, self.tradable(spec) & consumes, contributions)


def _cash_start(spec: ProblemSpec, run: str, *, floor: bool) -> list[str]:
    """The starting cash against the bound the order flow can only move away from, where the spec carries one."""
    cash = 1.0 - float(spec.w0.sum())
    lower, upper = spec.scalars.get("cash_lb"), spec.scalars.get("cash_ub")
    if floor and lower is not None and cash < lower - TOLERANCE:
        return [f"the book starts with cash {cash:.6f} below cash_lb {lower:.6f}, and {run} can only lower cash"]
    if not floor and upper is not None and cash > upper + TOLERANCE:
        return [f"the book starts with cash {cash:.6f} above cash_ub {upper:.6f}, and {run} can only raise cash"]
    return []


def _bound_starts(spec: ProblemSpec, violated: Flags, how: str) -> list[str]:
    """Names whose starting weight already sits past a bound the order flow cannot move it back inside; ``how`` says which bound."""
    names = [security for security, flag in zip(spec.security_ids, violated, strict=True) if flag]
    if not names:
        return []
    return [f"names whose {how}, which this order flow cannot trade out of: {names}"]


INFLOW: OrderFlowProfile = Inflow()
OUTFLOW: OrderFlowProfile = Outflow()
REBALANCE: OrderFlowProfile = Rebalance()

PROFILES: Mapping[OrderFlow, OrderFlowProfile] = {profile.order_flow: profile for profile in (INFLOW, OUTFLOW, REBALANCE)}
"""Every profile a config may select, by its ``order_flow`` value; ``cvx/order_flow.py`` carries the matching variables and identities."""


def profile_for(order_flow: str) -> OrderFlowProfile:
    """The profile ``order_flow`` selects; text because it may come from a manifest, and a value outside :data:`PROFILES` is an error."""
    for profile in PROFILES.values():
        if profile.order_flow == order_flow:
            return profile
    msg = f"order_flow {order_flow!r} is not one the engine knows; known: {sorted(PROFILES)}"
    raise ValueError(msg)
