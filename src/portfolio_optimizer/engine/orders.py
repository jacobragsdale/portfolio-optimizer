"""Turn a solution into whole-share orders: the one place float64 becomes Decimal again.

The trade is the split the order-flow profile reported, ``buy - sell``: ``w - w0`` clipped to the side the
run has, so solver noise on the other side never becomes an order. Weight deltas become whole shares by rounding to the
**nearest** share (half-even), then down to a lot multiple, then clamped so a sell never exceeds
what is held and a buy never exceeds the room under the security's upper bound. Nearest rounding
matters because solver noise of 1e-8 in weight space is a fraction of a share: rounding toward zero
would turn an exact 1250-share answer into 1249. The buy clamp makes ``spec.buyable`` structural:
the verifier tolerates a bound violation of ``violation_tol``, which at a large NAV is whole shares,
and a BUY in a security the portfolio cannot buy would be invisible to the dependency graph. The
drift this introduces is measured against the solved weights and bounded by :func:`rounding_drift`,
a diagnostic; what decides whether the orders may go out is that the book they leave —
:func:`executed_solution` — passes the same verification the solved weights did. Rounding can breach
a bound the solver sat on, by a fraction of a share; a row's ``tolerance`` is where a desk says how
much of that it accepts, and the engine adds none of its own.
"""

from dataclasses import replace
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal

import numpy as np
import pandas as pd

from portfolio_optimizer.domain.frames import validate_frame
from portfolio_optimizer.domain.order_flow import OrderFlowProfile
from portfolio_optimizer.domain.results import F64, DriftReport, OrderInputs, ProblemSpec, Solution
from portfolio_optimizer.domain.schemas import ORDERS


def solution_to_orders(spec: ProblemSpec, solution: Solution, inputs: OrderInputs, *, run_id: str) -> pd.DataFrame:
    """Whole-share orders for every name whose rounded delta survives the lot and dust filters."""
    if inputs.security_ids != spec.security_ids:
        msg = "order inputs are not aligned to the spec"
        raise ValueError(msg)
    rows: list[dict[str, object]] = []
    for index, security in enumerate(spec.security_ids):
        quantity, unrounded = _shares(index, solution, inputs)
        if quantity == 0:
            continue
        price = inputs.price[index]
        notional = Decimal(abs(quantity)) * price
        if notional < inputs.min_trade_notional:
            continue
        rows.append(
            {
                "portfolio_id": spec.portfolio_id,
                "security_id": security,
                "side": "BUY" if quantity > 0 else "SELL",
                "quantity": abs(quantity),
                "reference_price": price,
                "notional": notional,
                "target_weight": float(solution.w[index]),
                "unrounded_shares": unrounded,
                "spec_hash": solution.spec_hash,
                "run_id": run_id,
                "as_of_date": spec.as_of_date,
            }
        )
    return validate_frame(_orders_frame(rows), ORDERS)


def _shares(index: int, solution: Solution, inputs: OrderInputs) -> tuple[int, float]:
    """Signed whole shares for one name after nearest-share, lot, held-quantity, and bound rounding."""
    delta = Decimal(float(solution.buy[index] - solution.sell[index])) * inputs.nav / inputs.price[index]
    unrounded = float(delta)
    lot = inputs.lot_size[index]
    magnitude = int(abs(delta).to_integral_value(rounding=ROUND_HALF_EVEN))
    magnitude -= magnitude % lot
    if delta < 0:
        magnitude = min(magnitude, inputs.shares_held[index])
        magnitude -= magnitude % lot
        return -magnitude, unrounded
    room = int(((inputs.ub[index] - inputs.w0[index]) * inputs.nav / inputs.price[index]).to_integral_value(rounding=ROUND_FLOOR))
    magnitude = min(magnitude, max(room, 0))
    magnitude -= magnitude % lot
    return magnitude, unrounded


def _orders_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Assemble the frame one typed column at a time: no record scan, no cast pass, and an empty frame keeps every dtype."""
    return pd.DataFrame({column.name: pd.Series([row[column.name] for row in rows], dtype=column.dtype) for column in ORDERS.columns})


def executed_weights(spec: ProblemSpec, orders: pd.DataFrame) -> F64:
    """The weights the orders leave the book at: shares held plus the signed order quantities, times price, over NAV, from the spec's own vectors."""
    signed = np.zeros(spec.n)
    positions = {security: index for index, security in enumerate(spec.security_ids)}
    for security, side, quantity in zip(orders["security_id"], orders["side"], orders["quantity"], strict=True):
        signed[positions[str(security)]] += int(quantity) if str(side) == "BUY" else -int(quantity)
    return (spec.shares_held + signed) * spec.price / spec.nav


def executed_solution(spec: ProblemSpec, solution: Solution, orders: pd.DataFrame, profile: OrderFlowProfile) -> Solution:
    """The executed book as a solution the verifier checks exactly like the solver's: the weights the orders leave, the profile's split of them, no objective and no duals."""
    w = executed_weights(spec, orders)
    buy, sell = profile.split(w, spec.w0)
    return replace(solution, w=w, buy=buy, sell=sell, objective=None, duals={})


def rounding_drift(spec: ProblemSpec, solution: Solution, orders: pd.DataFrame, inputs: OrderInputs, *, violation_tol: float = 0.0) -> DriftReport:
    """Measure how far the executed weights sit from the solved weights.

    The tolerance is what rounding can cost by construction: one lot of the priciest name plus one
    dust-filtered trade, both as fractions of NAV, plus ``violation_tol`` — the bound overshoot the
    verifier accepts, which the buy clamp in :func:`solution_to_orders` may take back.
    """
    error = float(np.abs(executed_weights(spec, orders) - solution.w).max(initial=0.0))
    lot_cost = max((float(Decimal(lot) * price / inputs.nav) for lot, price in zip(inputs.lot_size, inputs.price, strict=True)), default=0.0)
    dust_cost = float(inputs.min_trade_notional / inputs.nav)
    traded = {str(security) for security in orders["security_id"]}
    dropped = sum(1 for i, security in enumerate(spec.security_ids) if security not in traded and abs(solution.w[i] - spec.w0[i]) * spec.nav >= float(inputs.price[i]))
    return DriftReport(max_weight_error=error, tolerance=lot_cost + dust_cost + violation_tol + 1e-9, dropped_orders=dropped)
