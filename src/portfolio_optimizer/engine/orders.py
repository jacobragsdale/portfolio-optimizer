"""Turn a solution into whole-share orders: the one place float64 becomes Decimal again.

Weight deltas become whole shares by rounding to the **nearest** share (half-even), then down to a
lot multiple, then clamped so a sell never exceeds what is held and a buy never exceeds the room
under the security's upper bound. Nearest rounding matters because solver noise of 1e-8 in weight
space is a fraction of a share: rounding toward zero would turn an exact 1250-share answer into
1249. The buy clamp makes ``spec.buyable`` structural: the verifier tolerates a bound violation of
``violation_tol``, which at a large NAV is whole shares, and a BUY in a security the portfolio
cannot buy would be invisible to the dependency graph. The drift this introduces is measured
against the solved weights and bounded by :func:`rounding_drift`; verification of every constraint
happens on the solved weights before rounding.
"""

from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal

import numpy as np
import pandas as pd

from portfolio_optimizer.domain.frames import validate_frame
from portfolio_optimizer.domain.results import DriftReport, OrderInputs, ProblemSpec, Solution
from portfolio_optimizer.domain.schemas import ORDERS


def solution_to_orders(spec: ProblemSpec, solution: Solution, inputs: OrderInputs, *, run_id: str) -> pd.DataFrame:
    """Whole-share orders for every name whose rounded delta survives the lot and dust filters."""
    if inputs.security_ids != spec.security_ids:
        msg = "order inputs are not aligned to the spec"
        raise ValueError(msg)
    rows: list[dict[str, object]] = []
    for index, security in enumerate(spec.security_ids):
        quantity, unrounded = _shares(index, spec, solution, inputs)
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
                "as_of": spec.as_of,
            }
        )
    return validate_frame(_orders_frame(rows), ORDERS)


def _shares(index: int, spec: ProblemSpec, solution: Solution, inputs: OrderInputs) -> tuple[int, float]:
    """Signed whole shares for one name after nearest-share, lot, held-quantity, and bound rounding."""
    delta = Decimal(float(solution.w[index] - spec.w0[index])) * inputs.nav / inputs.price[index]
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


def rounding_drift(spec: ProblemSpec, solution: Solution, orders: pd.DataFrame, inputs: OrderInputs, *, violation_tol: float = 0.0) -> DriftReport:
    """Rebuild the executed weights from the orders and measure their distance from the solved weights.

    The tolerance is what rounding can cost by construction: one lot of the priciest name plus one
    dust-filtered trade, both as fractions of NAV, plus ``violation_tol`` — the bound overshoot the
    verifier accepts, which the buy clamp in :func:`solution_to_orders` may take back.
    """
    signed: dict[str, int] = {}
    for security, side, quantity in zip(orders["security_id"], orders["side"], orders["quantity"], strict=True):
        signed[str(security)] = int(quantity) if str(side) == "BUY" else -int(quantity)
    executed = np.array([float((Decimal(inputs.shares_held[i] + signed.get(security, 0)) * inputs.price[i]) / inputs.nav) for i, security in enumerate(spec.security_ids)], dtype=np.float64)
    error = float(np.abs(executed - solution.w).max(initial=0.0))
    lot_cost = max((float(Decimal(lot) * price / inputs.nav) for lot, price in zip(inputs.lot_size, inputs.price, strict=True)), default=0.0)
    dust_cost = float(inputs.min_trade_notional / inputs.nav)
    traded = {str(security) for security in orders["security_id"]}
    dropped = sum(1 for i, security in enumerate(spec.security_ids) if security not in traded and abs(solution.w[i] - spec.w0[i]) * spec.nav >= float(inputs.price[i]))
    return DriftReport(max_weight_error=error, tolerance=lot_cost + dust_cost + violation_tol + 1e-9, dropped_orders=dropped)
