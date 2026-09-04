"""Turn a solution into executable orders: the one place float64 becomes Decimal again.

The trade is the split the order-flow profile reported, ``buy - sell``: ``w - w0`` clipped to the side the
run has, so solver noise on the other side never becomes an order. A weight delta becomes a quantity
at the dirty price, rounded to the **nearest** unit (half-even), then down to a multiple of the
security's ``increment``, then clamped so a sell never exceeds what is held and a buy never exceeds
the room under the security's upper bound, and finally held back from leaving a position below the
security's ``min_denomination``: a buy that cannot reach the minimum piece is dropped, a sell that
would leave less than it is shrunk to leave exactly it, and a sell of the whole position goes out as
it is. Nothing trades more than the solver asked. Nearest rounding matters because solver noise of
1e-8 in weight space is a fraction of a unit: rounding toward zero would turn an exact 1250-unit
answer into 1249. The buy clamp makes ``spec.buyable`` structural: the verifier tolerates a bound
violation of ``violation_tol``, which at a large NAV is whole units, and a BUY in a security the
portfolio cannot buy would be invisible to the dependency graph. The
drift this introduces is measured against the solved weights and bounded by :func:`rounding_drift`,
a diagnostic; what decides whether the orders may go out is that the book they leave —
:func:`executed_solution` — passes the same verification the solved weights did. Rounding can breach
a bound the solver sat on, by a fraction of a unit; a row's ``tolerance`` is where a desk says how
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
    """Executable orders for every name whose rounded delta survives the increment, minimum-piece, and dust filters."""
    if inputs.security_ids != spec.security_ids:
        msg = "order inputs are not aligned to the spec"
        raise ValueError(msg)
    rows: list[dict[str, object]] = []
    for index, security in enumerate(spec.security_ids):
        quantity, unrounded = _quantity(index, solution, inputs)
        if quantity == 0:
            continue
        price, accrued = inputs.price[index], inputs.accrued_interest[index]
        notional = Decimal(abs(quantity)) * (price + accrued)
        if notional < inputs.min_trade_notional:
            continue
        rows.append(
            {
                "portfolio_id": spec.portfolio_id,
                "security_id": security,
                "side": "BUY" if quantity > 0 else "SELL",
                "quantity": abs(quantity),
                "reference_price": price,
                "accrued_interest": accrued,
                "notional": notional,
                "target_weight": float(solution.w[index]),
                "unrounded_quantity": unrounded,
                "spec_hash": solution.spec_hash,
                "run_id": run_id,
                "as_of_date": spec.as_of_date,
            }
        )
    return validate_frame(_orders_frame(rows), ORDERS)


def _quantity(index: int, solution: Solution, inputs: OrderInputs) -> tuple[int, float]:
    """Signed executable quantity for one name: nearest unit, down to the increment, clamped to what is held or to the room under the bound, and never leaving a position below the minimum piece."""
    dirty = inputs.price[index] + inputs.accrued_interest[index]
    delta = Decimal(float(solution.buy[index] - solution.sell[index])) * inputs.nav / dirty
    unrounded = float(delta)
    increment, held, minimum = inputs.increment[index], inputs.quantity_held[index], inputs.min_denomination[index]
    magnitude = int(abs(delta).to_integral_value(rounding=ROUND_HALF_EVEN))
    if delta < 0:
        if magnitude >= held:
            return -held, unrounded  # a full liquidation goes out as the position is, on or off the increment grid
        magnitude -= magnitude % increment
        if 0 < held - magnitude < minimum:
            # ponytail: a sell that would leave a stub is shrunk to leave exactly the minimum piece; a desk that
            # would rather sell out changes this one branch. An odd lot below the minimum is held as it is.
            magnitude = max(held - minimum, 0)
            magnitude -= magnitude % increment
        return -magnitude, unrounded
    magnitude -= magnitude % increment
    room = int(((inputs.ub[index] - inputs.w0[index]) * inputs.nav / dirty).to_integral_value(rounding=ROUND_FLOOR))
    magnitude = min(magnitude, max(room, 0))
    magnitude -= magnitude % increment
    if 0 < held + magnitude < minimum:
        magnitude = 0  # buying less cannot reach the minimum piece, and buying more is not what was solved
    return magnitude, unrounded


def _orders_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Assemble the frame one typed column at a time: no record scan, no cast pass, and an empty frame keeps every dtype."""
    return pd.DataFrame({column.name: pd.Series([row[column.name] for row in rows], dtype=column.dtype) for column in ORDERS.columns})


def executed_weights(spec: ProblemSpec, orders: pd.DataFrame) -> F64:
    """The weights the orders leave the book at: the quantity held plus the signed order quantities, times price, over NAV, from the spec's own vectors."""
    signed = np.zeros(spec.n)
    positions = {security: index for index, security in enumerate(spec.security_ids)}
    for security, side, quantity in zip(orders["security_id"], orders["side"], orders["quantity"], strict=True):
        signed[positions[str(security)]] += int(quantity) if str(side) == "BUY" else -int(quantity)
    return (spec.quantity_held + signed) * spec.price / spec.nav


def executed_solution(spec: ProblemSpec, solution: Solution, orders: pd.DataFrame, profile: OrderFlowProfile) -> Solution:
    """The executed book as a solution the verifier checks exactly like the solver's: the weights the orders leave, the profile's split of them, no objective and no duals."""
    w = executed_weights(spec, orders)
    buy, sell = profile.split(w, spec.w0)
    return replace(solution, w=w, buy=buy, sell=sell, objective=None, duals={})


def rounding_drift(spec: ProblemSpec, solution: Solution, orders: pd.DataFrame, inputs: OrderInputs, *, violation_tol: float = 0.0) -> DriftReport:
    """Measure how far the executed weights sit from the solved weights.

    The tolerance is what rounding can cost by construction: one increment plus one minimum piece of
    the dearest name — a trade held back from leaving a stub moves the book by up to the piece — plus
    one dust-filtered trade, all as fractions of NAV, plus ``violation_tol`` — the bound overshoot the
    verifier accepts, which the buy clamp in :func:`solution_to_orders` may take back.
    """
    error = float(np.abs(executed_weights(spec, orders) - solution.w).max(initial=0.0))
    lot_cost = max(
        (
            float(Decimal(increment + minimum) * (price + accrued) / inputs.nav)
            for increment, minimum, price, accrued in zip(inputs.increment, inputs.min_denomination, inputs.price, inputs.accrued_interest, strict=True)
        ),
        default=0.0,
    )
    dust_cost = float(inputs.min_trade_notional / inputs.nav)
    traded = {str(security) for security in orders["security_id"]}
    dropped = sum(1 for i, security in enumerate(spec.security_ids) if security not in traded and abs(solution.w[i] - spec.w0[i]) * spec.nav >= spec.price[i])
    return DriftReport(max_weight_error=error, tolerance=lot_cost + dust_cost + violation_tol + 1e-9, dropped_orders=dropped)
