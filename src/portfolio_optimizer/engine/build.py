"""Turn a validated ``PortfolioData`` into a ``ProblemSpec``: the one place Decimal becomes float64.

Everything money-like is computed exactly in Decimal first (current weights, tax per dollar) and
converted once through :func:`to_float64`. The spec is chain-independent; chain-aware constraints
combine it with a ``ChainState`` at solve time.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from portfolio_optimizer.domain.data import PortfolioData
from portfolio_optimizer.domain.results import F64, OrderInputs, ProblemSpec
from portfolio_optimizer.domain.schemas import UNIVERSE

LONG_TERM_HOLDING = timedelta(days=365)
"""Positions held strictly longer than this are taxed at the long-term rate."""

PSD_SHIFT_TOLERANCE = 1e-8
"""Reject a covariance whose most negative eigenvalue exceeds this fraction of its largest."""

BPS = Decimal(10_000)


class BuildError(ValueError):
    """The bundle cannot be turned into a well-formed problem."""


@dataclass(frozen=True, slots=True)
class BuildOutput:
    """The spec plus the exact inputs the order step needs."""

    spec: ProblemSpec
    order_inputs: OrderInputs


def to_float64(values: Sequence[Decimal | int], name: str) -> F64:
    """Convert exact values to float64 — correctly rounded, and the only sanctioned way to do so."""
    out = np.empty(len(values), dtype=np.float64)
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Decimal | int):
            msg = f"{name}[{index}] is {type(value).__name__}, expected Decimal or int"
            raise BuildError(msg)
        if isinstance(value, Decimal) and not value.is_finite():
            msg = f"{name}[{index}] is not finite: {value}"
            raise BuildError(msg)
        out[index] = float(value)
    if not np.isfinite(out).all():
        msg = f"{name} overflowed float64"
        raise BuildError(msg)
    return out


def build_problem_spec(data: PortfolioData) -> BuildOutput:
    """Align every input to the sorted universe and express it as a fraction of NAV."""
    universe = data.universe.sort_values("security_id", kind="stable").reset_index(drop=True)
    ids = tuple(str(value) for value in universe["security_id"])
    n = len(ids)
    nav = data.details.nav
    price = [_decimal(value) for value in universe["price"]]
    held = data.holdings.set_index("security_id")
    shares_held = [_int(held.loc[security, "quantity"]) if security in held.index else 0 for security in ids]
    lot_size = [int(value) for value in universe["lot_size"]]
    w0 = [Decimal(shares) * px / nav for shares, px in zip(shares_held, price, strict=True)]
    targets = {str(security): _decimal(weight) for security, weight in zip(data.targets["security_id"], data.targets["weight"], strict=True)}
    w_target = [targets.get(security, Decimal(0)) for security in ids]
    tax = [_tax_per_dollar(data, security, px) for security, px in zip(ids, price, strict=True)]
    tcost = [_decimal(value) / BPS for value in universe["tcost_bps"]] if "tcost_bps" in universe.columns else [Decimal(0)] * n
    lb, ub = _bounds(data, universe, ids, w0)
    sector_names = tuple(sorted({str(value) for value in universe["sector"]}))
    sectors = [str(value) for value in universe["sector"]]
    sector_matrix = np.array([[1.0 if sector == name else 0.0 for sector in sectors] for name in sector_names], dtype=np.float64).reshape(len(sector_names), n)
    sector_lb = [data.style.sector_bounds[name][0] if name in data.style.sector_bounds else Decimal(0) for name in sector_names]
    sector_ub = [data.style.sector_bounds[name][1] if name in data.style.sector_bounds else Decimal(1) for name in sector_names]
    adv_capacity = [data.style.max_adv_participation * Decimal(int(adv)) * px / nav for adv, px in zip(universe["adv_shares"], price, strict=True)]
    sigma_factor, psd_shift = _sigma_factor(data, ids)
    spec = ProblemSpec(
        portfolio_id=data.details.portfolio_id,
        as_of=data.as_of,
        security_ids=ids,
        sector_names=sector_names,
        nav=float(nav),
        w0=to_float64(w0, "w0"),
        price=to_float64(price, "price"),
        shares_held=to_float64(shares_held, "shares_held"),
        lot_size=to_float64(lot_size, "lot_size"),
        w_target=to_float64(w_target, "w_target"),
        tax_per_dollar=to_float64(tax, "tax_per_dollar"),
        tcost_per_dollar=to_float64(tcost, "tcost_per_dollar"),
        lb=to_float64(lb, "lb"),
        ub=to_float64(ub, "ub"),
        adv_capacity=to_float64(adv_capacity, "adv_capacity"),
        sector_matrix=sector_matrix,
        sector_lb=to_float64(sector_lb, "sector_lb"),
        sector_ub=to_float64(sector_ub, "sector_ub"),
        max_turnover=float(data.style.max_turnover),
        cash_lb=float(data.style.cash_bounds[0]),
        cash_ub=float(data.style.cash_bounds[1]),
        min_trade_notional=float(data.style.min_trade_notional),
        sigma_factor=sigma_factor,
        psd_shift=psd_shift,
        columns=_extra_columns(universe),
    )
    inputs = OrderInputs(security_ids=ids, price=tuple(price), shares_held=tuple(shares_held), lot_size=tuple(lot_size), nav=nav, min_trade_notional=data.style.min_trade_notional)
    return BuildOutput(spec=spec, order_inputs=inputs)


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        msg = f"expected an integer share count, got {type(value).__name__}"
        raise BuildError(msg)
    return int(value)


def _decimal(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Decimal(value)
    msg = f"expected Decimal, got {type(value).__name__}"
    raise BuildError(msg)


def _tax_per_dollar(data: PortfolioData, security: str, price: Decimal) -> Decimal:
    """Signed tax owed per dollar sold: gain fraction times the rate for the holding period. Losses are negative."""
    rows = data.holdings[data.holdings["security_id"] == security]
    if len(rows) == 0:
        return Decimal(0)
    avg_cost = _decimal(rows["avg_cost"].iloc[0])
    acquired = rows["acquired_on"].iloc[0].to_pydatetime()
    rate = data.details.lt_tax_rate if data.as_of - acquired > LONG_TERM_HOLDING else data.details.st_tax_rate
    return (price - avg_cost) / price * rate


def _bounds(data: PortfolioData, universe: pd.DataFrame, ids: tuple[str, ...], w0: list[Decimal]) -> tuple[list[Decimal], list[Decimal]]:
    """Long-only floor and style cap, tightened by optional per-security columns; restricted names are frozen."""
    floors = [_optional_decimal(value) for value in universe["min_weight"]] if "min_weight" in universe.columns else [None] * len(ids)
    caps = [_optional_decimal(value) for value in universe["max_weight"]] if "max_weight" in universe.columns else [None] * len(ids)
    lb: list[Decimal] = []
    ub: list[Decimal] = []
    for index, (security, restricted) in enumerate(zip(ids, universe["restricted"], strict=True)):
        if bool(restricted):
            lb.append(w0[index])
            ub.append(w0[index])
            continue
        floor = floors[index]
        cap = caps[index]
        low = Decimal(0) if floor is None else max(Decimal(0), floor)
        high = data.style.max_weight if cap is None else min(data.style.max_weight, cap)
        if low > high:
            msg = f"{security}: lower bound {low} exceeds upper bound {high}"
            raise BuildError(msg)
        lb.append(low)
        ub.append(high)
    return lb, ub


def _optional_decimal(value: object) -> Decimal | None:
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    return _decimal(value)


def _sigma_factor(data: PortfolioData, ids: tuple[str, ...]) -> tuple[F64 | None, float]:
    """Factor the covariance as ``FᵀF = Σ`` after projecting to the PSD cone; record how much was clipped."""
    if data.covariance is None:
        return None, 0.0
    position = {security: index for index, security in enumerate(ids)}
    sigma = np.zeros((len(ids), len(ids)), dtype=np.float64)
    for a, b, value in zip(data.covariance["security_id_a"], data.covariance["security_id_b"], data.covariance["covariance"], strict=True):
        if str(a) in position and str(b) in position:
            sigma[position[str(a)], position[str(b)]] = float(value)
    sigma = (sigma + sigma.T) / 2.0
    with threadpool_limits(limits=1):  # multithreaded BLAS can change the last bits, and with them the spec hash
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    largest = float(eigenvalues.max(initial=0.0))
    psd_shift = float(max(0.0, -eigenvalues.min(initial=0.0)))
    if psd_shift > PSD_SHIFT_TOLERANCE * max(largest, np.finfo(np.float64).tiny):
        msg = f"covariance is not positive semidefinite: most negative eigenvalue {-psd_shift:.3e} against largest {largest:.3e}"
        raise BuildError(msg)
    clipped = np.clip(eigenvalues, 0.0, None)
    factor = (np.sqrt(clipped)[:, None] * eigenvectors.T).astype(np.float64)
    return factor, psd_shift


def _extra_columns(universe: pd.DataFrame) -> dict[str, F64]:
    """Export every numeric column the schema does not declare, by name, for custom terms."""
    declared = {column.name for column in UNIVERSE.columns}
    exported: dict[str, F64] = {}
    for name in universe.columns:
        column_name = str(name)
        if column_name in declared and column_name != "alpha":
            continue
        column = universe[column_name]
        if column_name == "alpha" or pd.api.types.is_numeric_dtype(column.dtype) or column.dtype == "object":
            values = _numeric_values(column, column_name)
            if values is not None:
                exported[column_name] = values
    return exported


def _numeric_values(column: pd.Series, name: str) -> F64 | None:
    if column.dtype == "object":
        if not all(isinstance(value, Decimal | int) and not isinstance(value, bool) for value in column):
            return None
        return to_float64([_decimal(value) for value in column], name)
    if column.isna().any():
        msg = f"column {name!r} has null values; fill them in a rule before the optimizer runs"
        raise BuildError(msg)
    values = column.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        msg = f"column {name!r} has non-finite values"
        raise BuildError(msg)
    return np.asarray(values, dtype=np.float64)
