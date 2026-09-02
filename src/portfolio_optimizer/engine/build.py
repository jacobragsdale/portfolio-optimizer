"""The shipped build step: turn a validated ``PortfolioData`` into a ``ProblemSpec``, the one place Decimal becomes float64.

``build`` is a configured step kind, ``(data: PortfolioData[, params]) -> ProblemSpec``, and
:func:`standard` is the default. Everything money-like is computed exactly in Decimal first
(current weights, tax per dollar) and converted once through :func:`to_float64`. The spec is
chain-independent; chain-aware constraints combine it with a ``ChainState`` at solve time. What the
bundle carries beyond the schemas is exported by name: every extra numeric universe column as a
column, every boolean one as a flag, every string one as a grouping, and every number on the
account's details row as a scalar. A build of your own — tax lots, a factor block, a different
bounds policy — is a function of the same shape named in the config; the engine derives the exact
order inputs from whatever spec it returns.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
from scipy.sparse import csc_array, csr_array

from portfolio_optimizer.domain.data import PortfolioData
from portfolio_optimizer.domain.results import F64, Flags, Grouping, OrderInputs, ProblemSpec
from portfolio_optimizer.domain.schemas import UNIVERSE

LONG_TERM_HOLDING = timedelta(days=365)
"""Positions held strictly longer than this are taxed at the long-term rate."""

BPS = Decimal(10_000)


class BuildError(ValueError):
    """The bundle cannot be turned into a well-formed problem."""


@dataclass(frozen=True, slots=True)
class _Position:
    quantity: int
    avg_cost: Decimal
    acquired_on: datetime


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


def standard(data: PortfolioData) -> ProblemSpec:
    """Align every input to the sorted universe and express it as a fraction of NAV.

    Derived per security: ``tax_per_dollar`` (signed; a loss is negative), ``tcost_per_dollar`` where
    the universe carries ``tcost_bps``, and ``adv_capacity`` — the style's participation times the
    day's volume, as a fraction of NAV — where it carries ``adv_shares``. The bounds fold the style's
    ``max_weight``, the optional ``min_weight``/``max_weight`` columns, and the ``restricted`` flag,
    which freezes a name at its current weight.
    """
    universe = data.universe.sort_values("security_id", kind="stable").reset_index(drop=True)
    ids = tuple(str(value) for value in universe["security_id"])
    unbuyable = sorted({str(value) for value in data.holdings["security_id"]} - set(ids))
    if unbuyable:
        msg = f"held securities missing from universe {unbuyable}; this build aligns every input to the universe, so add held names to it (restricted if they must not be bought)"
        raise BuildError(msg)
    n = len(ids)
    nav = data.details.nav
    price = [_decimal(value) for value in universe["price"]]
    positions = _positions(data.holdings)
    shares_held = [positions[security].quantity if security in positions else 0 for security in ids]
    lot_size = [int(value) for value in universe["lot_size"]] if "lot_size" in universe.columns else [1] * n
    w0 = [Decimal(shares) * px / nav for shares, px in zip(shares_held, price, strict=True)]
    lb, ub = _bounds(data, universe, ids, w0)
    columns: dict[str, F64] = {"tax_per_dollar": to_float64([_tax_per_dollar(data, positions.get(security), px) for security, px in zip(ids, price, strict=True)], "tax_per_dollar")}
    if "tcost_bps" in universe.columns:
        columns["tcost_per_dollar"] = to_float64([_decimal(value) / BPS for value in universe["tcost_bps"]], "tcost_per_dollar")
    if "adv_shares" in universe.columns:
        columns["adv_capacity"] = to_float64([data.details.max_adv_participation * Decimal(int(adv)) * px / nav for adv, px in zip(universe["adv_shares"], price, strict=True)], "adv_capacity")
    extra_columns, flags, groups = _exports(universe)
    return ProblemSpec(
        portfolio_id=data.details.portfolio_id,
        as_of_date=data.as_of_date,
        security_ids=ids,
        nav=float(nav),
        w0=to_float64(w0, "w0"),
        price=to_float64(price, "price"),
        shares_held=to_float64(shares_held, "shares_held"),
        lot_size=to_float64(lot_size, "lot_size"),
        lb=to_float64(lb, "lb"),
        ub=to_float64(ub, "ub"),
        columns={**extra_columns, **columns},
        flags=flags,
        groups=groups,
        scalars={name: float(value) for name, value in data.details.scalars().items()},
    )


def order_inputs(data: PortfolioData, spec: ProblemSpec) -> OrderInputs:
    """The exact inputs the order step needs, aligned to ``spec``: prices and shares from the bundle, the upper bounds from the spec.

    Works for any build whose spec is aligned to the universe's securities — a custom build included —
    so orders never reconstruct money from float64.
    """
    universe = data.universe.set_index(data.universe["security_id"].astype(str))
    missing = sorted(set(spec.security_ids) - set(universe.index))
    if missing:
        msg = f"the spec names securities the universe does not carry {missing}; a build must align its spec to the universe"
        raise BuildError(msg)
    aligned = universe.loc[list(spec.security_ids)]
    positions = _positions(data.holdings)
    nav = data.details.nav
    price = tuple(_decimal(value) for value in aligned["price"])
    shares = tuple(positions[security].quantity if security in positions else 0 for security in spec.security_ids)
    lots = tuple(int(value) for value in aligned["lot_size"]) if "lot_size" in aligned.columns else (1,) * spec.n
    w0 = tuple(Decimal(held) * px / nav for held, px in zip(shares, price, strict=True))
    return OrderInputs(
        security_ids=spec.security_ids,
        price=price,
        shares_held=shares,
        lot_size=lots,
        w0=w0,
        ub=tuple(Decimal(repr(float(bound))) for bound in spec.ub),
        nav=nav,
        min_trade_notional=data.details.min_trade_notional,
    )


def _membership(column: pd.Series, name: str) -> Grouping:
    """A string column as its *K*-by-*N* 0/1 membership matrix, built in numpy and carried sparse.

    One nonzero per column, so it is a megabyte at 100,000 names however many groups there are; the
    dense form is 8 *K* *N* bytes and was most of every large spec.
    """
    if column.isna().any():
        msg = f"grouping column {name!r} has null values; fill them in a rule before the optimizer runs"
        raise BuildError(msg)
    names = tuple(sorted({str(value) for value in column}))
    codes = np.asarray(pd.Categorical(column.astype("string"), categories=list(names)).codes, dtype=np.int64)
    n = len(codes)
    by_column = csc_array((np.ones(n, dtype=np.float64), codes, np.arange(n + 1, dtype=np.int64)), shape=(len(names), n))
    return Grouping(names, csr_array(by_column))


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


def _datetime(value: object) -> datetime:
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    msg = f"expected a timestamp, got {type(value).__name__}"
    raise BuildError(msg)


def _positions(holdings: pd.DataFrame) -> dict[str, _Position]:
    """Every held name by id, read from the frame once.

    The build asks for each universe security in turn, so a per-security scan of the holdings
    frame would cost universe times holdings per portfolio; this dict makes each lookup constant.
    """
    return {
        str(security): _Position(quantity=_int(quantity), avg_cost=_decimal(cost), acquired_on=_datetime(acquired))
        for security, quantity, cost, acquired in zip(holdings["security_id"], holdings["quantity"], holdings["avg_cost"], holdings["acquired_on"], strict=True)
    }


def _tax_per_dollar(data: PortfolioData, position: _Position | None, price: Decimal) -> Decimal:
    """Signed tax owed per dollar sold: gain fraction times the rate for the holding period. Losses are negative."""
    if position is None:
        return Decimal(0)
    rate = data.details.lt_tax_rate if data.as_of_date - position.acquired_on > LONG_TERM_HOLDING else data.details.st_tax_rate
    return (price - position.avg_cost) / price * rate


def _bounds(data: PortfolioData, universe: pd.DataFrame, ids: tuple[str, ...], w0: list[Decimal]) -> tuple[list[Decimal], list[Decimal]]:
    """Long-only floor and style cap, tightened by optional per-security columns; restricted names are frozen."""
    floors = [_optional_decimal(value) for value in universe["min_weight"]] if "min_weight" in universe.columns else [None] * len(ids)
    caps = [_optional_decimal(value) for value in universe["max_weight"]] if "max_weight" in universe.columns else [None] * len(ids)
    frozen = [bool(value) for value in universe["restricted"]] if "restricted" in universe.columns else [False] * len(ids)
    lb: list[Decimal] = []
    ub: list[Decimal] = []
    for index, security in enumerate(ids):
        if frozen[index]:
            lb.append(w0[index])
            ub.append(w0[index])
            continue
        floor = floors[index]
        cap = caps[index]
        low = Decimal(0) if floor is None else max(Decimal(0), floor)
        high = data.details.max_weight if cap is None else min(data.details.max_weight, cap)
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


def _exports(universe: pd.DataFrame) -> tuple[dict[str, F64], dict[str, Flags], dict[str, Grouping]]:
    """Every universe column the spec carries by name: numeric extras (and ``alpha``) as columns, booleans as flags, strings as groupings.

    The schema's own numeric columns are folded into the fixed vectors and derived columns and are
    not exported again. Holdings' extra columns are not exported: this build is aligned to the
    universe, and a per-position analytic has no value for names not held.
    """
    declared = {column.name for column in UNIVERSE.columns}
    columns: dict[str, F64] = {}
    flags: dict[str, Flags] = {}
    groups: dict[str, Grouping] = {}
    for name in universe.columns:
        column_name = str(name)
        if column_name == "security_id":
            continue
        column = universe[column_name]
        if pd.api.types.is_bool_dtype(column.dtype):
            flags[column_name] = _flag_values(column, column_name)
        elif pd.api.types.is_string_dtype(column.dtype) and column.dtype != "object":
            groups[column_name] = _membership(column, column_name)
        elif column_name in declared and column_name != "alpha":
            continue
        elif pd.api.types.is_numeric_dtype(column.dtype) or column.dtype == "object":
            values = _numeric_values(column, column_name)
            if values is not None:
                columns[column_name] = values
    return columns, flags, groups


def _flag_values(column: pd.Series, name: str) -> Flags:
    if column.isna().any():
        msg = f"flag column {name!r} has null values; fill them in a rule before the optimizer runs"
        raise BuildError(msg)
    return np.asarray(column.to_numpy(dtype="bool"), dtype=np.bool_)


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
