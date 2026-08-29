"""Pure-data results: the problem spec, solutions, verification reports, and the chain between portfolios.

Everything here is picklable and free of cvxpy, so it can cross process boundaries and be
persisted for audit.
"""

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

type F64 = NDArray[np.float64]
type Flags = NDArray[np.bool_]

_SCALAR_FIELDS: tuple[str, ...] = ("nav", "max_turnover", "cash_lb", "cash_ub", "min_trade_notional")
_VECTOR_FIELDS: tuple[str, ...] = ("w0", "price", "shares_held", "lot_size", "w_target", "tax_per_dollar", "tcost_per_dollar", "lb", "ub", "adv_capacity")


class ProblemSpecError(ValueError):
    """The spec is not a well-formed optimization problem."""


class MissingSpecColumnError(KeyError):
    """A term asked for a per-security column or flag the spec does not carry."""

    def __init__(self, name: str, available: tuple[str, ...], kind: str = "column") -> None:
        self.name = name
        self.available = available
        super().__init__(f"spec has no {kind} {name!r}; available: {list(available)}")


def _readonly(array: F64) -> F64:
    result = np.ascontiguousarray(array, dtype=np.float64)
    if result is array:
        result = result.copy()
    result.flags.writeable = False
    return result


def _readonly_flags(array: Flags) -> Flags:
    result = np.ascontiguousarray(array, dtype=np.bool_)
    if result is array:
        result = result.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True, eq=False)
class ProblemSpec:
    """The optimization problem for one portfolio as pure numpy data.

    Every vector is aligned to ``security_ids`` and expressed as a fraction of NAV. The spec is
    independent of prior portfolios; chain-aware constraints combine it with a
    :class:`ChainState` at solve time. ``columns`` are the numeric per-security columns the build
    exported from the universe, ``flags`` the boolean ones; the two namespaces do not overlap.
    :attr:`buyable` is the set the chain couples this portfolio through.
    """

    portfolio_id: str
    as_of: datetime
    security_ids: tuple[str, ...]
    sector_names: tuple[str, ...]
    nav: float
    w0: F64
    price: F64
    shares_held: F64
    lot_size: F64
    w_target: F64
    tax_per_dollar: F64
    tcost_per_dollar: F64
    lb: F64
    ub: F64
    adv_capacity: F64
    sector_matrix: F64
    sector_lb: F64
    sector_ub: F64
    max_turnover: float
    cash_lb: float
    cash_ub: float
    min_trade_notional: float
    columns: Mapping[str, F64] = field(default_factory=dict)
    flags: Mapping[str, Flags] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in _VECTOR_FIELDS:
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, "sector_matrix", _readonly(self.sector_matrix))
        object.__setattr__(self, "sector_lb", _readonly(self.sector_lb))
        object.__setattr__(self, "sector_ub", _readonly(self.sector_ub))
        object.__setattr__(self, "columns", {name: _readonly(array) for name, array in sorted(self.columns.items())})
        object.__setattr__(self, "flags", {name: _readonly_flags(array) for name, array in sorted(self.flags.items())})
        failures = list(self._failures())
        if failures:
            raise ProblemSpecError(f"portfolio {self.portfolio_id!r}: " + "; ".join(failures))

    def _failures(self) -> Iterator[str]:
        structural = list(self._structural_failures())
        yield from structural
        if structural:
            return  # value checks below assume the declared shapes
        for name, array in self._arrays():
            if not np.isfinite(array).all():
                yield f"{name} contains non-finite values"
        for name in _SCALAR_FIELDS:
            if not np.isfinite(getattr(self, name)):
                yield f"{name} is not finite"
        if np.any(self.lb > self.ub):
            yield "lb > ub for some security"
        if np.any(self.sector_lb > self.sector_ub):
            yield "sector_lb > sector_ub for some sector"
        if self.cash_lb > self.cash_ub:
            yield "cash_lb > cash_ub"
        if self.nav <= 0.0:
            yield "nav must be positive"
        if np.any(self.price <= 0.0):
            yield "price must be positive"
        if np.any(self.lot_size < 1.0):
            yield "lot_size must be at least 1"

    def _structural_failures(self) -> Iterator[str]:
        n = len(self.security_ids)
        k = len(self.sector_names)
        if len(set(self.security_ids)) != n:
            yield "security_ids are not unique"
        if list(self.security_ids) != sorted(self.security_ids):
            yield "security_ids are not sorted"
        for name in _VECTOR_FIELDS:
            array = getattr(self, name)
            if array.shape != (n,):
                yield f"{name} has shape {array.shape}, expected {(n,)}"
        for name, array in self.columns.items():
            if array.shape != (n,):
                yield f"column {name!r} has shape {array.shape}, expected {(n,)}"
        for name, array in self.flags.items():
            if array.shape != (n,):
                yield f"flag {name!r} has shape {array.shape}, expected {(n,)}"
        shared = sorted(set(self.columns) & set(self.flags))
        if shared:
            yield f"names {shared} are both a column and a flag"
        if self.sector_matrix.shape != (k, n):
            yield f"sector_matrix has shape {self.sector_matrix.shape}, expected {(k, n)}"
        if self.sector_lb.shape != (k,) or self.sector_ub.shape != (k,):
            yield f"sector bounds have shapes {self.sector_lb.shape}, {self.sector_ub.shape}, expected {(k,)}"

    def _arrays(self) -> Iterator[tuple[str, F64]]:
        for name in _VECTOR_FIELDS:
            yield name, getattr(self, name)
        yield "sector_matrix", self.sector_matrix
        yield "sector_lb", self.sector_lb
        yield "sector_ub", self.sector_ub
        for name, array in self.columns.items():
            yield f"columns.{name}", array

    @property
    def n(self) -> int:
        """Number of securities."""
        return len(self.security_ids)

    @property
    def buyable(self) -> Flags:
        """Securities a strictly positive net buy is allowed in: ``ub > w0``.

        Portfolios couple across a run through buys only, so this is the set the dependency graph and
        the chain state are built from; a security frozen or capped at its current weight is outside it.
        """
        return self.ub > self.w0

    def column(self, name: str) -> F64:
        """Return a numeric per-security column exported from the universe frame."""
        try:
            return self.columns[name]
        except KeyError as error:
            raise MissingSpecColumnError(name, tuple(self.columns)) from error

    def flag(self, name: str) -> Flags:
        """Return a boolean per-security column exported from the universe frame, as a real boolean mask."""
        try:
            return self.flags[name]
        except KeyError as error:
            raise MissingSpecColumnError(name, tuple(self.flags), kind="flag") from error

    def content_hash(self) -> str:
        """Deterministic sha256 of every input the solver will see."""
        digest = hashlib.sha256()
        digest.update(json.dumps(self._metadata(), sort_keys=True, separators=(",", ":")).encode())
        for name, array in self._arrays():
            digest.update(name.encode())
            digest.update(str(array.shape).encode())
            digest.update(array.dtype.str.encode())
            digest.update(np.ascontiguousarray(array + 0.0).tobytes())  # `+ 0.0` maps -0.0 to 0.0 so equal specs hash equal
        for name, array in self.flags.items():
            digest.update(f"flags.{name}".encode())
            digest.update(str(array.shape).encode())
            digest.update(array.dtype.str.encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def _metadata(self) -> dict[str, object]:
        return {
            "portfolio_id": self.portfolio_id,
            "as_of": self.as_of.isoformat(),
            "security_ids": list(self.security_ids),
            "sector_names": list(self.sector_names),
            "column_names": list(self.columns),
            "flag_names": list(self.flags),
            **{name: repr(float(getattr(self, name))) for name in _SCALAR_FIELDS},
        }

    def to_npz(self, path: Path) -> None:
        """Persist the spec as a single ``.npz`` file readable without pickle."""
        arrays: dict[str, F64 | Flags] = {name.replace("columns.", "col__"): np.ascontiguousarray(array) for name, array in self._arrays()}
        arrays.update({f"flag__{name}": np.ascontiguousarray(array) for name, array in self.flags.items()})
        np.savez(path, allow_pickle=False, __meta__=np.array(json.dumps(self._metadata(), sort_keys=True)), **arrays)

    @classmethod
    def from_npz(cls, path: Path) -> Self:
        """Load a spec written by :meth:`to_npz`."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["__meta__"]))
            loaded: dict[str, F64] = {key: np.asarray(data[key], dtype=np.float64) for key in data.files if key != "__meta__" and not key.startswith("flag__")}
            flags: dict[str, Flags] = {key.removeprefix("flag__"): np.asarray(data[key], dtype=np.bool_) for key in data.files if key.startswith("flag__")}
        vectors = {name: loaded[name] for name in _VECTOR_FIELDS}
        columns = {key.removeprefix("col__"): array for key, array in loaded.items() if key.startswith("col__")}
        return cls(
            portfolio_id=str(meta["portfolio_id"]),
            as_of=datetime.fromisoformat(str(meta["as_of"])),
            security_ids=tuple(str(s) for s in meta["security_ids"]),
            sector_names=tuple(str(s) for s in meta["sector_names"]),
            nav=float(meta["nav"]),
            max_turnover=float(meta["max_turnover"]),
            cash_lb=float(meta["cash_lb"]),
            cash_ub=float(meta["cash_ub"]),
            min_trade_notional=float(meta["min_trade_notional"]),
            sector_matrix=loaded["sector_matrix"],
            sector_lb=loaded["sector_lb"],
            sector_ub=loaded["sector_ub"],
            columns=columns,
            flags=flags,
            **vectors,
        )


@dataclass(frozen=True, slots=True)
class OrderInputs:
    """Exact (Decimal/int) per-security inputs the order step needs, aligned like the spec.

    Built alongside the spec so orders never reconstruct money from float64.
    """

    security_ids: tuple[str, ...]
    price: tuple[Decimal, ...]
    shares_held: tuple[int, ...]
    lot_size: tuple[int, ...]
    w0: tuple[Decimal, ...]
    ub: tuple[Decimal, ...]
    nav: Decimal
    min_trade_notional: Decimal

    def __post_init__(self) -> None:
        n = len(self.security_ids)
        if not (len(self.price) == len(self.shares_held) == len(self.lot_size) == len(self.w0) == len(self.ub) == n):
            msg = f"order inputs are not aligned to {n} securities"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DriftReport:
    """How far integer rounding moved the executed weights from the solved ones."""

    max_weight_error: float
    tolerance: float
    dropped_orders: int

    @property
    def passed(self) -> bool:
        """True when rounding stayed within the bound implied by lot sizes and the dust filter."""
        return self.max_weight_error <= self.tolerance


@dataclass(frozen=True, slots=True)
class StepRef:
    """A step's identity and parameters as data, for cvxpy-free verification and the manifest."""

    qualname: str
    params: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Artifact:
    """A file a sink wrote, with its hash for the manifest."""

    path: str
    sha256: str
    size_bytes: int


class SolveStatus(StrEnum):
    """Normalized solver outcome."""

    OPTIMAL = "optimal"
    OPTIMAL_INACCURATE = "optimal_inaccurate"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    SOLVER_ERROR = "solver_error"


@dataclass(frozen=True, slots=True, eq=False)
class Solution:
    """The solver's answer for one spec, with enough provenance to reproduce it."""

    w: F64
    buy: F64
    sell: F64
    objective: float
    status: SolveStatus
    solver: str
    solver_version: str
    cvxpy_version: str
    solve_time_s: float
    iterations: int | None
    spec_hash: str

    def __post_init__(self) -> None:
        for name in ("w", "buy", "sell"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    def to_npz(self, path: Path) -> None:
        """Persist the solution vectors and provenance without pickle."""
        meta = {name: getattr(self, name) for name in ("objective", "status", "solver", "solver_version", "cvxpy_version", "solve_time_s", "iterations", "spec_hash")}
        np.savez(path, allow_pickle=False, __meta__=np.array(json.dumps(meta, sort_keys=True)), w=self.w, buy=self.buy, sell=self.sell)

    @classmethod
    def from_npz(cls, path: Path) -> Self:
        """Load a solution written by :meth:`to_npz`."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["__meta__"]))
            w, buy, sell = (np.asarray(data[key], dtype=np.float64) for key in ("w", "buy", "sell"))
        return cls(
            w=w,
            buy=buy,
            sell=sell,
            objective=float(meta["objective"]),
            status=SolveStatus(str(meta["status"])),
            solver=str(meta["solver"]),
            solver_version=str(meta["solver_version"]),
            cvxpy_version=str(meta["cvxpy_version"]),
            solve_time_s=float(meta["solve_time_s"]),
            iterations=None if meta["iterations"] is None else int(meta["iterations"]),
            spec_hash=str(meta["spec_hash"]),
        )


@dataclass(frozen=True, slots=True)
class Tolerances:
    """Verification tolerances; deliberately looser than the solver's so a pass is meaningful."""

    eq: float = 1e-6
    ineq: float = 1e-6
    obj_rel: float = 1e-5
    obj_abs: float = 1e-9


@dataclass(frozen=True, slots=True)
class ConstraintCheck:
    """Maximum violation of one constraint group, compared with its tolerance."""

    name: str
    violation: float
    tolerance: float
    passed: bool
    worst_security: str | None


@dataclass(frozen=True, slots=True)
class ConstraintReport:
    """Independent re-verification of a solution against its spec."""

    checks: tuple[ConstraintCheck, ...]
    objective_terms: tuple[tuple[str, float], ...]
    recomputed_objective: float
    solver_objective: float
    objective_gap: float
    objective_passed: bool
    unverified: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """True when every check and the objective comparison passed."""
        return self.objective_passed and all(check.passed for check in self.checks)

    @property
    def max_violation(self) -> float:
        """The largest violation across all checks."""
        return max((check.violation for check in self.checks), default=0.0)

    @property
    def violated(self) -> tuple[str, ...]:
        """Names of the checks that failed."""
        return tuple(check.name for check in self.checks if not check.passed)


@dataclass(frozen=True, slots=True)
class AssemblyAuditRecord:
    """What one assembly step did to the run's datasets."""

    qualname: str
    source_sha256: str
    params_sha256: str
    rows_in: Mapping[str, int]
    rows_out: Mapping[str, int]
    columns_added: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class RuleAuditRecord:
    """What one rule did to one portfolio's bundle."""

    qualname: str
    source_sha256: str
    params_sha256: str
    rows_in: Mapping[str, int]
    rows_out: Mapping[str, int]


@dataclass(frozen=True, slots=True, eq=False)
class ChainState:
    """What higher-priority portfolios bought among the securities this one may buy, aligned to one spec.

    ``bought_shares[i]`` is the whole shares predecessors bought of ``security_ids[i]``, and zero
    wherever this portfolio cannot buy (``ub == w0``): portfolios couple through buys only, so a
    security this portfolio cannot buy carries no chain state. That mask is what makes the state a
    function of the *overlapping* predecessors alone — the same array whether the run folded every
    earlier portfolio or only those sharing a buyable name. ``predecessors`` names what was folded,
    in solve order; it is provenance, not an input, so :meth:`content_hash` covers the ids and the
    shares only.
    """

    security_ids: tuple[str, ...]
    bought_shares: F64
    predecessors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bought_shares", _readonly(self.bought_shares))
        if self.bought_shares.shape != (len(self.security_ids),):
            msg = f"bought_shares has shape {self.bought_shares.shape}, expected {(len(self.security_ids),)}"
            raise ValueError(msg)

    def content_hash(self) -> str:
        """Deterministic sha256 of the chain inputs a solve depended on; independent of which predecessors produced them."""
        digest = hashlib.sha256()
        digest.update(json.dumps({"security_ids": list(self.security_ids)}).encode())
        digest.update(np.ascontiguousarray(self.bought_shares + 0.0).tobytes())
        return digest.hexdigest()

    @classmethod
    def empty(cls, security_ids: tuple[str, ...]) -> Self:
        """The state of a portfolio with no predecessors."""
        return cls(security_ids=security_ids, bought_shares=np.zeros(len(security_ids)))

    def to_npz(self, path: Path) -> None:
        """Persist the chain inputs a solve depended on."""
        meta = {"security_ids": list(self.security_ids), "predecessors": list(self.predecessors)}
        np.savez(path, allow_pickle=False, __meta__=np.array(json.dumps(meta, sort_keys=True)), bought_shares=self.bought_shares)

    @classmethod
    def from_npz(cls, path: Path) -> Self:
        """Load a chain state written by :meth:`to_npz`."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["__meta__"]))
            shares = np.asarray(data["bought_shares"], dtype=np.float64)
        return cls(security_ids=tuple(str(s) for s in meta["security_ids"]), bought_shares=shares, predecessors=tuple(str(p) for p in meta["predecessors"]))


@dataclass(frozen=True, slots=True, eq=False)
class Contribution:
    """One solved portfolio's buys, as the slim object a dependent solve receives: whole shares bought, by security."""

    portfolio_id: str
    security_ids: tuple[str, ...]
    bought_shares: F64

    def __post_init__(self) -> None:
        object.__setattr__(self, "bought_shares", _readonly(self.bought_shares))
        if self.bought_shares.shape != (len(self.security_ids),):
            msg = f"bought_shares has shape {self.bought_shares.shape}, expected {(len(self.security_ids),)}"
            raise ValueError(msg)

    @classmethod
    def from_orders(cls, portfolio_id: str, orders: pd.DataFrame) -> Self:
        """The BUY rows of an orders frame; sells never reach a later portfolio."""
        buys = orders[orders["side"] == "BUY"]
        return cls(
            portfolio_id=portfolio_id,
            security_ids=tuple(str(security) for security in buys["security_id"]),
            bought_shares=np.array([float(int(quantity)) for quantity in buys["quantity"]], dtype=np.float64),
        )


@dataclass(frozen=True, slots=True, eq=False)
class PortfolioResult:
    """A fully processed portfolio: spec, solution, verification, orders, and audit trail."""

    portfolio_id: str
    spec: ProblemSpec
    solution: Solution
    report: ConstraintReport
    orders: pd.DataFrame
    rule_audit: tuple[RuleAuditRecord, ...]
    chain_state: ChainState
    drift: DriftReport


@dataclass(frozen=True, slots=True)
class PortfolioFailure:
    """A portfolio that did not produce orders, and where it failed."""

    portfolio_id: str
    stage: str
    error_type: str
    message: str


def derive_chain_state(security_ids: tuple[str, ...], buyable: Flags, contributions: Sequence[Contribution]) -> ChainState:
    """Fold predecessors' buys onto ``security_ids`` and zero every security this portfolio cannot buy.

    ``contributions`` must already be in solve order; a security no predecessor bought is zero.
    """
    if buyable.shape != (len(security_ids),):
        msg = f"buyable has shape {buyable.shape}, expected {(len(security_ids),)}"
        raise ValueError(msg)
    totals: dict[str, float] = {}
    for contribution in contributions:
        for security, shares in zip(contribution.security_ids, contribution.bought_shares, strict=True):
            totals[security] = totals.get(security, 0.0) + float(shares)
    projected = np.array([totals.get(security, 0.0) for security in security_ids], dtype=np.float64)
    return ChainState(security_ids=security_ids, bought_shares=np.where(buyable, projected, 0.0), predecessors=tuple(contribution.portfolio_id for contribution in contributions))
