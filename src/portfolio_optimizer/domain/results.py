"""Pure-data results: the problem spec, solutions, verification reports, the chain between portfolios, and the audit records.

Everything here is picklable and free of cvxpy, so it can cross process boundaries and be
persisted for audit. The audit records are strict models because the manifest carries them as they
are: what a step did is recorded once, in one shape, from the worker to the file on disk.
"""

import hashlib
import json
import traceback
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
from scipy.sparse import csr_array

from portfolio_optimizer.domain.types import StrictModel

type F64 = NDArray[np.float64]
type Flags = NDArray[np.bool_]
type I64 = NDArray[np.int64]

RUN_SCOPED = "*"
"""The ``portfolio_id`` of a failure no portfolio owns: the cluster, the sink, a worker's own config resolution."""

TRACEBACK_LIMIT = 32_768
"""Characters of a formatted traceback kept; anything longer is elided in the middle, never at the ends."""

VECTOR_FIELDS: tuple[str, ...] = ("w0", "price", "shares_held", "lot_size", "lb", "ub")
"""The per-security vectors every spec carries; ``spec.column`` reads these by name beside the exported columns."""


class ProblemSpecError(ValueError):
    """The spec is not a well-formed optimization problem."""


class MissingSpecColumnError(KeyError):
    """A term or constraint asked for a column, flag, scalar, or grouping the spec does not carry."""

    def __init__(self, name: str, available: tuple[str, ...], kind: str = "column") -> None:
        self.name = name
        self.available = available
        super().__init__(f"spec has no {kind} {name!r}; available: {list(available)}")


def _readonly[T: np.generic](array: NDArray[T], dtype: type[T]) -> NDArray[T]:
    """A contiguous copy at ``dtype``, frozen: nothing a spec, solution, or chain state carries is mutated in place."""
    result: NDArray[T] = np.ascontiguousarray(array, dtype=dtype)
    if result is array:
        result = result.copy()
    result.flags.writeable = False
    return result


def _aligned_shares(security_ids: tuple[str, ...], traded_shares: F64) -> F64:
    """Freeze a per-security share vector and insist it lines up with the ids; what a chain state and a contribution both carry."""
    frozen = _readonly(traded_shares, np.float64)
    if frozen.shape != (len(security_ids),):
        msg = f"traded_shares has shape {frozen.shape}, expected {(len(security_ids),)}"
        raise ValueError(msg)
    return frozen


def _readonly_sparse(matrix: csr_array | F64) -> csr_array:
    """A canonical CSR copy — float64, duplicates summed, indices sorted — with its three arrays frozen."""
    result = csr_array(matrix, dtype=np.float64, copy=True)
    result.sum_duplicates()
    result.sort_indices()
    for array in (result.data, result.indices, result.indptr):
        array.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True, eq=False)
class Grouping:
    """One categorical universe column as a membership matrix: ``names`` are its distinct values, sorted, and ``matrix`` is *K*-by-*N* with one nonzero per security.

    Sparse, so a grouping is a megabyte at 100,000 names however many groups it has; the dense form
    was most of every large spec. A dense matrix is accepted on construction and converted.
    """

    names: tuple[str, ...]
    matrix: csr_array

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(str(name) for name in self.names))
        object.__setattr__(self, "matrix", _readonly_sparse(self.matrix))
        if self.matrix.shape[0] != len(self.names):
            msg = f"grouping has {len(self.names)} name(s) but a matrix of {self.matrix.shape[0]} row(s)"
            raise ValueError(msg)

    def row(self, name: str) -> csr_array:
        """The one row that ``name`` selects — which securities belong to that group — kept sparse."""
        try:
            index = self.names.index(name)
        except ValueError:
            raise MissingSpecColumnError(name, self.names, kind="group") from None
        return csr_array(self.matrix[[index], :])

    def parts(self, prefix: str) -> Iterator[tuple[str, F64 | I64]]:
        """The matrix as the three CSR arrays that define it, named under ``prefix``, in a fixed order."""
        yield f"{prefix}__data", np.asarray(self.matrix.data, dtype=np.float64)
        yield f"{prefix}__indices", np.asarray(self.matrix.indices, dtype=np.int64)
        yield f"{prefix}__indptr", np.asarray(self.matrix.indptr, dtype=np.int64)


@dataclass(frozen=True, slots=True, eq=False)
class ProblemSpec:
    """The optimization problem for one portfolio as pure numpy data.

    Every vector is aligned to ``security_ids`` and expressed as a fraction of NAV. The spec is
    independent of prior portfolios; chain-aware constraints combine it with a
    :class:`ChainState` at solve time. Six vectors are fixed — the starting weights, price, shares
    held, lot size, and the per-security bounds the build derived — and everything else is named:
    ``columns`` are per-security numbers (a derived ``tax_per_dollar`` beside an exported ``alpha``),
    ``flags`` are per-security booleans, ``groups`` are categorical columns as membership matrices,
    and ``scalars`` are per-account numbers (``cash_ub``, ``max_turnover``, and whatever else the
    account's row carried). A term or constraint reads any of them by name and refuses, by name,
    what the spec does not carry. :attr:`buyable` and :attr:`sellable` are the sets a side profile
    couples this portfolio through.
    """

    portfolio_id: str
    as_of_date: datetime
    security_ids: tuple[str, ...]
    nav: float
    w0: F64
    price: F64
    shares_held: F64
    lot_size: F64
    lb: F64
    ub: F64
    columns: Mapping[str, F64] = field(default_factory=dict)
    flags: Mapping[str, Flags] = field(default_factory=dict)
    groups: Mapping[str, Grouping] = field(default_factory=dict)
    scalars: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in VECTOR_FIELDS:
            object.__setattr__(self, name, _readonly(getattr(self, name), np.float64))
        object.__setattr__(self, "columns", {name: _readonly(array, np.float64) for name, array in sorted(self.columns.items())})
        object.__setattr__(self, "flags", {name: _readonly(array, np.bool_) for name, array in sorted(self.flags.items())})
        object.__setattr__(self, "groups", dict(sorted(self.groups.items())))
        object.__setattr__(self, "scalars", {name: float(value) for name, value in sorted(self.scalars.items())})
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
        for name, grouping in self.groups.items():
            if not np.isfinite(np.asarray(grouping.matrix.data, dtype=np.float64)).all():
                yield f"grouping {name!r} contains non-finite values"
        if not np.isfinite(self.nav):
            yield "nav is not finite"
        for name, value in self.scalars.items():
            if not np.isfinite(value):
                yield f"scalar {name!r} is not finite"
        if np.any(self.lb > self.ub):
            yield "lb > ub for some security"
        if self.nav <= 0.0:
            yield "nav must be positive"
        if np.any(self.price <= 0.0):
            yield "price must be positive"
        if np.any(self.lot_size < 1.0):
            yield "lot_size must be at least 1"

    def _structural_failures(self) -> Iterator[str]:
        n = len(self.security_ids)
        if len(set(self.security_ids)) != n:
            yield "security_ids are not unique"
        if list(self.security_ids) != sorted(self.security_ids):
            yield "security_ids are not sorted"
        for name in VECTOR_FIELDS:
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
        fixed = sorted(set(self.columns) & set(VECTOR_FIELDS))
        if fixed:
            yield f"columns {fixed} shadow the spec's own vectors"
        for name, grouping in self.groups.items():
            if grouping.matrix.shape[1] != n:
                yield f"grouping {name!r} has shape {grouping.matrix.shape}, expected {(len(grouping.names), n)}"

    def _arrays(self) -> Iterator[tuple[str, F64]]:
        for name in VECTOR_FIELDS:
            yield name, getattr(self, name)
        for name, array in self.columns.items():
            yield f"columns.{name}", array

    def _sparse_parts(self) -> Iterator[tuple[str, F64 | I64]]:
        """Every grouping's matrix as its three CSR arrays, named ``group__<column>__<part>``, in a fixed order."""
        for name, grouping in self.groups.items():
            yield from grouping.parts(f"group__{name}")

    @property
    def n(self) -> int:
        """Number of securities."""
        return len(self.security_ids)

    @property
    def buyable(self) -> Flags:
        """Securities a strictly positive net buy is allowed in: ``ub > w0``.

        The set a run that couples through buys builds its dependency graph and chain state from; a
        security frozen or capped at its current weight is outside it.
        """
        return self.ub > self.w0

    @property
    def sellable(self) -> Flags:
        """Securities a strictly positive net sell is allowed in: held, and ``lb < w0``.

        The mirror of :attr:`buyable` for a run that couples through sells; a security frozen or
        floored at its current weight, or not held at all, is outside it.
        """
        return (self.w0 > 0.0) & (self.lb < self.w0)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Every per-security vector a term or constraint may read by name: the fixed ones, then the exported columns."""
        return (*VECTOR_FIELDS, *self.columns)

    def column(self, name: str) -> F64:
        """A per-security numeric vector: one of the spec's own (``w0``, ``ub``, ...) or a column the build exported by name."""
        if name in VECTOR_FIELDS:
            vector: F64 = getattr(self, name)
            return vector
        try:
            return self.columns[name]
        except KeyError as error:
            raise MissingSpecColumnError(name, self.column_names) from error

    def flag(self, name: str) -> Flags:
        """A per-security boolean mask the build exported by name."""
        try:
            return self.flags[name]
        except KeyError as error:
            raise MissingSpecColumnError(name, tuple(self.flags), kind="flag") from error

    def scalar(self, name: str) -> float:
        """A per-account number: a style limit such as ``cash_ub``, or any numeric column the account's row carried."""
        try:
            return self.scalars[name]
        except KeyError as error:
            raise MissingSpecColumnError(name, tuple(self.scalars), kind="scalar") from error

    def group(self, column: str) -> Grouping:
        """A categorical column as a membership matrix; a constraint that bounds a group reads its members here and its numbers from its own row."""
        try:
            return self.groups[column]
        except KeyError as error:
            raise MissingSpecColumnError(column, tuple(self.groups), kind="grouping") from error

    def content_hash(self) -> str:
        """Deterministic sha256 of every input the solver will see."""
        digest = hashlib.sha256()
        digest.update(json.dumps(self._metadata(), sort_keys=True, separators=(",", ":")).encode())
        for name, array in self._arrays():
            digest.update(name.encode())
            digest.update(str(array.shape).encode())
            digest.update(array.dtype.str.encode())
            digest.update(np.ascontiguousarray(array + 0.0).tobytes())  # `+ 0.0` maps -0.0 to 0.0 so equal specs hash equal
        for name, part in self._sparse_parts():
            digest.update(name.encode())
            digest.update(str(part.shape).encode())
            digest.update(part.dtype.str.encode())
            digest.update(np.ascontiguousarray(part + 0.0 if part.dtype == np.float64 else part).tobytes())
        for name, array in self.flags.items():
            digest.update(f"flags.{name}".encode())
            digest.update(str(array.shape).encode())
            digest.update(array.dtype.str.encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def _metadata(self) -> dict[str, object]:
        return {
            "portfolio_id": self.portfolio_id,
            "as_of_date": self.as_of_date.isoformat(),
            "security_ids": list(self.security_ids),
            "nav": repr(float(self.nav)),
            "column_names": list(self.columns),
            "flag_names": list(self.flags),
            "group_names": {name: list(grouping.names) for name, grouping in self.groups.items()},
            "scalars": {name: repr(value) for name, value in self.scalars.items()},
        }

    def to_npz(self, path: Path) -> None:
        """Persist the spec as a single ``.npz`` file readable without pickle."""
        arrays: dict[str, F64 | I64 | Flags] = {name.replace("columns.", "col__"): np.ascontiguousarray(array) for name, array in self._arrays()}
        arrays.update({name: np.ascontiguousarray(part) for name, part in self._sparse_parts()})
        arrays.update({f"flag__{name}": np.ascontiguousarray(array) for name, array in self.flags.items()})
        np.savez(path, allow_pickle=False, __meta__=np.array(json.dumps(self._metadata(), sort_keys=True)), **arrays)

    @classmethod
    def from_npz(cls, path: Path) -> Self:
        """Load a spec written by :meth:`to_npz`."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["__meta__"]))
            loaded: dict[str, F64] = {key: np.asarray(data[key], dtype=np.float64) for key in data.files if key != "__meta__" and not key.startswith("flag__") and not key.startswith("group__")}
            flags: dict[str, Flags] = {key.removeprefix("flag__"): np.asarray(data[key], dtype=np.bool_) for key in data.files if key.startswith("flag__")}
            n = len(meta["security_ids"])
            groups = {
                name: Grouping(
                    tuple(str(group) for group in names),
                    csr_array(
                        (
                            np.asarray(data[f"group__{name}__data"], dtype=np.float64),
                            np.asarray(data[f"group__{name}__indices"], dtype=np.int64),
                            np.asarray(data[f"group__{name}__indptr"], dtype=np.int64),
                        ),
                        shape=(len(names), n),
                    ),
                )
                for name, names in meta["group_names"].items()
            }
        vectors = {name: loaded[name] for name in VECTOR_FIELDS}
        columns = {key.removeprefix("col__"): array for key, array in loaded.items() if key.startswith("col__")}
        return cls(
            portfolio_id=str(meta["portfolio_id"]),
            as_of_date=datetime.fromisoformat(str(meta["as_of_date"])),
            security_ids=tuple(str(s) for s in meta["security_ids"]),
            nav=float(meta["nav"]),
            columns=columns,
            flags=flags,
            groups=groups,
            scalars={str(name): float(value) for name, value in meta["scalars"].items()},
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


class Artifact(StrictModel):
    """A file the run wrote, with its hash for the manifest."""

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


type ConstraintRecord = dict[str, object]
"""A typed constraint as JSON-safe data — its ``kind`` and fields — the form a solution carries and the manifest records; the registry in ``domain/constraints.py`` parses it back."""


@dataclass(frozen=True, slots=True, eq=False)
class Solution:
    """The solver's answer for one spec, with enough provenance to reproduce it."""

    w: F64
    buy: F64
    sell: F64
    objective: float | None
    status: SolveStatus
    solver: str
    solver_version: str
    solve_time_s: float
    iterations: int | None
    spec_hash: str
    constraints: tuple[ConstraintRecord, ...] = ()
    """The typed constraints the solve step applied, as records.

    The step says what it made of this portfolio's constraint rows; the verifier parses each record
    back into its model and re-checks it through the model's own residual. Persisted with the rest of
    the provenance so an offline ``verify`` from the ``.npz`` alone checks exactly what the run did.
    """

    duals: Mapping[str, float] = field(default_factory=dict)
    """Per constraint name, the largest dual value the solver reported for it — its shadow price, zero where it did not bind; empty for a step that reports none."""

    def __post_init__(self) -> None:
        for name in ("w", "buy", "sell"):
            object.__setattr__(self, name, _readonly(getattr(self, name), np.float64))
        object.__setattr__(self, "constraints", tuple(dict(record) for record in self.constraints))
        object.__setattr__(self, "duals", {str(name): float(value) for name, value in sorted(self.duals.items())})

    def to_npz(self, path: Path) -> None:
        """Persist the solution vectors and provenance without pickle."""
        meta: dict[str, object] = {name: getattr(self, name) for name in ("objective", "status", "solver", "solver_version", "solve_time_s", "iterations", "spec_hash")}
        meta["constraints"] = list(self.constraints)
        meta["duals"] = dict(self.duals)
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
            objective=None if meta["objective"] is None else float(meta["objective"]),
            status=SolveStatus(str(meta["status"])),
            solver=str(meta["solver"]),
            solver_version=str(meta["solver_version"]),
            solve_time_s=float(meta["solve_time_s"]),
            iterations=None if meta["iterations"] is None else int(meta["iterations"]),
            spec_hash=str(meta["spec_hash"]),
            constraints=tuple({str(key): value for key, value in record.items()} for record in meta.get("constraints", ())),
            duals={str(name): float(value) for name, value in meta.get("duals", {}).items()},
        )


@dataclass(frozen=True, slots=True)
class Tolerances:
    """Verification tolerances; deliberately looser than the solver's so a pass is meaningful.

    ``violation`` bounds every residual, equality and inequality alike; the objective comparison
    passes within ``obj_abs + obj_rel · |recomputed|``.
    """

    violation: float = 1e-6
    obj_rel: float = 1e-5
    obj_abs: float = 1e-9


@dataclass(frozen=True, slots=True)
class ConstraintCheck:
    """Maximum violation of one residual, compared with the tolerance; ``label`` names the constraint it belongs to.

    ``active`` says the residual sits within the tolerance of its bound — the constraint is binding,
    or was breached — which is what answers "why did the solver stop here".
    """

    name: str
    violation: float
    tolerance: float
    passed: bool
    worst_security: str | None
    label: str
    active: bool = False

    @property
    def display(self) -> str:
        """The residual's name, qualified by the constraint's label where the two differ: two constraints of one kind produce residuals of the same name."""
        return self.name if self.label in (self.name, "identity", "solution") else f"{self.label}/{self.name}"


@dataclass(frozen=True, slots=True)
class ConstraintReport:
    """Independent re-verification of a solution against its spec."""

    checks: tuple[ConstraintCheck, ...]
    objective_terms: tuple[tuple[str, float], ...]
    recomputed_objective: float
    solver_objective: float | None
    objective_gap: float
    objective_passed: bool

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
        """The checks that failed, by display name."""
        return tuple(check.display for check in self.checks if not check.passed)

    @property
    def active(self) -> tuple[str, ...]:
        """The checks that bind, by display name, in report order: where the answer sits against a limit."""
        return tuple(check.display for check in self.checks if check.active)


class AssemblyAuditRecord(StrictModel):
    """What one assembly step did to the run's datasets: row counts per dataset before and after, and the columns it added."""

    qualname: str
    source_sha256: str
    params_sha256: str
    rows_in: dict[str, int]
    rows_out: dict[str, int]
    columns_added: dict[str, tuple[str, ...]]


class RuleAuditRecord(StrictModel):
    """What one rule did to one portfolio's bundle: row counts per frame before and after."""

    qualname: str
    source_sha256: str
    params_sha256: str
    rows_in: dict[str, int]
    rows_out: dict[str, int]


@dataclass(frozen=True, slots=True, eq=False)
class ChainState:
    """What higher-priority portfolios traded, on the side the run couples through, among the securities this one can trade there; aligned to one spec.

    ``traded_shares[i]`` is the whole shares predecessors traded of ``security_ids[i]`` on that side —
    bought in a run that couples through buys, sold in one that couples through sells — always
    positive, and zero wherever this portfolio cannot trade the security on that side: a run couples
    through one side only, so a security outside this portfolio's tradable set carries no chain
    state. That mask is what makes the state a function of the *overlapping* predecessors alone — the
    same array whether the run folded every earlier portfolio or only those sharing a tradable name.
    ``predecessors`` names what was folded, in solve order; it is provenance, not an input, so
    :meth:`content_hash` covers the ids and the shares only.
    """

    security_ids: tuple[str, ...]
    traded_shares: F64
    predecessors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "traded_shares", _aligned_shares(self.security_ids, self.traded_shares))

    def content_hash(self) -> str:
        """Deterministic sha256 of the chain inputs a solve depended on; independent of which predecessors produced them, and of what the shares are called."""
        digest = hashlib.sha256()
        digest.update(json.dumps({"security_ids": list(self.security_ids)}).encode())
        digest.update(np.ascontiguousarray(self.traded_shares + 0.0).tobytes())
        return digest.hexdigest()

    @classmethod
    def empty(cls, security_ids: tuple[str, ...]) -> Self:
        """The state of a portfolio with no predecessors."""
        return cls(security_ids=security_ids, traded_shares=np.zeros(len(security_ids)))

    def to_npz(self, path: Path) -> None:
        """Persist the chain inputs a solve depended on."""
        meta = {"security_ids": list(self.security_ids), "predecessors": list(self.predecessors)}
        np.savez(path, allow_pickle=False, __meta__=np.array(json.dumps(meta, sort_keys=True)), traded_shares=self.traded_shares)

    @classmethod
    def from_npz(cls, path: Path) -> Self:
        """Load a chain state written by :meth:`to_npz`."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["__meta__"]))
            shares = np.asarray(data["traded_shares"], dtype=np.float64)
        return cls(security_ids=tuple(str(s) for s in meta["security_ids"]), traded_shares=shares, predecessors=tuple(str(p) for p in meta["predecessors"]))


@dataclass(frozen=True, slots=True, eq=False)
class Contribution:
    """One solved portfolio's trades on the side the run couples through, as the slim object a dependent solve receives: whole shares traded, by security."""

    portfolio_id: str
    security_ids: tuple[str, ...]
    traded_shares: F64

    def __post_init__(self) -> None:
        object.__setattr__(self, "traded_shares", _aligned_shares(self.security_ids, self.traded_shares))

    @classmethod
    def from_orders(cls, portfolio_id: str, orders: pd.DataFrame, side: str) -> Self:
        """The rows of an orders frame on ``side`` (``BUY`` or ``SELL``); the other side never reaches a later portfolio."""
        rows = orders[orders["side"] == side]
        return cls(
            portfolio_id=portfolio_id,
            security_ids=tuple(str(security) for security in rows["security_id"]),
            traded_shares=np.array([float(int(quantity)) for quantity in rows["quantity"]], dtype=np.float64),
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
    contribution: Contribution


@dataclass(frozen=True, slots=True)
class PortfolioFailure:
    """A portfolio that did not produce orders, and where it failed.

    ``portfolio_id`` is :data:`RUN_SCOPED` for a failure no portfolio owns — the cluster never came up,
    the sink refused the orders, a worker could not resolve the config.

    ``traceback`` is the formatted traceback of the exception behind the failure, carried home from
    whichever process raised it. It is the whole reason a failure is debuggable at all once the run is
    over: a worker's own stderr goes to a pod that outlives nothing, so this is the only surviving
    record of *where* the failure happened. It is ``None`` for a failure no exception produced — a
    portfolio skipped after another's, a worker refused for its environment, an input simply absent.
    Observability, never identity: like the timing spans, neither it nor the file the run writes it to
    is compared by ``diff-manifests``.
    """

    portfolio_id: str
    stage: str
    error_type: str
    message: str
    traceback: str | None = None

    @classmethod
    def from_exception(cls, portfolio_id: str, stage: str, error: BaseException, *, message: str | None = None) -> Self:
        """Record ``error`` as the failure of ``portfolio_id`` at ``stage``, keeping its traceback.

        ``message`` overrides the exception's own text where the caller has a steadier one to record —
        a worker death names the task it blames rather than the worker address.
        """
        return cls(portfolio_id=portfolio_id, stage=stage, error_type=type(error).__name__, message=str(error) if message is None else message, traceback=format_traceback(error))


def format_traceback(error: BaseException) -> str:
    """The exception, its causes, and their frames as text, capped at :data:`TRACEBACK_LIMIT`.

    This string is carried from a worker to the client for every failed portfolio, so it needs a bound.
    Deep recursion is not what threatens one — Python collapses repeated frames itself — but a message
    that names every offending row does, and a book has a hundred thousand of them. The cap elides the
    middle and keeps both ends, where the origin and the raise site are.
    """
    text = "".join(traceback.format_exception(error))
    if len(text) <= TRACEBACK_LIMIT:
        return text
    half = TRACEBACK_LIMIT // 2
    return f"{text[:half]}\n... {len(text) - TRACEBACK_LIMIT} character(s) elided ...\n{text[-half:]}"


def derive_chain_state(security_ids: tuple[str, ...], tradable: Flags, contributions: Sequence[Contribution]) -> ChainState:
    """Fold predecessors' trades onto ``security_ids`` and zero every security outside ``tradable``, this portfolio's set on the side the run couples through.

    ``contributions`` must already be in solve order; a security no predecessor traded is zero.
    """
    if tradable.shape != (len(security_ids),):
        msg = f"tradable has shape {tradable.shape}, expected {(len(security_ids),)}"
        raise ValueError(msg)
    totals: dict[str, float] = {}
    for contribution in contributions:
        for security, shares in zip(contribution.security_ids, contribution.traded_shares, strict=True):
            totals[security] = totals.get(security, 0.0) + float(shares)
    projected = np.array([totals.get(security, 0.0) for security in security_ids], dtype=np.float64)
    return ChainState(security_ids=security_ids, traded_shares=np.where(tradable, projected, 0.0), predecessors=tuple(contribution.portfolio_id for contribution in contributions))
