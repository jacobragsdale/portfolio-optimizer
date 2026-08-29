"""In-house DataFrame schema validation.

A :class:`FrameSchema` declares the exact column set, pandas dtype, nullability, value bounds,
unique key, and frame-level invariants of a boundary DataFrame. :func:`validate_frame` checks a
frame against it and raises :class:`FrameSchemaError` listing every failure at once; it never
coerces. :func:`coerce_frame` is the loader-side helper that casts raw input to the declared
dtypes (the one place string or float input becomes ``Decimal``).
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

import numpy as np
import pandas as pd

ColumnKind = Literal["string", "Int64", "bool", "Float64", "decimal", "datetime_utc"]

_PANDAS_DTYPE: Mapping[ColumnKind, str] = {"string": "string", "Int64": "Int64", "bool": "bool", "Float64": "Float64", "decimal": "object", "datetime_utc": "datetime64[ns, UTC]"}
_BOUNDED_KINDS: frozenset[ColumnKind] = frozenset({"Int64", "Float64", "decimal"})


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """One column: its pandas dtype, nullability, and value domain.

    ``decimal`` columns are ``object`` dtype whose every non-null element is a finite
    :class:`decimal.Decimal`; they carry money, prices, quantities-as-fractions, and rates.
    ``Float64`` is reserved for statistical estimates (alpha, covariance).
    """

    name: str
    kind: ColumnKind
    nullable: bool = False
    required: bool = True
    ge: Decimal | None = None
    gt: Decimal | None = None
    le: Decimal | None = None
    lt: Decimal | None = None
    allowed: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.kind == "bool" and self.nullable:
            msg = f"column {self.name!r}: bool columns cannot be nullable; use string with allowed values instead"
            raise ValueError(msg)
        has_bounds = any(bound is not None for bound in (self.ge, self.gt, self.le, self.lt))
        if has_bounds and self.kind not in _BOUNDED_KINDS:
            msg = f"column {self.name!r}: bounds apply only to Int64, Float64, and decimal columns"
            raise ValueError(msg)
        if self.allowed is not None and self.kind != "string":
            msg = f"column {self.name!r}: allowed values apply only to string columns"
            raise ValueError(msg)

    @property
    def dtype(self) -> str:
        """The pandas dtype string this column must have."""
        return _PANDAS_DTYPE[self.kind]


FrameCheckFn = Callable[[pd.DataFrame], str | None]


@dataclass(frozen=True, slots=True)
class FrameCheck:
    """A frame-level invariant. ``check`` returns a failure message, or ``None`` when satisfied."""

    name: str
    check: FrameCheckFn


@dataclass(frozen=True, slots=True)
class FrameSchema:
    """The contract for one boundary DataFrame."""

    name: str
    columns: tuple[ColumnSpec, ...]
    key: tuple[str, ...]
    checks: tuple[FrameCheck, ...] = ()
    allow_extra: bool = False

    def __post_init__(self) -> None:
        names = [column.name for column in self.columns]
        if len(set(names)) != len(names):
            msg = f"schema {self.name!r}: duplicate column names"
            raise ValueError(msg)
        by_name = {column.name: column for column in self.columns}
        for key_column in self.key:
            if key_column not in by_name:
                msg = f"schema {self.name!r}: key column {key_column!r} is not declared"
                raise ValueError(msg)
            if not by_name[key_column].required or by_name[key_column].nullable:
                msg = f"schema {self.name!r}: key column {key_column!r} must be required and non-nullable"
                raise ValueError(msg)

    @property
    def dtypes(self) -> dict[str, str]:
        """Column name to pandas dtype for every declared column (required and optional)."""
        return {column.name: column.dtype for column in self.columns}

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Names of the columns that must be present."""
        return tuple(column.name for column in self.columns if column.required)

    def column(self, name: str) -> ColumnSpec:
        """Return the spec for ``name``; raise ``KeyError`` when the schema does not declare it."""
        for column in self.columns:
            if column.name == name:
                return column
        msg = f"schema {self.name!r} has no column {name!r}"
        raise KeyError(msg)


class FrameSchemaError(ValueError):
    """A frame violated its schema. ``failures`` lists every violation found."""

    def __init__(self, schema_name: str, failures: Sequence[str]) -> None:
        self.schema_name = schema_name
        self.failures = tuple(failures)
        super().__init__(f"{schema_name}: {len(self.failures)} schema failure(s): " + "; ".join(self.failures))


def validate_frame(frame: pd.DataFrame, schema: FrameSchema) -> pd.DataFrame:
    """Check ``frame`` against ``schema`` and return it unchanged.

    Raises :class:`FrameSchemaError` listing every failure. Frame-level checks run only when
    every column check passed, because they may assume the declared shape.
    """
    failures: list[str] = []
    present = {str(name) for name in frame.columns}
    declared = {column.name for column in schema.columns}
    missing = [name for name in schema.required_columns if name not in present]
    extra = sorted(present - declared)
    if missing:
        failures.append(f"missing columns {missing}")
    if extra and not schema.allow_extra:
        failures.append(f"unexpected columns {extra}")
    for spec in schema.columns:
        if spec.name in present:
            failures.extend(_column_failures(frame[spec.name], spec))
    if not failures and schema.key and bool(frame.duplicated(subset=list(schema.key)).any()):
        duplicates = int(frame.duplicated(subset=list(schema.key)).sum())
        failures.append(f"key {schema.key} has {duplicates} duplicate row(s)")
    if not failures:
        for check in schema.checks:
            message = check.check(frame)
            if message is not None:
                failures.append(f"{check.name}: {message}")
    if failures:
        raise FrameSchemaError(schema.name, failures)
    return frame


def coerce_frame(frame: pd.DataFrame, schema: FrameSchema) -> pd.DataFrame:
    """Cast raw input columns to the schema's dtypes; the loader-side counterpart of validation.

    ``decimal`` columns are built from ``str``, ``int``, or ``Decimal`` values exactly and from
    ``float`` values via ``Decimal(repr(value))`` — the shortest round-tripping representation,
    which is the only defensible reading of a float that arrived from a file. Columns the schema
    does not declare are left untouched. The result still has to pass :func:`validate_frame`.
    """
    result = frame.copy()
    present = [spec for spec in schema.columns if spec.name in result.columns]
    for spec in present:
        if spec.kind == "decimal":
            result[spec.name] = result[spec.name].map(_to_decimal).astype("object")
    return result.astype({spec.name: spec.dtype for spec in present if spec.kind != "decimal"})


def _to_decimal(value: object) -> Decimal | None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        msg = f"cannot coerce bool {value!r} to Decimal"
        raise TypeError(msg)
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(repr(value))
    if isinstance(value, str):
        try:
            return Decimal(value.strip())
        except InvalidOperation as error:
            msg = f"cannot coerce {value!r} to Decimal"
            raise ValueError(msg) from error
    msg = f"cannot coerce {type(value).__name__} to Decimal"
    raise TypeError(msg)


def _column_failures(series: pd.Series, spec: ColumnSpec) -> list[str]:
    failures: list[str] = []
    actual = str(series.dtype)
    if actual != spec.dtype:
        failures.append(f"column {spec.name!r}: dtype {actual!r}, expected {spec.dtype!r}")
        return failures
    null_count = int(series.isna().sum())
    if null_count and not spec.nullable:
        failures.append(f"column {spec.name!r}: {null_count} null value(s)")
    non_null = series.dropna()
    if spec.kind == "decimal":
        bad = [value for value in non_null if not (isinstance(value, Decimal) and value.is_finite())]
        if bad:
            failures.append(f"column {spec.name!r}: {len(bad)} value(s) are not finite Decimals, e.g. {bad[0]!r}")
            return failures
    if spec.kind == "Float64" and not bool(np.isfinite(non_null.to_numpy(dtype="float64")).all()):
        failures.append(f"column {spec.name!r}: non-finite value(s)")
        return failures
    if spec.allowed is not None:
        disallowed = sorted({str(value) for value in non_null if str(value) not in spec.allowed})
        if disallowed:
            failures.append(f"column {spec.name!r}: value(s) {disallowed} not in {sorted(spec.allowed)}")
    failures.extend(_bound_failures(non_null, spec))
    return failures


def _bound_failures(non_null: pd.Series, spec: ColumnSpec) -> list[str]:
    if len(non_null) == 0 or spec.kind not in _BOUNDED_KINDS:
        return []
    values: list[Decimal] = list(non_null) if spec.kind == "decimal" else [Decimal(repr(float(value))) for value in non_null]
    lowest = min(values)
    highest = max(values)
    failures: list[str] = []
    if spec.ge is not None and lowest < spec.ge:
        failures.append(f"column {spec.name!r}: value {lowest} < {spec.ge}")
    if spec.gt is not None and lowest <= spec.gt:
        failures.append(f"column {spec.name!r}: value {lowest} <= {spec.gt}")
    if spec.le is not None and highest > spec.le:
        failures.append(f"column {spec.name!r}: value {highest} > {spec.le}")
    if spec.lt is not None and highest >= spec.lt:
        failures.append(f"column {spec.name!r}: value {highest} >= {spec.lt}")
    return failures
