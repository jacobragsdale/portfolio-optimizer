"""Dataset loaders — yours to edit.

A loader is an ordinary function ``(request: LoadRequest, params: P) -> pd.DataFrame`` named in
the run config. The ``constraints`` dataset's loader returns ``dict[str, dict[str, object]]``
keyed by portfolio id instead. Loaders are the only place file, database, or network access
belongs; everything downstream is pure.

The shipped loaders read files under ``request.data_root``. For an engine-known dataset they
cast columns to that dataset's schema; for any other dataset the params say which columns are
money (``decimal_columns``) or timestamps (``utc_datetime_columns``).
"""

import json
from collections.abc import Mapping

import pandas as pd
from pydantic import Field

from portfolio_optimizer.domain.data import LoadRequest
from portfolio_optimizer.domain.frames import ColumnSpec, FrameSchema, coerce_frame
from portfolio_optimizer.domain.schemas import DATASET_SCHEMAS, PORTFOLIOS
from portfolio_optimizer.domain.types import Params

LOADER_SCHEMAS: Mapping[str, FrameSchema] = {**DATASET_SCHEMAS, "portfolios": PORTFOLIOS}


class CsvParams(Params):
    """Parameters for :func:`csv`."""

    path: str = Field(min_length=1)
    decimal_columns: tuple[str, ...] = ()
    utc_datetime_columns: tuple[str, ...] = ()
    dtypes: dict[str, str] = Field(default_factory=dict)


def csv(request: LoadRequest, params: CsvParams) -> pd.DataFrame:
    """Read a CSV file with every dtype declared up front."""
    schema = LOADER_SCHEMAS.get(request.dataset)
    if schema is not None:
        raw = pd.read_csv(request.data_root / params.path, dtype=_read_dtypes(schema))
        return coerce_frame(_parse_utc(raw, [c.name for c in schema.columns if c.kind == "datetime_utc" and c.name in raw.columns]), schema)
    read_dtypes: dict[str, str] = {**params.dtypes, **dict.fromkeys(params.decimal_columns, "string"), **dict.fromkeys(params.utc_datetime_columns, "string")}
    raw = pd.read_csv(request.data_root / params.path, dtype=read_dtypes)
    decimals = FrameSchema("extra", tuple(_decimal_spec(name) for name in params.decimal_columns), ())
    return coerce_frame(_parse_utc(raw, list(params.utc_datetime_columns)), decimals)


class ParquetParams(Params):
    """Parameters for :func:`parquet`."""

    path: str = Field(min_length=1)
    decimal_columns: tuple[str, ...] = ()


def parquet(request: LoadRequest, params: ParquetParams) -> pd.DataFrame:
    """Read a Parquet file; Arrow decimal columns arrive as ``Decimal`` already."""
    raw = pd.read_parquet(request.data_root / params.path)
    schema = LOADER_SCHEMAS.get(request.dataset)
    if schema is not None:
        return coerce_frame(raw, schema)
    return coerce_frame(raw, FrameSchema("extra", tuple(_decimal_spec(name) for name in params.decimal_columns), ()))


class JsonConstraintsParams(Params):
    """Parameters for :func:`json_constraints`."""

    path: str = Field(min_length=1)


def json_constraints(request: LoadRequest, params: JsonConstraintsParams) -> dict[str, dict[str, object]]:
    """Read ``{"<portfolio_id>": {<style constraints>}, ...}`` from a JSON file."""
    loaded = json.loads((request.data_root / params.path).read_text())
    if not isinstance(loaded, dict) or not all(isinstance(value, dict) for value in loaded.values()):
        msg = f"{params.path}: expected an object mapping portfolio ids to constraint objects"
        raise ValueError(msg)
    return {str(portfolio_id): {str(key): value for key, value in constraints.items()} for portfolio_id, constraints in loaded.items()}


def _read_dtypes(schema: FrameSchema) -> dict[str, str]:
    dtypes: dict[str, str] = {}
    for column in schema.columns:
        if column.kind in ("decimal", "datetime_utc"):
            dtypes[column.name] = "string"
        elif column.kind == "bool":
            dtypes[column.name] = "boolean"
        else:
            dtypes[column.name] = column.dtype
    return dtypes


def _parse_utc(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame
    for name in columns:
        result = result.assign(**{name: pd.to_datetime(result[name], utc=True).astype("datetime64[ns, UTC]")})
    return result


def _decimal_spec(name: str) -> ColumnSpec:
    return ColumnSpec(name, "decimal", nullable=True)
