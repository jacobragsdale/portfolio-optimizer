"""Dataset loaders — yours to edit.

A loader is an ordinary function ``(request: LoadRequest, params: P) -> pd.DataFrame`` named in
the run config; it may be ``async def``. The ``constraints`` dataset's loader returns
``dict[str, dict[str, object]]`` keyed by portfolio id instead. Loaders are the only place file,
database, or network access belongs; everything downstream is pure.

The engine loads every dataset concurrently: an async loader runs on the event loop, a plain one
in a worker thread. A loader that makes many calls — one per portfolio, say — wraps each call in
``async with request.rate_limiter:`` (``with request.rate_limiter.sync:`` from a plain loader) so a
large run stays inside the backend's limits; the pool is configured per dataset in the run config.

The shipped loaders read files under ``request.data_root``. For an engine-known dataset they cast
columns to that dataset's schema; for any other dataset ``dtypes`` declares each column's kind in
the same vocabulary the schemas are written in. :func:`csv_per_portfolio` is the fan-out pattern —
one call per portfolio, concurrently, under the rate limit — with files in place of a network
client.
"""

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from pydantic import Field

from portfolio_optimizer.domain.data import LoadRequest
from portfolio_optimizer.domain.frames import ColumnKind, ColumnSpec, FrameSchema, coerce_frame
from portfolio_optimizer.domain.schemas import DATASET_SCHEMAS, PORTFOLIOS
from portfolio_optimizer.domain.types import Params, PortfolioId
from portfolio_optimizer.ratelimit import fan_out

LOADER_SCHEMAS: Mapping[str, FrameSchema] = {**DATASET_SCHEMAS, "portfolios": PORTFOLIOS}


class ExtraColumns(Params):
    """How to type the columns of a dataset the engine does not know.

    Each column is declared as one of the kinds the engine's own schemas are written in, so an extra
    dataset is typed in the same vocabulary as an engine-known one. A column not named here arrives as
    pandas inferred it; for an engine-known dataset the schema types every column and this is ignored.
    """

    dtypes: dict[str, ColumnKind] = Field(
        default_factory=dict,
        description="Column name to kind: `string`, `Int64`, `Float64`, `bool`, `decimal` (an exact `Decimal` — money, prices, rates), or `datetime_utc` (a timezone-aware timestamp).",
    )


class CsvParams(ExtraColumns):
    """Parameters for :func:`csv`."""

    path: str = Field(min_length=1)


def csv(request: LoadRequest, params: CsvParams) -> pd.DataFrame:
    """Read a CSV file with every dtype declared up front."""
    return _read_csv(request.data_root / params.path, request.dataset, params)


class CsvPerPortfolioParams(ExtraColumns):
    """Parameters for :func:`csv_per_portfolio`."""

    directory: str = Field(min_length=1)


async def csv_per_portfolio(request: LoadRequest, params: CsvPerPortfolioParams) -> pd.DataFrame:
    """Read ``<directory>/<portfolio_id>.csv`` for every requested portfolio, concurrently, under the dataset's rate limit.

    This is the shape of a loader for a source that answers one portfolio per call. Each read runs
    in a worker thread so the event loop stays free; ``request.rate_limiter`` decides how many run
    at once. Swap the thread call for an ``await client.get(...)`` and the structure is unchanged.
    """
    directory = request.data_root / params.directory

    async def read(portfolio_id: PortfolioId) -> pd.DataFrame:
        return await asyncio.to_thread(_read_csv, directory / f"{portfolio_id}.csv", request.dataset, params)

    parts = await fan_out(request.portfolio_ids, read, limiter=request.rate_limiter)
    if not parts:
        return _empty_frame(request.dataset, params)
    return pd.concat(parts, ignore_index=True)


class ParquetParams(ExtraColumns):
    """Parameters for :func:`parquet`."""

    path: str = Field(min_length=1)


def parquet(request: LoadRequest, params: ParquetParams) -> pd.DataFrame:
    """Read a Parquet file; Arrow decimal columns arrive as ``Decimal`` already."""
    raw = pd.read_parquet(request.data_root / params.path)
    return coerce_frame(raw, _schema_for(request.dataset, params))


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


def _read_csv(path: Path, dataset: str, columns: ExtraColumns) -> pd.DataFrame:
    schema = _schema_for(dataset, columns)
    raw = pd.read_csv(path, dtype=_read_dtypes(schema))
    return coerce_frame(_parse_utc(raw, [c.name for c in schema.columns if c.kind == "datetime_utc" and c.name in raw.columns]), schema)


def _empty_frame(dataset: str, columns: ExtraColumns) -> pd.DataFrame:
    """No portfolios were requested: a zero-row frame with the columns the dataset's schema declares."""
    return pd.DataFrame({column.name: pd.Series(dtype=column.dtype) for column in _schema_for(dataset, columns).columns})


def _schema_for(dataset: str, columns: ExtraColumns) -> FrameSchema:
    """The dataset's own schema when the engine knows it, otherwise one built from the declared kinds."""
    return LOADER_SCHEMAS.get(dataset) or FrameSchema("extra", tuple(ColumnSpec(name, kind, nullable=kind != "bool") for name, kind in columns.dtypes.items()), ())


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
