"""Deterministic content hashes for frames, JSON-like values, and files."""

import hashlib
import json
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd


def frame_sha256(frame: pd.DataFrame, key: Sequence[str] = ()) -> str:
    """Hash a frame's content independently of row order (given ``key``), column order, and index.

    Decimals are normalized (``0.50`` and ``0.5`` hash equal), timestamps become UTC nanoseconds,
    and the dtype of every column is part of the hash so an ``Int64`` → ``float64`` drift changes it.
    """
    canonical = frame.reindex(sorted(str(column) for column in frame.columns), axis=1)
    if key:
        canonical = canonical.sort_values(list(key), kind="stable")
    canonical = canonical.reset_index(drop=True)
    digest = hashlib.sha256()
    for name in canonical.columns:
        column = canonical[name]
        digest.update(f"{name}:{_dtype_label(column)}".encode())
        digest.update(_column_bytes(column))
    return digest.hexdigest()


def _dtype_label(column: pd.Series) -> str:
    """Equal instants hash equal regardless of the zone they were expressed in."""
    if isinstance(column.dtype, pd.DatetimeTZDtype):
        return "datetime64[ns, UTC]"
    return str(column.dtype)


def _column_bytes(column: pd.Series) -> bytes:
    if isinstance(column.dtype, pd.DatetimeTZDtype):
        naive = column.dt.tz_convert("UTC").dt.tz_localize(None)
        return _bytes(naive.to_numpy(dtype="datetime64[ns]").astype(np.int64).tobytes())
    if column.dtype == "object":
        rendered = pd.Series([_render(value) for value in column], dtype="string")
        return pd.util.hash_pandas_object(rendered, index=False).to_numpy().tobytes()
    if str(column.dtype) in ("Float64", "float64"):
        return _bytes((column.to_numpy(dtype="float64", na_value=np.nan) + 0.0).tobytes())
    return pd.util.hash_pandas_object(column, index=False).to_numpy().tobytes()


def _bytes(value: object) -> bytes:
    """pandas' array conversions are untyped; establish the type at this boundary."""
    if not isinstance(value, bytes):
        msg = f"expected bytes, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _render(value: object) -> str:
    if value is None or value is pd.NA:
        return "<NA>"
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    return repr(value)


def json_sha256(value: object) -> str:
    """Hash a JSON-serializable value in canonical form (sorted keys, no whitespace)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode()).hexdigest()


def _json_default(value: object) -> str:
    if isinstance(value, Decimal):
        return format(value, "f")
    msg = f"object of type {type(value).__name__} is not JSON serializable"
    raise TypeError(msg)


def file_sha256(path: Path) -> str:
    """Hash a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
