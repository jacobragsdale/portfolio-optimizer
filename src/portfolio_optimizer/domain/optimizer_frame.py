"""Stack frames that describe the same kind of row — holdings and the buy universe — into one frame.

The optimizer consumes a single frame: every held position and every buyable security as rows, with
whatever analytics columns assembly and rules attached. The two source frames need not carry the same
columns; a column present on one side is null on the other. What they must agree on is the dtype of
every column they share, because a ``Float64`` score on the universe and a ``float64`` score on the
holdings would otherwise be silently unified to something neither side declared. That rule is enforced
here and, for the per-portfolio bundle, on every construction of ``PortfolioData``.
"""

from collections.abc import Mapping

import pandas as pd

_NULLABLE_FORM: Mapping[str, str] = {"bool": "boolean", "int64": "Int64", "float64": "Float64"}
"""Numpy dtypes that cannot hold a missing value, and the pandas dtype each is promoted to when it must."""


class FrameStackError(ValueError):
    """The frames cannot be stacked into one well-typed frame."""


def column_dtype_conflicts(frames: Mapping[str, pd.DataFrame]) -> list[str]:
    """Name every column that two of ``frames`` share with different dtypes, one message per column."""
    first_seen: dict[str, tuple[str, str]] = {}
    conflicts: list[str] = []
    for frame_name, frame in frames.items():
        for column in frame.columns:
            label = str(frame[column].dtype)
            seen = first_seen.get(str(column))
            if seen is None:
                first_seen[str(column)] = (frame_name, label)
            elif seen[1] != label:
                conflicts.append(f"column {column!r}: {seen[0]} has dtype {seen[1]!r}, {frame_name} has {label!r}")
    return conflicts


def stack_frames(frames: Mapping[str, pd.DataFrame], *, source_column: str | None = None) -> pd.DataFrame:
    """Concatenate ``frames`` row-wise over the union of their columns, keeping every dtype.

    Columns keep first-seen order. A column some frame lacks is filled with typed nulls there; when
    its dtype cannot hold a null (``bool``, ``int64``, ``float64``) it is promoted to the nullable
    pandas form (``boolean``, ``Int64``, ``Float64``) in every frame. A column two frames share with
    different dtypes is a :class:`FrameStackError`, as is a dtype that can hold no null at all.
    ``source_column`` names an optional leading ``string`` column carrying each row's frame name.
    """
    if not frames:
        msg = "at least one frame is needed"
        raise FrameStackError(msg)
    conflicts = column_dtype_conflicts(frames)
    if conflicts:
        raise FrameStackError("; ".join(conflicts))
    dtypes: dict[str, str] = {}
    for frame in frames.values():
        for column in frame.columns:
            dtypes.setdefault(str(column), str(frame[column].dtype))
    if source_column is not None and source_column in dtypes:
        msg = f"source column {source_column!r} is already a column of {[name for name, frame in frames.items() if source_column in frame.columns]}"
        raise FrameStackError(msg)
    for column, dtype in dtypes.items():
        if any(column not in frame.columns for frame in frames.values()):
            dtypes[column] = _NULLABLE_FORM.get(dtype, dtype)
    parts = [_aligned(name, frame, dtypes, source_column) for name, frame in frames.items()]
    return pd.concat(parts, ignore_index=True)


def _aligned(name: str, frame: pd.DataFrame, dtypes: Mapping[str, str], source_column: str | None) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    if source_column is not None:
        data[source_column] = pd.Series(name, index=frame.index, dtype="string")
    for column, dtype in dtypes.items():
        if column in frame.columns:
            series = frame[column]
            data[column] = series if str(series.dtype) == dtype else series.astype(pd.api.types.pandas_dtype(dtype))
        else:
            data[column] = _nulls(column, dtype, frame.index)
    return pd.DataFrame(data, index=frame.index)


def _nulls(column: str, dtype: str, index: pd.Index) -> pd.Series:
    if dtype == "object":
        return pd.Series([None] * len(index), index=index, dtype="object")
    try:
        return pd.Series([pd.NA] * len(index), index=index, dtype=dtype)
    except (TypeError, ValueError) as error:
        msg = f"column {column!r} has dtype {dtype!r}, which cannot hold a null for the frames that lack it"
        raise FrameStackError(msg) from error
