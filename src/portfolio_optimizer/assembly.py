"""Assembly steps — yours to edit.

An assembly step is an ordinary function ``(frames: Frames, params: P) -> Frames`` named in the run
config's ``assembly`` list. It runs once per run, after every loader has returned and before the
engine-known frames (``holdings``, ``universe``, ``details``, ``targets``) are validated against their
schemas, and it sees every loaded dataset by name. Steps are pure: same frames in, same frames out,
no I/O. Each is recorded in the manifest with its source hash, its params hash, row counts per dataset
before and after, and the columns it added.

Most assembly work is attaching security analytics to ``universe`` — and to ``holdings`` when it is a
global dataset, since a ``per_portfolio`` one is never passed to assembly and takes its columns from
the ``attach_universe_columns`` rule instead. Both frames accept any columns beyond their schemas, and
the two are later stacked into one optimizer frame, so a column attached to both must have the same
dtype on both. The shipped steps cover the common shapes:
``join`` brings columns from one dataset into another under a declared cardinality; ``union`` stacks
datasets with the same meaning (holdings from two custodians) into one; ``select`` trims and renames
columns; ``drop`` discards a dataset that has served its purpose. Anything else — a computed column, a
pivot, a vendor-specific merge — is a function of the same shape in this module or in your package.
"""

from typing import Literal

import pandas as pd
from pydantic import Field

from portfolio_optimizer.domain.data import Frames
from portfolio_optimizer.domain.optimizer_frame import stack_frames
from portfolio_optimizer.domain.types import Params

type JoinHow = Literal["left", "inner"]
type JoinCardinality = Literal["one_to_one", "one_to_many", "many_to_one"]


class JoinParams(Params):
    """Parameters for :func:`join`."""

    into: str = Field(min_length=1, description="The dataset that receives the columns.")
    source: str = Field(min_length=1, description="The dataset the columns come from; any dataset other than `into`.")
    on: tuple[str, ...] = Field(min_length=1, description="Join key columns present in both datasets. Their dtypes are aligned to `into` before merging so a text key never silently becomes `object`.")
    how: JoinHow = Field(default="left", description="`left` keeps every row of `into`; `inner` keeps only matched rows.")
    cardinality: JoinCardinality = Field(
        description="Expected key cardinality, enforced by pandas `merge(validate=...)`: a duplicate key on the wrong side aborts the run instead of multiplying rows."
    )
    require_all_matched: bool = Field(default=False, description="When true, every row of `into` must find a match in `source`; unmatched keys are reported and the run is rejected.")
    columns: tuple[str, ...] | None = Field(default=None, description="Source columns to bring across, besides the keys. Default: every non-key column of `source`.")
    rename: dict[str, str] = Field(default_factory=dict, description="Source column to the name it takes in `into`, applied after `columns` is chosen.")
    overwrite: bool = Field(default=False, description="Allow brought columns to replace columns `into` already has. Default: such a join is refused.")


def join(frames: Frames, params: JoinParams) -> Frames:
    """Enrich ``into`` with columns from ``source``, matched on ``on`` under a declared cardinality."""
    if params.into == params.source:
        msg = f"cannot join {params.into!r} into itself"
        raise ValueError(msg)
    into = frames[params.into]
    source = frames[params.source]
    on = list(params.on)
    missing_left = [column for column in on if column not in into.columns]
    missing_right = [column for column in on if column not in source.columns]
    if missing_left or missing_right:
        msg = f"join columns missing: {params.into} lacks {missing_left}, {params.source} lacks {missing_right}"
        raise ValueError(msg)
    brought = [str(column) for column in source.columns if column not in on] if params.columns is None else list(params.columns)
    absent = [column for column in brought if column not in source.columns or column in on]
    if absent:
        msg = f"columns {absent} are not non-key columns of {params.source}"
        raise ValueError(msg)
    unknown_renames = sorted(set(params.rename) - set(brought))
    if unknown_renames:
        msg = f"rename refers to columns {unknown_renames} that are not brought from {params.source}"
        raise ValueError(msg)
    trimmed = source[[*on, *brought]].rename(columns=params.rename)
    overlapping = sorted((set(trimmed.columns) & set(into.columns)) - set(on))
    if overlapping and not params.overwrite:
        msg = f"{params.source} would overwrite columns {overlapping} already present in {params.into}; set overwrite to allow it"
        raise ValueError(msg)
    left = into.drop(columns=overlapping) if overlapping else into
    aligned = trimmed.astype({column: left[column].dtype for column in on})
    how = "left" if params.require_all_matched else params.how  # an inner join drops unmatched rows before they can be counted
    try:
        merged = left.merge(aligned, on=on, how=how, validate=params.cardinality, indicator=params.require_all_matched)
    except pd.errors.MergeError as error:
        msg = f"cardinality {params.cardinality!r} violated joining {params.source} into {params.into} on {on}: {error}"
        raise ValueError(msg) from error
    if params.require_all_matched:
        unmatched = merged[merged["_merge"] != "both"]
        if len(unmatched):
            keys = unmatched[on].astype(str).agg("|".join, axis=1).tolist()[:10]
            msg = f"{len(unmatched)} row(s) of {params.into} had no match in {params.source}, e.g. {keys}"
            raise ValueError(msg)
        merged = merged.drop(columns=["_merge"])
    return frames.with_frame(params.into, _nulls_as_none(merged, [params.rename.get(column, column) for column in brought]))


def _nulls_as_none(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Unmatched rows of an ``object`` (Decimal) column come back from ``merge`` as float ``NaN``; make them ``None`` like every other null Decimal."""
    for column in columns:
        if frame[column].dtype == "object" and frame[column].isna().any():
            frame = frame.assign(**{column: frame[column].where(frame[column].notna(), None)})
    return frame


class UnionParams(Params):
    """Parameters for :func:`union`."""

    into: str = Field(min_length=1, description="Name of the stacked result. An existing dataset of this name must be one of `sources`.")
    sources: tuple[str, ...] = Field(min_length=1, description="Datasets stacked in order. Columns they share must agree on dtype; a column some lack is null there.")
    source_column: str | None = Field(default=None, description="Optional column recording which source each row came from.")
    keep_sources: bool = Field(default=False, description="Keep the source datasets after stacking. Default: they are dropped, so only `into` is carried forward.")


def union(frames: Frames, params: UnionParams) -> Frames:
    """Stack ``sources`` row-wise into ``into``, keeping every dtype and refusing conflicting ones."""
    if params.into in frames and params.into not in params.sources:
        msg = f"{params.into!r} already exists and is not among the sources; choose another name or include it"
        raise ValueError(msg)
    stacked = stack_frames({name: frames[name] for name in params.sources}, source_column=params.source_column)
    result = frames.with_frame(params.into, stacked)
    if params.keep_sources:
        return result
    return result.without(*[name for name in dict.fromkeys(params.sources) if name != params.into])


class SelectParams(Params):
    """Parameters for :func:`select`."""

    dataset: str = Field(min_length=1, description="The dataset to trim.")
    columns: tuple[str, ...] | None = Field(default=None, description="Keep exactly these columns, in this order. Mutually exclusive with `drop`.")
    drop: tuple[str, ...] = Field(default=(), description="Columns to remove. Mutually exclusive with `columns`.")
    rename: dict[str, str] = Field(default_factory=dict, description="Old column name to new, applied after `columns` or `drop`.")


def select(frames: Frames, params: SelectParams) -> Frames:
    """Keep, drop, and rename columns of one dataset."""
    if params.columns is not None and params.drop:
        msg = "columns and drop are mutually exclusive"
        raise ValueError(msg)
    frame = frames[params.dataset]
    wanted = list(params.columns) if params.columns is not None else [str(column) for column in frame.columns if column not in params.drop]
    absent = sorted({*wanted, *params.drop, *params.rename} - {str(column) for column in frame.columns})
    if absent:
        msg = f"{params.dataset} has no columns {absent}"
        raise ValueError(msg)
    return frames.with_frame(params.dataset, frame[wanted].rename(columns=params.rename))


class DropParams(Params):
    """Parameters for :func:`drop`."""

    datasets: tuple[str, ...] = Field(min_length=1, description="Datasets to discard; they are not carried into any portfolio's bundle.")


def drop(frames: Frames, params: DropParams) -> Frames:
    """Discard datasets that have served their purpose."""
    return frames.without(*params.datasets)
