"""Run the configured loaders once, combine datasets per the assembly config, and slice per portfolio."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from portfolio_optimizer.config.models import JoinSpec, RunConfig
from portfolio_optimizer.config.resolve import ResolvedConfig, ResolvedStep
from portfolio_optimizer.domain.data import LoadRequest, PortfolioData, details_from_frame, style_constraints_from_mapping
from portfolio_optimizer.domain.frames import FrameSchemaError, validate_frame
from portfolio_optimizer.domain.schemas import COVARIANCE, DATASET_SCHEMAS, DETAILS, PORTFOLIOS
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.hashing import frame_sha256, json_sha256


class LoadError(ValueError):
    """A loader returned something other than its contract promises, or a dataset failed its schema."""


class AssemblyError(ValueError):
    """A join violated its declared cardinality or match requirement."""


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    """Provenance of one loaded dataset for the manifest."""

    name: str
    loader_qualname: str
    loader_source_sha256: str
    params_sha256: str
    rows: int
    columns: tuple[str, ...]
    content_sha256: str


@dataclass(frozen=True, slots=True)
class LoadedDatasets:
    """Everything the loaders returned, before assembly."""

    portfolio_ids: tuple[PortfolioId, ...]
    frames: Mapping[str, pd.DataFrame]
    constraints: Mapping[str, Mapping[str, object]]
    audits: tuple[DatasetAudit, ...]


@dataclass(frozen=True, slots=True)
class AssembledDatasets:
    """Engine-known frames after joins and schema validation, ready to slice per portfolio."""

    portfolio_ids: tuple[PortfolioId, ...]
    holdings: pd.DataFrame
    universe: pd.DataFrame
    details: pd.DataFrame
    targets: pd.DataFrame
    covariance: pd.DataFrame | None
    constraints: Mapping[str, Mapping[str, object]]
    as_of: datetime


def load_datasets(resolved: ResolvedConfig, *, data_root: Path, run_id: str) -> LoadedDatasets:
    """Invoke the portfolio-list loader and then every dataset loader exactly once."""
    as_of = resolved.config.run.as_of
    portfolios = _load_frame(resolved.portfolios, LoadRequest(dataset="portfolios", portfolio_ids=(), as_of=as_of, data_root=data_root, run_id=run_id))
    try:
        validate_frame(portfolios, PORTFOLIOS)
    except FrameSchemaError as error:
        msg = f"portfolios: {error}"
        raise LoadError(msg) from error
    ordered = portfolios.sort_values("solve_order", kind="stable")
    portfolio_ids = tuple(PortfolioId(str(value)) for value in ordered["portfolio_id"])
    audits = [_audit("portfolios", resolved.portfolios, portfolios, PORTFOLIOS.key)]
    frames: dict[str, pd.DataFrame] = {}
    constraints: Mapping[str, Mapping[str, object]] = {}
    for name, step in resolved.loaders.items():
        request = LoadRequest(dataset=name, portfolio_ids=portfolio_ids, as_of=as_of, data_root=data_root, run_id=run_id)
        if name == "constraints":
            constraints = _load_constraints(step, request)
            audits.append(DatasetAudit(name, step.qualname, step.source_sha256, step.params_sha256, len(constraints), (), json_sha256(constraints)))
            continue
        frame = _load_frame(step, request)
        frames[name] = frame
        schema = DATASET_SCHEMAS.get(name)
        audits.append(_audit(name, step, frame, schema.key if schema is not None else ()))
    return LoadedDatasets(portfolio_ids=portfolio_ids, frames=frames, constraints=constraints, audits=tuple(audits))


def _load_frame(step: ResolvedStep, request: LoadRequest) -> pd.DataFrame:
    result = step.invoke(request=request)
    if not isinstance(result, pd.DataFrame):
        msg = f"loader {step.qualname!r} for {request.dataset!r} returned {type(result).__name__}, expected DataFrame"
        raise LoadError(msg)
    return result


def _load_constraints(step: ResolvedStep, request: LoadRequest) -> Mapping[str, Mapping[str, object]]:
    result = step.invoke(request=request)
    msg = f"loader {step.qualname!r} for 'constraints' must return a mapping of portfolio id to constraints mapping"
    if not isinstance(result, Mapping):
        raise LoadError(msg)
    constraints: dict[str, dict[str, object]] = {}
    for portfolio_id, mapping in result.items():
        if not isinstance(mapping, Mapping):
            raise LoadError(msg)
        constraints[str(portfolio_id)] = {str(key): value for key, value in mapping.items()}
    return constraints


def _audit(name: str, step: ResolvedStep, frame: pd.DataFrame, key: tuple[str, ...]) -> DatasetAudit:
    return DatasetAudit(name, step.qualname, step.source_sha256, step.params_sha256, len(frame), tuple(str(column) for column in frame.columns), frame_sha256(frame, key))


def assemble(loaded: LoadedDatasets, config: RunConfig) -> AssembledDatasets:
    """Apply the configured joins, then validate every engine-known frame against its schema."""
    frames = dict(loaded.frames)
    for index, join in enumerate(config.assembly.joins):
        frames[join.into] = _apply_join(frames[join.into], frames[join.source], join, f"assembly.joins[{index}]")
    failures: list[str] = []
    for name, schema in DATASET_SCHEMAS.items():
        if name not in frames:
            continue
        try:
            validate_frame(frames[name], schema)
        except FrameSchemaError as error:
            failures.extend(f"{name}: {failure}" for failure in error.failures)
    if failures:
        raise LoadError("; ".join(failures))
    details = frames["details"]
    missing_details = sorted(set(loaded.portfolio_ids) - {str(value) for value in details["portfolio_id"]})
    if missing_details:
        msg = f"details missing for portfolios {missing_details}"
        raise LoadError(msg)
    missing_constraints = sorted(set(loaded.portfolio_ids) - set(loaded.constraints))
    if missing_constraints:
        msg = f"constraints missing for portfolios {missing_constraints}"
        raise LoadError(msg)
    return AssembledDatasets(
        portfolio_ids=loaded.portfolio_ids,
        holdings=frames["holdings"],
        universe=frames["universe"],
        details=validate_frame(details, DETAILS),
        targets=frames["targets"],
        covariance=validate_frame(frames["covariance"], COVARIANCE) if "covariance" in frames else None,
        constraints=loaded.constraints,
        as_of=config.run.as_of,
    )


def _apply_join(into: pd.DataFrame, source: pd.DataFrame, join: JoinSpec, where: str) -> pd.DataFrame:
    on = list(join.on)
    missing_left = [column for column in on if column not in into.columns]
    missing_right = [column for column in on if column not in source.columns]
    if missing_left or missing_right:
        msg = f"{where}: join columns missing — {join.into} lacks {missing_left}, {join.source} lacks {missing_right}"
        raise AssemblyError(msg)
    overlapping = sorted((set(into.columns) & set(source.columns)) - set(on))
    if overlapping:
        msg = f"{where}: {join.source} would overwrite columns {overlapping} already present in {join.into}"
        raise AssemblyError(msg)
    aligned = source.astype({column: into[column].dtype for column in on})  # a `str` key joined to a `string` key would silently become `object`
    how = "left" if join.require_all_matched else join.how  # an inner join drops unmatched rows before they can be counted
    try:
        merged = into.merge(aligned, on=on, how=how, validate=join.cardinality, indicator=join.require_all_matched)
    except pd.errors.MergeError as error:
        msg = f"{where}: cardinality {join.cardinality!r} violated joining {join.source} into {join.into} on {on}: {error}"
        raise AssemblyError(msg) from error
    if join.require_all_matched:
        unmatched = merged[merged["_merge"] != "both"]
        if len(unmatched):
            keys = unmatched[on].astype(str).agg("|".join, axis=1).tolist()[:10]
            msg = f"{where}: {len(unmatched)} row(s) of {join.into} had no match in {join.source}, e.g. {keys}"
            raise AssemblyError(msg)
        merged = merged.drop(columns=["_merge"])
    return merged


def slice_portfolio(assembled: AssembledDatasets, portfolio_id: PortfolioId) -> PortfolioData:
    """Build the validated per-portfolio bundle: its own holdings and constraints, its benchmark's targets."""
    details = details_from_frame(assembled.details, portfolio_id)
    holdings = assembled.holdings[assembled.holdings["portfolio_id"] == portfolio_id].reset_index(drop=True)
    targets = assembled.targets[assembled.targets["benchmark_id"] == details.benchmark_id].reset_index(drop=True)
    return PortfolioData(
        details=details,
        holdings=holdings,
        universe=assembled.universe.reset_index(drop=True),
        targets=targets,
        covariance=None if assembled.covariance is None else assembled.covariance.reset_index(drop=True),
        style=style_constraints_from_mapping(assembled.constraints[portfolio_id]),
        as_of=assembled.as_of,
    )
