"""Generate the JSON Schema (draft 2020-12) for the run config.

The schema is derived from the Pydantic models, so it cannot disagree with what the engine accepts,
and then tightened with what the models alone cannot say: a separate definition per kind of step
with the parameter schema of every shipped function and the required dataset names. The checked-in ``configs/run-config.schema.json`` is
this function's output; a test fails when the two drift apart.
"""

import inspect
import json
from collections.abc import Mapping
from types import ModuleType
from typing import get_type_hints

import pandas as pd

from portfolio_optimizer import assembly, loaders, rules, sinks, solve_order, solvers, terms
from portfolio_optimizer.config.models import STEP_NAME_DESCRIPTION, STEP_NAME_PATTERN, RunConfig
from portfolio_optimizer.config.steps import StepKind
from portfolio_optimizer.cvx.adapter import ConstraintSet, ObjectiveTerm
from portfolio_optimizer.domain.schemas import REQUIRED_DATASETS, REQUIRED_FRAMES
from portfolio_optimizer.domain.types import Params

SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_ID = "https://raw.githubusercontent.com/jacobragsdale/portfolio-optimizer/main/configs/run-config.schema.json"

type JsonObject = dict[str, object]

_ENUM_DESCRIPTIONS: Mapping[str, str] = {
    "DatasetScope": "How a dataset is partitioned across loader calls; see `datasets.<name>.scope`.",
    "Dependencies": "Which higher-priority portfolios a portfolio waits for; see `execution.dependencies`.",
    "JoinCardinality": "Expected key cardinality of a join, enforced by pandas.",
    "JoinHow": "Join type: keep every left row, or only matched rows.",
    "OnError": "What happens after a portfolio fails.",
    "Sides": "Which side the run trades: `both`, `buy`, or `sell`; see `sides`.",
}

_STEP_DEFINITIONS: Mapping[StepKind, tuple[str, str, ModuleType]] = {
    "loader": ("LoaderStep", "A dataset loader from `loaders.py`: `(request: LoadRequest[, params]) -> DataFrame`, plain or `async def`.", loaders),
    "assembly": ("AssemblyStep", "An assembly step from `assembly.py`: `(frames: Frames[, params]) -> Frames`, run once over every loaded dataset.", assembly),
    "rule": ("RuleStep", "A business-logic rule from `rules.py`: `(data: PortfolioData[, params]) -> PortfolioData`.", rules),
    "solve_order": ("SolveOrderStep", "A solve-order step from `solve_order.py`: `(data: PortfolioData[, params]) -> Decimal`; lower keys solve first, ties break on `portfolio_id`.", solve_order),
    "term": ("TermStep", "An objective term from `terms.py`: `(x: DecisionVars, spec: ProblemSpec[, params][, chain: ChainState]) -> ObjectiveTerm`.", terms),
    "solve": ("SolveStep", "The solve step from `solvers.py`: `(request: SolveRequest[, params]) -> SolveResult`; `cvxpy` is the default.", solvers),
    "sink": ("SinkStep", "An order sink from `sinks.py`: `(orders: DataFrame, io: IoContext[, params]) -> tuple[Artifact, ...]`.", sinks),
}


def run_config_schema() -> JsonObject:
    """The complete, documented JSON Schema for a run config."""
    base = RunConfig.model_json_schema()
    defs = _object(base["$defs"])
    del defs["StepSpec"]
    for kind, (title, description, module) in _STEP_DEFINITIONS.items():
        defs[title] = _step_definition(title, description, shipped_steps(module, kind), defs)
    for name, description in _ENUM_DESCRIPTIONS.items():
        defs[name] = {**_object(defs[name]), "description": description}
    properties = _object(base["properties"])
    properties["portfolios"] = _portfolios_schema(properties["portfolios"])
    properties["assembly"] = _with_items(properties["assembly"], "AssemblyStep")
    properties["rules"] = _with_items(properties["rules"], "RuleStep")
    properties["solve_order"] = _with_nullable_ref(properties["solve_order"], "SolveOrderStep")
    properties["solve"] = _with_ref(properties["solve"], "SolveStep")
    properties["sink"] = _with_ref(properties["sink"], "SinkStep")
    properties["datasets"] = _datasets_schema(properties["datasets"])
    dataset_config = _object(defs["DatasetConfig"])
    dataset_properties = _object(dataset_config["properties"])
    dataset_properties["loader"] = _with_ref(dataset_properties["loader"], "LoaderStep")
    dataset_config["properties"] = dataset_properties
    defs["DatasetConfig"] = dataset_config
    objective = _object(defs["ObjectiveConfig"])
    objective_properties = _object(objective["properties"])
    objective_properties["terms"] = _with_items(objective_properties["terms"], "TermStep")
    objective["properties"] = objective_properties
    defs["ObjectiveConfig"] = objective
    return {
        "$schema": SCHEMA_DIALECT,
        "$id": SCHEMA_ID,
        "title": "Portfolio optimizer run config",
        **{key: value for key, value in base.items() if key not in ("$defs", "title", "properties")},
        "properties": properties,
        "if": {"properties": {"assembly": {"minItems": 1}}, "required": ["assembly"]},
        "else": {"properties": {"datasets": {"required": list(REQUIRED_DATASETS)}}, "$comment": "Without assembly steps, nothing can produce the engine-known frames, so every one must be loaded."},
        "$defs": dict(sorted(defs.items())),
    }


def schema_json(schema: JsonObject) -> str:
    """Canonical text form of the schema: sorted keys, two-space indent, trailing newline."""
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def shipped_steps(module: ModuleType, kind: StepKind) -> dict[str, type[Params] | None]:
    """Every public function in ``module`` that is a step of ``kind``, with its params model (or ``None``)."""
    found: dict[str, type[Params] | None] = {}
    for name, value in sorted(vars(module).items()):
        if name.startswith("_") or not inspect.isfunction(value) or value.__module__ != module.__name__:
            continue
        hints = get_type_hints(value)
        if _kind_of(module, hints.get("return")) != kind:
            continue
        params = hints.get("params")
        found[name] = params if inspect.isclass(params) and issubclass(params, Params) else None
    return found


def _kind_of(module: ModuleType, returns: object) -> StepKind | None:
    if module is loaders:
        return "loader" if returns is pd.DataFrame else None
    if module is assembly:
        return "assembly"
    if module is rules:
        return "rule"
    if module is solve_order:
        return "solve_order"
    if module is sinks:
        return "sink"
    if module is solvers:
        return "solve"
    if returns is ObjectiveTerm:
        return "term"
    if returns is ConstraintSet:
        return "constraint"
    return None


def _step_definition(title: str, description: str, shipped: Mapping[str, type[Params] | None], defs: JsonObject, extra_properties: JsonObject | None = None) -> JsonObject:
    needs_params = sorted(name for name, model in shipped.items() if model is not None and any(field.is_required() for field in model.model_fields.values()))
    string_form: JsonObject = {"type": "string", "pattern": STEP_NAME_PATTERN, "description": f"A step without parameters. {STEP_NAME_DESCRIPTION}"}
    if needs_params:
        string_form["not"] = {"enum": needs_params}
        string_form["$comment"] = f"These shipped steps have required parameters and must use the object form: {needs_params}"
    conditions: list[JsonObject] = []
    for name, model in shipped.items():
        params_schema: JsonObject = (
            _params_schema(model, defs) if model is not None else {"type": "object", "additionalProperties": False, "maxProperties": 0, "description": f"`{name}` takes no parameters."}
        )
        then: JsonObject = {"properties": {"params": params_schema}}
        if name in needs_params:
            then["required"] = ["params"]
        conditions.append({"if": {"properties": {"name": {"const": name}}, "required": ["name"]}, "then": then})
    object_form: JsonObject = {
        "type": "object",
        "description": "A step with parameters.",
        "properties": {
            "name": {"type": "string", "pattern": STEP_NAME_PATTERN, "description": STEP_NAME_DESCRIPTION},
            "params": {"type": "object", "description": "Parameters validated against the function's `params` model; for shipped steps the exact shape is given below."},
            **(extra_properties or {}),
        },
        "required": ["name"],
        "additionalProperties": False,
        "allOf": conditions,
    }
    return {"title": title, "description": description, "$comment": f"Shipped steps: {sorted(shipped)}", "anyOf": [string_form, object_form]}


def _params_schema(model: type[Params], defs: JsonObject) -> JsonObject:
    """A params model's schema with its own definitions (enum aliases, nested models) hoisted into the top-level ``$defs``."""
    schema = _object(model.model_json_schema())
    schema.pop("title", None)
    for name, definition in _object(schema.pop("$defs", {})).items():
        if name in defs and defs[name] != definition:
            msg = f"params model {model.__name__} defines {name!r} differently from an existing definition"
            raise ValueError(msg)
        defs[name] = definition
    return schema


def _datasets_schema(datasets: object) -> JsonObject:
    schema = _object(datasets)
    properties: JsonObject = {name: {"$ref": "#/$defs/DatasetConfig"} for name in (*REQUIRED_FRAMES, "constraints")}
    return {
        **schema,
        "properties": dict(sorted(properties.items())),
        "additionalProperties": {"$ref": "#/$defs/DatasetConfig"},
        "$comment": f"Required unless an assembly step produces them: {list(REQUIRED_FRAMES)}. `constraints` is engine-known but optional. Any other key is an extra dataset, available to assembly steps and carried into each portfolio's bundle.",
    }


def _portfolios_schema(property_schema: object) -> JsonObject:
    """A bare loader step, or the full `{"loader": step, "rate_limit": ...}` form; the model normalizes the first into the second."""
    schema = _object(property_schema)
    schema.pop("$ref", None)
    schema.pop("allOf", None)
    return {**schema, "anyOf": [{"$ref": "#/$defs/LoaderStep"}, {"$ref": "#/$defs/DatasetConfig"}], "$comment": 'A bare step is shorthand for {"loader": step}.'}


def _with_ref(property_schema: object, definition: str) -> JsonObject:
    schema = _object(property_schema)
    schema.pop("allOf", None)
    return {**schema, "$ref": f"#/$defs/{definition}"}


def _with_nullable_ref(property_schema: object, definition: str) -> JsonObject:
    """An optional step: pydantic emits ``anyOf [StepSpec, null]``; point the first branch at the kind's definition."""
    schema = _object(property_schema)
    schema.pop("anyOf", None)
    return {**schema, "anyOf": [{"$ref": f"#/$defs/{definition}"}, {"type": "null"}]}


def _with_items(property_schema: object, definition: str) -> JsonObject:
    schema = _object(property_schema)
    return {**schema, "items": {"$ref": f"#/$defs/{definition}"}}


def _object(value: object) -> JsonObject:
    if not isinstance(value, dict):
        msg = f"expected a JSON object, got {type(value).__name__}"
        raise TypeError(msg)
    return {str(key): item for key, item in value.items()}
