"""Generate the JSON Schema (draft 2020-12) for the run config.

The schema is derived from the Pydantic models, so it cannot disagree with what the engine accepts,
and then tightened with what the models alone cannot say: a separate definition per kind of step
with the parameter schema of every step this environment can name — the template's and the ones
installed packages publish — the objective as the union of every known term kind's own schema, and
the required dataset names. The checked-in ``configs/run-config.schema.json`` is this function's
output over the template alone; a test fails when the two drift apart.
"""

import importlib
import inspect
import json
from collections.abc import Mapping
from types import ModuleType
from typing import get_type_hints

from portfolio_optimizer.config.models import STEP_NAME_DESCRIPTION, STEP_NAME_PATTERN, RunConfig
from portfolio_optimizer.config.resolve import CONTRACTS, TEMPLATE_MODULES, published_steps
from portfolio_optimizer.config.steps import StepKind
from portfolio_optimizer.domain.objective import TypedTerm, term_kinds
from portfolio_optimizer.domain.registry import kind_name
from portfolio_optimizer.domain.schemas import REQUIRED_DATASETS
from portfolio_optimizer.domain.types import Params

SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_ID = "https://raw.githubusercontent.com/jacobragsdale/portfolio-optimizer/main/configs/run-config.schema.json"

type JsonObject = dict[str, object]

_ENUM_DESCRIPTIONS: Mapping[str, str] = {
    "DatasetScope": "How a dataset is partitioned across loader calls; see `datasets.<name>.scope`.",
    "DatasetSpec": "One entry of `datasets`: a loaded input, or — for `portfolios` only — the inline list of ids.",
    "Dependencies": "Which higher-priority portfolios a portfolio waits for; see `execution.dependencies`.",
    "JoinCardinality": "Expected key cardinality of a join, enforced by pandas.",
    "JoinHow": "Join type: keep every left row, or only matched rows.",
    "OnError": "What happens after a portfolio fails.",
    "OrderFlow": "The run's order flow: `inflow` (the run buys), `outflow` (the run sells), or `rebalance` (either way, no cash moved on purpose); see `order_flow`.",
    "Vector": "The decision quantity a term or constraint reads: the target weight `w`, the `buy` or `sell` split, or `trade`, the amount traded on the side the run has.",
}

_STEP_DEFINITIONS: Mapping[StepKind, tuple[str, str]] = {
    "loader": ("LoaderStep", "A dataset loader from `loaders.py`: `(request: LoadRequest[, params]) -> DataFrame`, plain or `async def`."),
    "assembly": ("AssemblyStep", "An assembly step from `assembly.py`: `(frames: Frames[, params]) -> Frames`, run once over every loaded dataset."),
    "rule": ("RuleStep", "A business-logic rule from `rules.py`: `(data: PortfolioData[, params]) -> PortfolioData`."),
    "solve_order": ("SolveOrderStep", "A solve-order step from `solve_order.py`: `(data: PortfolioData[, params]) -> Decimal`; lower keys solve first, ties break on `portfolio_id`."),
    "build": ("BuildStep", "The build step from `engine/build.py`: `(data: PortfolioData[, params]) -> ProblemSpec`; `standard` is the default."),
    "solve": ("SolveStep", "The solve step from `solvers.py`: `(request: SolveRequest[, params]) -> SolveResult`; `cvxpy` is the default."),
    "sink": ("SinkStep", "An order sink from `sinks.py`: `(orders: DataFrame, io: IoContext[, params]) -> tuple[Artifact, ...]`."),
}
"""The definition each kind of step gets; the template module behind it is imported when the schema is generated, since the shipped solve step imports cvxpy."""


def run_config_schema() -> JsonObject:
    """The complete, documented JSON Schema for a run config, over every step and term kind this environment can name."""
    base = RunConfig.model_json_schema()
    defs = _object(base["$defs"])
    del defs["StepSpec"]
    for kind, (title, description) in _STEP_DEFINITIONS.items():
        defs[title] = _step_definition(title, description, installed_steps(kind), defs)
    properties = _object(base["properties"])
    properties["assembly"] = _with_items(properties["assembly"], "AssemblyStep")
    properties["rules"] = _with_items(properties["rules"], "RuleStep")
    properties["solve_order"] = _with_nullable_ref(properties["solve_order"], "SolveOrderStep")
    properties["build"] = _with_ref(properties["build"], "BuildStep")
    properties["solve"] = _with_ref(properties["solve"], "SolveStep")
    properties["sink"] = _with_ref(properties["sink"], "SinkStep")
    properties["datasets"] = _datasets_schema(properties["datasets"])
    properties["objective"] = _objective_schema(properties["objective"], defs)
    dataset_config = _object(defs["DatasetConfig"])
    dataset_properties = _object(dataset_config["properties"])
    dataset_properties["loader"] = _with_ref(dataset_properties["loader"], "LoaderStep")
    dataset_config["properties"] = dataset_properties
    defs["DatasetConfig"] = dataset_config
    for name, description in _ENUM_DESCRIPTIONS.items():
        if name in defs:
            defs[name] = {**_object(defs[name]), "description": description}
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


def installed_steps(kind: StepKind) -> dict[str, type[Params] | None]:
    """Every step a bare name of ``kind`` can resolve to, with its params model (or ``None``): the template module's, then what installed packages publish."""
    found = shipped_steps(importlib.import_module(TEMPLATE_MODULES[kind]), kind)
    for name, (module_name, function_name) in published_steps(kind).items():
        if name in found:
            continue  # the template module wins a bare name, as resolution does
        function = getattr(importlib.import_module(module_name), function_name, None)
        if not inspect.isfunction(function):
            continue
        params = get_type_hints(function).get("params")
        found[name] = params if inspect.isclass(params) and issubclass(params, Params) else None
    return found


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
    """Which kind a public function of a template module is: the module's kind, when the function returns what that kind's contract promises; a helper returns something else."""
    for kind, template in TEMPLATE_MODULES.items():
        if module.__name__ == template:
            return kind if any(returns == allowed for allowed in CONTRACTS[kind].returns) else None
    return None


def _step_definition(title: str, description: str, shipped: Mapping[str, type[Params] | None], defs: JsonObject) -> JsonObject:
    needs_params = sorted(name for name, model in shipped.items() if model is not None and any(field.is_required() for field in model.model_fields.values()))
    string_form: JsonObject = {"type": "string", "pattern": STEP_NAME_PATTERN, "description": f"A step without parameters. {STEP_NAME_DESCRIPTION}"}
    if needs_params:
        string_form["not"] = {"enum": needs_params}
        string_form["$comment"] = f"These steps have required parameters and must use the object form: {needs_params}"
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
            "params": {"type": "object", "description": "Parameters validated against the function's `params` model; for the steps this environment can name the exact shape is given below."},
        },
        "required": ["name"],
        "additionalProperties": False,
        "allOf": conditions,
    }
    return {"title": title, "description": description, "$comment": f"Steps this environment can name: {sorted(shipped)}", "anyOf": [string_form, object_form]}


def _params_schema(model: type[Params], defs: JsonObject) -> JsonObject:
    """A params model's schema with its own definitions (enum aliases, nested models) hoisted into the top-level ``$defs``."""
    schema = _object(model.model_json_schema())
    schema.pop("title", None)
    _hoist(schema, defs, model.__name__)
    return schema


def _hoist(schema: JsonObject, defs: JsonObject, owner: str) -> None:
    for name, definition in _object(schema.pop("$defs", {})).items():
        if name in defs and defs[name] != definition:
            msg = f"{owner} defines {name!r} differently from an existing definition"
            raise ValueError(msg)
        defs[name] = definition


def _objective_schema(objective: object, defs: JsonObject) -> JsonObject:
    """The objective as an array whose items are any known term kind, each validated against its model's own schema."""
    schema = _object(objective)
    kinds: list[JsonObject] = []
    for name, model in sorted(term_kinds().items()):
        kind_schema = _term_kind_schema(model, defs)
        kinds.append({"title": f"{name} term", **kind_schema})
    return {
        **schema,
        "items": {"anyOf": kinds, "description": "One objective term: an object whose `kind` names its model."},
        "$comment": f"Term kinds this environment can name: {sorted(term_kinds())}",
    }


def _term_kind_schema(model: type[TypedTerm], defs: JsonObject) -> JsonObject:
    schema = _object(model.model_json_schema())
    schema.pop("title", None)
    properties = _object(schema.get("properties", {}))
    properties["kind"] = {"description": f"The kind's name: `{kind_name(model)}`.", **_object(properties.get("kind", {}))}
    schema["properties"] = properties
    _hoist(schema, defs, model.__name__)
    return schema


def _datasets_schema(datasets: object) -> JsonObject:
    schema = _object(datasets)
    schema.pop("additionalProperties", None)
    properties: JsonObject = {name: {"$ref": "#/$defs/DatasetConfig"} for name in (*REQUIRED_DATASETS, "constraints")}
    properties["portfolios"] = _portfolios_property()
    return {
        **schema,
        "properties": dict(sorted(properties.items())),
        "required": ["portfolios"],
        "additionalProperties": {"$ref": "#/$defs/DatasetConfig"},
        "$comment": f"`portfolios` is always required; {list(REQUIRED_DATASETS)} are required unless an assembly step produces them. `constraints` is engine-known but optional. Any other key is an extra dataset, available to assembly steps and carried into each portfolio's bundle.",
    }


def _portfolios_property() -> JsonObject:
    """The one dataset that may be written inline: a loaded input held to `scope: global`, the `{"ids": [...]}` object, or the bare array of ids."""
    return {
        "description": 'The portfolio list: a loaded dataset (always `scope: global`), or the ids written inline — `{"ids": [...]}` or the bare array, whose written order is the solve order.',
        "anyOf": [
            {"allOf": [{"$ref": "#/$defs/DatasetConfig"}, {"properties": {"scope": {"const": "global"}}}]},
            {"$ref": "#/$defs/InlinePortfolios"},
            {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1, "$comment": 'A bare array is shorthand for {"ids": [...]}.'},
        ],
    }


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
