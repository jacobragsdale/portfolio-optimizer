"""Tier 2/3: the published JSON Schema is current, accepts the example, and refuses what the models refuse."""

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.protocols import Validator
from pydantic import ValidationError

from portfolio_optimizer.config.models import config_sha256, load_run_config
from portfolio_optimizer.config.resolve import ConfigResolutionError, resolve_config
from portfolio_optimizer.config.schema import SCHEMA_DIALECT, run_config_schema, schema_json
from tests.conftest import EXAMPLE_CONFIG, REPO_ROOT, example_body, example_datasets

SCHEMA_PATH = REPO_ROOT / "configs" / "run-config.schema.json"


@pytest.fixture(scope="module")
def schema() -> dict[str, object]:
    return run_config_schema()


@pytest.fixture(scope="module")
def validator(schema: dict[str, object]) -> Validator:
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def errors(validator: Validator, instance: object) -> list[str]:
    return sorted(f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}" for error in validator.iter_errors(instance))


def test_checked_in_schema_is_current(schema: dict[str, object]) -> None:
    assert SCHEMA_PATH.read_text() == schema_json(schema), "regenerate with: uv run portfolio-optimizer schema > configs/run-config.schema.json"


def test_schema_declares_its_dialect_and_documents_every_property(schema: dict[str, object]) -> None:
    assert schema["$schema"] == SCHEMA_DIALECT
    undocumented = list(_undocumented(schema, "#"))
    assert undocumented == [], undocumented


def _undocumented(node: object, path: str) -> Iterator[str]:
    if not isinstance(node, dict):
        return
    properties = node.get("properties")
    if isinstance(properties, dict):
        for name, subschema in properties.items():
            if isinstance(subschema, dict) and "description" not in subschema and "$ref" not in subschema:
                yield f"{path}/{name}"
            yield from _undocumented(subschema, f"{path}/{name}")
    defs = node.get("$defs")
    if isinstance(defs, dict):
        for name, definition in defs.items():
            if isinstance(definition, dict) and "description" not in definition:
                yield f"{path}/$defs/{name}"
            yield from _undocumented(definition, f"{path}/$defs/{name}")


def test_the_example_validates_against_the_schema(validator: Validator) -> None:
    assert errors(validator, example_body()) == []


REJECTED: list[tuple[str, dict[str, object], str]] = [
    ("unknown top-level key", {"parallelism": 4}, "<root>"),
    ("dataset scope the engine does not have", {"datasets": example_datasets(holdings={"loader": "load_holdings", "scope": "per_security"})}, "datasets"),
    ("malformed step name", {"rules": ["1bad"]}, "rules/0"),
    ("shipped rule with required params given as a string", {"rules": ["cap_single_name"]}, "rules/0"),
    ("shipped rule with a wrong param type", {"rules": [{"name": "restrict_low_liquidity", "params": {"min_adv_shares": "many"}}]}, "rules/0"),
    ("shipped rule with an unknown param", {"rules": [{"name": "add_zero_alpha", "params": {"fill": 0}}]}, "rules/0"),
    ("shipped term with a negative weight", {"objective": {"terms": [{"name": "alpha", "params": {"weight": -1}}]}}, "objective/terms/0"),
    ("execution mechanics in the config", {"execution": {"on_error": "fail_fast", "max_workers": 2}}, "execution"),
    ("the removed execution mode", {"execution": {"mode": "sequential"}}, "execution"),
    ("shipped assembly step missing a required param", {"assembly": [{"name": "join", "params": {"into": "universe", "source": "analytics", "on": ["security_id"]}}]}, "assembly/0"),
    ("batch size that is not a count", {"datasets": example_datasets(holdings={"loader": "load_holdings", "scope": "per_portfolio", "batch_size": 0})}, "datasets"),
    ("no book of record", {"datasets": {name: spec for name, spec in example_datasets().items() if name != "portfolios"}}, "datasets"),
    ("a book loaded per portfolio", {"datasets": example_datasets(portfolios={"loader": "load_portfolios", "scope": "per_portfolio"})}, "datasets"),
    ("depends_on that is not a list", {"datasets": example_datasets(holdings={"loader": "load_holdings", "depends_on": "portfolios"})}, "datasets"),
    ("an inline list for a dataset that is not the book", {"datasets": example_datasets(holdings={"ids": ["P1"]})}, "datasets"),
]


@pytest.mark.parametrize(("patch", "where"), [case[1:] for case in REJECTED], ids=[case[0] for case in REJECTED])
def test_schema_and_models_agree_on_what_to_reject(validator: Validator, patch: dict[str, object], where: str) -> None:
    instance = example_body() | patch
    found = errors(validator, instance)
    assert any(line.startswith(where) for line in found), found
    try:
        config = load_run_config(json.dumps(instance))
    except (ValidationError, ValueError):
        return  # the models refused it, as the schema did
    with pytest.raises(ConfigResolutionError):  # params-level problems are the resolver's to refuse
        resolve_config(config)


def test_custom_qualified_steps_are_allowed_with_any_params(validator: Validator) -> None:
    instance = example_body() | {"rules": [{"name": "my_firm.rules:tilt", "params": {"anything": [1, 2, 3]}}, "my_firm.rules:plain"]}
    assert errors(validator, instance) == []


def as_object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


def test_every_shipped_step_is_described(schema: dict[str, object]) -> None:
    defs = as_object(schema["$defs"])
    assert "cap_single_name" in str(as_object(defs["RuleStep"])["$comment"])
    assert "transaction_cost" in str(as_object(defs["TermStep"])["$comment"])
    assert "orders_to_parquet" in str(as_object(defs["SinkStep"])["$comment"])
    assert "union" in str(as_object(defs["AssemblyStep"])["$comment"])
    assert "load_holdings" in str(as_object(defs["LoaderStep"])["$comment"])


def test_schema_file_is_valid_json_with_sorted_keys() -> None:
    text = SCHEMA_PATH.read_text()
    assert json.dumps(json.loads(text), indent=2, sort_keys=True, ensure_ascii=False) + "\n" == text


def test_a_schema_pointer_is_accepted_and_does_not_change_the_config_hash(validator: Validator) -> None:
    """The example does not carry one, but a config may: it is what gives an editor live validation."""
    pointed = example_body() | {"$schema": "./run-config.schema.json"}
    assert errors(validator, pointed) == []
    assert load_run_config(json.dumps(pointed)).schema_ref == "./run-config.schema.json"
    assert config_sha256(load_run_config(json.dumps(pointed))) == config_sha256(load_run_config(json.dumps(example_body())))
    assert Path(EXAMPLE_CONFIG).exists()


README = REPO_ROOT / "README.md"
ANNOTATED_BLOCK = re.compile(r"^```jsonc\n(.*?)^```$", re.DOTALL | re.MULTILINE)


def test_the_readme_annotated_config_is_the_real_one_with_comments() -> None:
    """The README carries the example config annotated; the shipped file stays strict JSON with no comments.

    Only whole-line ``//`` comments are used, so stripping them is unambiguous — a trailing comment
    could otherwise be inside a string. The two are compared parsed, not as text, so the annotated copy
    is free to break lines wherever it reads best.
    """
    blocks = ANNOTATED_BLOCK.findall(README.read_text())
    assert len(blocks) == 1, "the README should carry exactly one annotated jsonc config block"
    stripped = "\n".join(line for line in blocks[0].splitlines() if not line.lstrip().startswith("//"))
    assert json.loads(stripped) == json.loads(EXAMPLE_CONFIG.read_text()), "the README's annotated config has drifted from configs/example_run.json; update the annotated copy"
