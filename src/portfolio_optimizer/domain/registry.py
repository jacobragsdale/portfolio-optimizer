"""The kind registries: how a ``kind`` in a constraint row or an objective term finds its model.

A typed constraint or term is a strict pydantic model whose ``kind`` field is a literal naming it.
The shipped kinds live beside their base classes; a package adds kinds by declaring entry points
in the group named here (``portfolio_optimizer.constraints`` or ``portfolio_optimizer.terms``), and
a notebook or a test registers one directly. Every consumer — the config resolver, the schedule,
the shipped cvxpy step, the verifier, the JSON Schema — reads the same registry, so a kind that is
known anywhere is known everywhere.
"""

import functools
import importlib.metadata
import json
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ValidationError


class KindError(ValueError):
    """A kind is unknown, an entry point is not a kind, or a kind's fields do not validate."""


def kind_name(model: type[BaseModel]) -> str:
    """The literal a model's ``kind`` field carries; a base class without one has no name."""
    kind = model.model_fields.get("kind")
    if kind is None or not isinstance(kind.default, str):
        msg = f"{model.__name__} declares no `kind` literal"
        raise KindError(msg)
    return kind.default


@functools.cache
def _entry_point_kinds(group: str) -> tuple[type[BaseModel], ...]:
    """Every model an installed package published under ``group``; loaded once per process."""
    found: list[type[BaseModel]] = []
    for entry_point in importlib.metadata.entry_points(group=group):
        loaded = entry_point.load()
        if not (isinstance(loaded, type) and issubclass(loaded, BaseModel)):
            msg = f"entry point {entry_point.name!r} in {group!r} is {loaded!r}, not a model class"
            raise KindError(msg)
        found.append(loaded)
    return tuple(found)


def kinds_from[T: BaseModel](group: str, base: type[T], shipped: Iterable[type[T]], registered: Iterable[type[T]]) -> dict[str, type[T]]:
    """The registry for ``base``: the shipped kinds, the ones packages published under ``group``, and the ones registered in this process, by kind name."""
    kinds: dict[str, type[T]] = {}
    for model in (*shipped, *_entry_point_kinds(group), *registered):
        if not issubclass(model, base):
            msg = f"{model.__name__} is not a {base.__name__}"
            raise KindError(msg)
        name = kind_name(model)
        if name in kinds and kinds[name] is not model:
            msg = f"kind {name!r} is declared by both {kinds[name].__name__} and {model.__name__}"
            raise KindError(msg)
        kinds[name] = model
    return kinds


def parse_kind[T: BaseModel](kinds: Mapping[str, type[T]], body: Mapping[str, object], where: str) -> T:
    """Validate ``body`` as the kind it names; every failure names ``where`` — a row, a config index."""
    kind = body.get("kind")
    if not isinstance(kind, str) or not kind:
        msg = f"{where}: a `kind` is required; known kinds: {sorted(kinds)}"
        raise KindError(msg)
    model = kinds.get(kind)
    if model is None:
        msg = f"{where}: unknown kind {kind!r}; known kinds: {sorted(kinds)}"
        raise KindError(msg)
    try:
        return model.model_validate_json(json.dumps(dict(body)))
    except ValidationError as error:
        details = "; ".join(f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}" for item in error.errors())
        msg = f"{where}: {kind}: {details}"
        raise KindError(msg) from error
    except (TypeError, ValueError) as error:
        msg = f"{where}: {kind}: {error}"
        raise KindError(msg) from error
