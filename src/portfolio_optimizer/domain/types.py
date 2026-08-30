"""Identifiers, the strict model base classes, and the protocols for injected dependencies."""

from datetime import datetime
from typing import NewType, Protocol

from pydantic import BaseModel, ConfigDict

PortfolioId = NewType("PortfolioId", str)
SecurityId = NewType("SecurityId", str)

STRICT_CONFIG = ConfigDict(strict=True, extra="forbid", frozen=True, validate_default=True, revalidate_instances="always", allow_inf_nan=False)


class StrictModel(BaseModel):
    """Boundary model: strict, frozen, rejects unknown fields and non-finite numbers."""

    model_config = STRICT_CONFIG


class Params(StrictModel):
    """Base class for a step's parameter model.

    A step of any kind — a loader, assembly step, rule, solve-order step, term, constraint, solve step,
    or sink, in ``loaders.py``, ``assembly.py``, ``rules.py``, ``solve_order.py``, ``terms.py``,
    ``solvers.py``, or ``sinks.py`` — declares its parameters by annotating its ``params`` argument with
    a subclass. The engine validates the JSON ``params`` object against that subclass when the config
    resolves, so a typo fails before any data is loaded.
    """


class Clock(Protocol):
    """Source of the current time, injected so runs are reproducible from their manifest."""

    def now(self) -> datetime:
        """Return the current time as a timezone-aware UTC datetime."""
        ...


class IdFactory(Protocol):
    """Source of run identifiers, injected so tests can use fixed ids."""

    def new_run_id(self) -> str:
        """Return a new unique run id."""
        ...
