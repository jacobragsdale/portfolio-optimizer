"""Typed objective terms: the model the config names, the cvxpy the shipped step renders, and the value the verifier recomputes.

A term is a strict, hashable pydantic model exactly as a constraint is: its ``kind`` names it, its
fields are its parameters, :meth:`TypedTerm.to_cvxpy` renders it for the shipped cvxpy step, and
:meth:`TypedTerm.value` recomputes it in plain numpy for the verifier, so a kind a package ships is
verified exactly like a shipped one and the two halves cannot drift. The engine minimizes the sum
of the configured terms; a reward is a term with a negative ``weight``. This module never imports
cvxpy at import time — the renderer reaches for the adapter when called.

The shipped kind is ``linear``: ``weight · cᵀv`` for a per-security column ``c`` (any the spec
carries, ``tax_per_dollar`` and an exported ``alpha`` alike; omitted, every name counts once) and
a decision vector ``v``. Anything convex beyond that is a kind of its own: subclass
:class:`TypedTerm`, write both halves through the adapter's atoms, and publish it under the
:data:`TERM_GROUP` entry-point group or register it in the process.
"""

from collections.abc import Iterator, Mapping, Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar, Literal, override

import numpy as np
from pydantic import Field

from portfolio_optimizer.domain.constraints import CONSTRAINT_NAME_PATTERN, Vector, vector_values
from portfolio_optimizer.domain.order_flow import OrderFlowProfile
from portfolio_optimizer.domain.registry import KindError, kinds_from, parse_kind
from portfolio_optimizer.domain.results import ChainState, MissingSpecColumnError, ProblemSpec, Solution
from portfolio_optimizer.domain.types import StrictModel

if TYPE_CHECKING:
    from portfolio_optimizer.cvx.adapter import DecisionVars, ObjectiveTerm

TERM_GROUP = "portfolio_optimizer.term"
"""The entry-point group a package publishes term kinds under."""


class TermSpecError(ValueError):
    """An objective term is malformed, names an unknown kind, or reads something the spec does not carry."""


class TypedTerm(StrictModel):
    """What every objective term declares; frozen and hashable.

    ``name`` is what the verifier's report and the manifest key on — unique among the run's terms.
    ``weight`` multiplies the term; negative for a reward, since the sum is minimized. A subclass
    narrows ``kind`` to the literal that names it, sets ``reads_chain`` when it reads what
    predecessors traded, and implements :meth:`value`, :meth:`to_cvxpy`, and — for what the spec must
    carry — :meth:`requirements`.
    """

    name: str = Field(pattern=CONSTRAINT_NAME_PATTERN, description="What the report and the manifest key on; unique among the run's terms.")
    weight: Decimal = Field(default=Decimal(1), description="Multiplies the term; negative for a reward, since the sum is minimized. A string, so the manifest records it exactly.")

    reads_chain: ClassVar[bool] = False
    """Whether this kind reads what higher-priority portfolios traded; a run with such a term couples every portfolio through its whole tradable set."""

    def requirements(self, spec: ProblemSpec) -> Iterator[str]:
        """Every reason this term cannot apply to ``spec``; empty when it can."""
        del spec
        return iter(())

    def value(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: OrderFlowProfile) -> float:
        """The term's value at the solution, in plain numpy: the verifier's half."""
        raise NotImplementedError

    def to_cvxpy(self, x: "DecisionVars", spec: ProblemSpec, chain: ChainState) -> "ObjectiveTerm":
        """The term over the decision variables, for the shipped cvxpy step: a convex expression."""
        raise NotImplementedError

    def record(self) -> dict[str, object]:
        """The term as JSON-safe data, the form the manifest records; :func:`parse_term` reads it back."""
        return {str(key): value for key, value in self.model_dump(mode="json").items()}


class Linear(TypedTerm):
    """``weight · cᵀv``: a per-security column against a decision vector — ``alpha`` against ``w`` with a negative weight is the expected-return reward, ``tax_per_dollar`` against ``sell`` the tax cost, ``tcost_per_dollar`` against ``trade`` the trading cost.

    Omit ``column`` and every name counts once, so ``trade`` alone is a turnover penalty.
    """

    kind: Literal["linear"] = Field(default="linear", description="The kind: a per-security column against a decision vector.")
    column: str | None = Field(
        default=None, min_length=1, description="A per-security column of the spec — `alpha`, `tax_per_dollar`, `tcost_per_dollar`, any exported universe column; omitted, every name counts once."
    )
    vector: Vector = Field(default="w", description="The decision vector the column multiplies.")

    def coefficients(self, spec: ProblemSpec) -> np.ndarray:
        """The column's values, or ones."""
        return np.ones(spec.n) if self.column is None else spec.column(self.column)

    @override
    def requirements(self, spec: ProblemSpec) -> Iterator[str]:
        try:
            self.coefficients(spec)
        except MissingSpecColumnError as error:
            yield str(error)

    @override
    def value(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: OrderFlowProfile) -> float:
        del chain, profile
        return float(self.weight) * float((self.coefficients(spec) * vector_values(solution, self.vector)).sum())

    @override
    def to_cvxpy(self, x: "DecisionVars", spec: ProblemSpec, chain: ChainState) -> "ObjectiveTerm":
        from portfolio_optimizer.cvx.adapter import ObjectiveTerm, dot, scale

        del chain
        vector = x.vector(self.vector)  # read the side first: a side the run lacks is a config error, a missing column a question for the data
        return ObjectiveTerm(self.name, scale(float(self.weight), dot(self.coefficients(spec), vector)))


SHIPPED_TERM_KINDS: tuple[type[TypedTerm], ...] = (Linear,)
"""The kinds this package ships; a package adds its own under the :data:`TERM_GROUP` entry-point group."""

_REGISTERED: dict[str, type[TypedTerm]] = {}


def register_term_kind[T: TypedTerm](model: type[T]) -> type[T]:
    """Make ``model`` a known kind in this process — what loading an entry point does, for a notebook or a test; usable as a decorator."""
    _REGISTERED[model.__name__] = model
    return model


def term_kinds() -> Mapping[str, type[TypedTerm]]:
    """Every term kind known here, by name: shipped, published by installed packages, or registered."""
    return kinds_from(TERM_GROUP, TypedTerm, SHIPPED_TERM_KINDS, _REGISTERED.values())


def parse_term(body: Mapping[str, object], where: str = "term") -> TypedTerm:
    """Validate one term record as the kind it names; a failure names ``where``."""
    try:
        return parse_kind(term_kinds(), body, where)
    except KindError as error:
        raise TermSpecError(str(error)) from error


def parse_terms(items: Sequence[object], where: str = "objective") -> tuple[TypedTerm, ...]:
    """The config's ``objective`` list as models, in order; every failure is raised together, and names must be unique."""
    failures: list[str] = []
    terms: list[TypedTerm] = []
    names: dict[str, int] = {}
    for index, item in enumerate(items):
        if isinstance(item, TypedTerm):
            term = item
        elif isinstance(item, Mapping):
            try:
                term = parse_term({str(key): value for key, value in item.items()}, f"{where}[{index}]")
            except TermSpecError as error:
                failures.append(str(error))
                continue
        else:
            failures.append(f"{where}[{index}]: a term is an object with a `kind`, got {type(item).__name__}")
            continue
        if term.name in names:
            failures.append(f"{where}[{index}]: name {term.name!r} is also used by {where}[{names[term.name]}]; names are what reports key on, so they must be unique")
            continue
        names[term.name] = index
        terms.append(term)
    if failures:
        raise TermSpecError("; ".join(failures))
    return tuple(terms)
