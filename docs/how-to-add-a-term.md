# How to add an objective term or a constraint kind

Terms and constraints are typed *kinds*: strict pydantic models that carry both halves of themselves —
`to_cvxpy` for the shipped solve step, and a plain-numpy `value` (a term) or `residual` (a constraint)
for the verifier — so nothing the solver was told goes unchecked. A term is a record in the config's
`objective`; a constraint is a row in the `constraints` dataset. This guide writes a term kind the
shipped `linear` cannot express, a constraint kind, and makes each known to a run.

## Prerequisites

- Check the shipped kinds first: `uv run portfolio-optimizer steps` lists them with their fields. A
  linear reward or cost over any per-security column is a `linear` term record, not code; a cap on a
  name, a group, an exposure, cash, turnover, or ADV participation is a constraint row of a shipped
  kind ([the reference](reference-run-config.md#constraints-the-constraints-dataset)). Write a kind
  only for a shape none of them has.
- The data the kind needs is on the `ProblemSpec`: the fixed vectors (`w0`, `price`, `shares_held`,
  `lot_size`, `lb`, `ub`), named `columns` (the derived `tax_per_dollar`, `tcost_per_dollar`, and
  `adv_capacity`, plus every numeric universe column beyond the schema, `alpha` included), `flags`
  (every boolean column), `groups` (every string column, as a sparse membership matrix), and
  `scalars` (every number on the account's `details` row). Read them with `spec.column(name)`,
  `spec.flag(name)`, `spec.group(column)`, and `spec.scalar(name)`; see
  [the bundle reference](reference-portfolio-data.md#problemspec). A signal that is not in the
  universe yet is attached with an assembly step or a rule first
  ([how to add security analytics](how-to-add-security-analytics.md)).

## 1. Write a term kind — in `src/portfolio_optimizer/domain/objective.py`, or in your package

A kind subclasses `TypedTerm`, narrows `kind` to the literal that names it, declares its fields, and
implements three methods. `name` and `weight` come from the base. This one is a diagonal variance
penalty, `weight · Σ dᵢ wᵢ²` — convex, so not a `linear`:

```python
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal, override

import numpy as np
from pydantic import Field

from portfolio_optimizer.domain.objective import TypedTerm
from portfolio_optimizer.domain.results import ChainState, MissingSpecColumnError, ProblemSpec, Solution
from portfolio_optimizer.domain.order_flow import OrderFlowProfile

if TYPE_CHECKING:
    from portfolio_optimizer.cvx.adapter import DecisionVars, ObjectiveTerm


class DiagonalRisk(TypedTerm):
    """``weight · Σ dᵢ wᵢ²``: a diagonal variance penalty over a per-security variance column."""

    kind: Literal["diagonal_risk"] = "diagonal_risk"
    column: str = Field(default="variance", min_length=1, description="A per-security variance column of the spec.")

    @override
    def requirements(self, spec: ProblemSpec) -> Iterator[str]:
        try:
            spec.column(self.column)
        except MissingSpecColumnError as error:
            yield str(error)

    @override
    def value(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: OrderFlowProfile) -> float:
        return float(self.weight) * float((spec.column(self.column) * solution.w**2).sum())

    @override
    def to_cvxpy(self, x: "DecisionVars", spec: ProblemSpec, chain: ChainState) -> "ObjectiveTerm":
        from portfolio_optimizer.cvx.adapter import ObjectiveTerm, scale, sum_squares, weighted

        return ObjectiveTerm(self.name, scale(float(self.weight), sum_squares(weighted(np.sqrt(spec.column(self.column)), x.w))))
```

- `kind` is the name a config record uses. `requirements` yields every reason the kind cannot apply to
  a spec — a missing column, flag, scalar, or group — and the build fails the portfolio with them at
  stage `build`, before any solve is scheduled on it.
- `value` is the verifier's half: the term at the solution, in plain numpy. `to_cvxpy` is the shipped
  solve step's half, and it imports the adapter inside the method so the module stays cvxpy-free and
  `verify` runs without the solver stack. The two must agree; that is the whole contract.
- `x.w` is the target weight and `x.trade` the amount traded, as fractions of NAV; `x.buy` and
  `x.sell` are the non-negative split and exist only on a side the run has — reading `x.sell` in an
  inflow raises `SideUnavailableError`, which `validate-config` reports, naming the side.
  `x.coupled` is the amount traded on the side the run couples through, and `x.vector(name)` reads any
  of them by name. Write "the amount traded" as `x.trade` and the term runs under every `order_flow`.
- Write the math through the atoms in `portfolio_optimizer.cvx.adapter`: the affine `total`, `dot`,
  `matvec`, `scale`, `weighted`, `masked`, `plus`, `minus`, `shifted`, `shortfall`; the convex
  `sum_squares`, `norm1`, `absolute`, `pos`; and the comparisons `equals`, `at_most`, `at_least`. They
  are the whole cvxpy surface the template exposes. A convex atom scaled by a negative weight is not
  DCP, and `validate-config` refuses it: every term is rendered once against a one-security dummy
  spec under the run's `order_flow` and the problem is checked for DCP compliance before any data loads.
- A kind that reads `chain.traded_shares` sets `reads_chain: ClassVar[bool] = True`; see step 4.
- Read numbers from the spec or from the kind's own fields, never from a file or a global: the spec is
  hashed into the manifest and the term's record is written to it, so anything the term used is
  recorded.

A term that *rewards* a side — a negative cost per unit of `x.sell`, a harvestable loss — is exact,
because the run has one variable: the trade is an expression of `w`, no name can be bought and sold in
one solve, and there is no round trip for the reward to pay for. That is the profiles' property, not
the term's, so a kind need not guard against it.

## 2. Write a constraint kind — in `src/portfolio_optimizer/domain/constraints.py`, or in your package

A constraint kind subclasses `TypedConstraint`, which supplies `name`, `direction`, `scope`,
`allow_current_weight`, and `tolerance`, and implements `requirements`, `residual`, and `to_cvxpy`.
This one caps the total absolute deviation from a per-security target column — `Σ |wᵢ − tᵢ| ≤ bound`
over the scope — which no affine shipped kind can say:

```python
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal, Self, override

import numpy as np
from pydantic import Field, model_validator

from portfolio_optimizer.domain.constraints import ScalarBound, TypedConstraint, bounds_above, scalar_bound
from portfolio_optimizer.domain.results import F64, ChainState, MissingSpecColumnError, ProblemSpec, Solution
from portfolio_optimizer.domain.order_flow import OrderFlowProfile

if TYPE_CHECKING:
    from portfolio_optimizer.cvx.adapter import ConstraintSet, DecisionVars


class TrackingLimit(TypedConstraint):
    """``Σ |wᵢ − tᵢ| ≤ bound`` over the scope: total absolute deviation from a per-security target column."""

    kind: Literal["tracking_limit"] = "tracking_limit"
    column: str = Field(default="target", min_length=1, description="A per-security target-weight column of the spec.")
    bounds: ScalarBound = Field(description='The bound: a number, or a spec scalar (`{"scalar": "max_tracking"}`).')

    @model_validator(mode="after")
    def _bounds_from_above(self) -> Self:
        if not bounds_above(self.direction):
            msg = f"{self.name}: tracking_limit only bounds from above; direction must be '<='"
            raise ValueError(msg)
        return self

    @override
    def requirements(self, spec: ProblemSpec) -> Iterator[str]:
        yield from super().requirements(spec)
        try:
            spec.column(self.column)
            scalar_bound(self.bounds, spec)
        except MissingSpecColumnError as error:
            yield str(error)

    @override
    def residual(self, spec: ProblemSpec, solution: Solution, chain: ChainState, profile: OrderFlowProfile) -> list[tuple[str, F64]]:
        deviation = float((np.abs(solution.w - spec.column(self.column)) * self.scope_mask(spec)).sum())
        return [("tracking_limit", self._signed(np.array([deviation]), np.array([scalar_bound(self.bounds, spec)])))]

    @override
    def to_cvxpy(self, x: "DecisionVars", spec: ProblemSpec, chain: ChainState) -> "ConstraintSet":
        from portfolio_optimizer.cvx.adapter import ConstraintSet, at_most, masked, norm1, shifted

        deviation = norm1(masked(self.scope_mask(spec), shifted(x.w, spec.column(self.column))))
        return ConstraintSet(self.name, (at_most(deviation, scalar_bound(self.bounds, spec)),))
```

- `residual` returns named violation vectors, positive where breached beyond `tolerance`; `_signed`
  applies the row's `direction` and tolerance for you. A residual of length *n* names the worst
  security in the report; a length-one residual does not. The verifier reports each under
  `name/residual` and marks it `[binding]` when it sits within the tolerance of its bound.
- `scope_mask(spec)` is the row's scope as a mask (every security when unset); `_effective(bounds,
  current)` applies `allow_current_weight` to a `w`-shaped bound. The module's helpers —
  `scalar_bound`, `vector_bound`, `bound_requirements`, `starting_values`, `vector_values`,
  `adv_remaining` — are what the shipped kinds are written with; use them so a bound written as
  `{"scalar": ...}` or `{"column": ...}` works in yours too.
- The row's `kind`, `label`, and `params` become the model, so the shape a desk writes is the model's
  fields and nothing else.

## 3. Make the kind known

Every consumer — the resolver, the schedule, the shipped cvxpy step, the verifier, the JSON Schema —
reads one registry per kind, so a kind that is known anywhere is known everywhere. Three ways in:

- **In this repository**: add the class to `SHIPPED_TERM_KINDS` in `domain/objective.py` or
  `SHIPPED_CONSTRAINT_KINDS` in `domain/constraints.py`. `tests/test_conventions.py` checks that every
  shipped kind is registered under its own name.
- **In a package**: publish it as an entry point and install the package wherever tasks run — a worker
  without it cannot resolve the config (a term kind) or build any portfolio whose rows name it (a
  constraint kind), and says so.

  ```toml
  [project.entry-points."portfolio_optimizer.term"]
  diagonal_risk = "my_firm.terms:DiagonalRisk"

  [project.entry-points."portfolio_optimizer.constraint"]
  tracking_limit = "my_firm.constraints:TrackingLimit"
  ```

  `uv run portfolio-optimizer schema` in that environment emits a JSON Schema that includes the new
  term kind, and `steps` lists both.
- **In a notebook or a test**: `register_term_kind(DiagonalRisk)` or
  `register_constraint_kind(TrackingLimit)`, usable as decorators — what loading an entry point does,
  for the current process.

## 4. Name a term in the config, a constraint in the data

```json
"objective": [
  {"kind": "linear", "name": "alpha", "column": "alpha", "weight": "-1"},
  {"kind": "diagonal_risk", "name": "risk", "column": "variance", "weight": "2.5"}
]
```

Names must be unique among the run's terms. `validate-config` parses each record as its kind, renders
it once, and lists it as `term  risk (DiagonalRisk)`. A term naming a column the dummy spec lacks is
not refused there — whether the universe carries it is the data's business — and fails the portfolio
at stage `build` if the real spec lacks it too.

```csv
portfolio_id,kind,label,params
P1,tracking_limit,tracking,"{""direction"": ""<="", ""bounds"": {""scalar"": ""max_tracking""}}"
```

One row per account, so a limit that applies to one account is a row, not a config edit — which is the
point of loading them. `max_tracking` here is a column of that account's `details` row, exported as a
spec scalar; a literal `"0.05"` works too. Two rows of one kind for one account need different labels.

## 5. If the kind reads earlier portfolios' results

Set `reads_chain: ClassVar[bool] = True` and read `chain.traded_shares`: what higher-priority
portfolios *traded on the side the run couples through* (bought under `inflow`, sold under `outflow`), as
whole shares aligned to `spec.security_ids`, zero wherever this portfolio cannot trade the
name on that side, with `chain.predecessors` naming what was folded. `participation_limit` is the
worked example: `x.coupled` is the decision vector on that side and `profile.coupled(solution)` its
numpy twin, so a constraint written against them runs under every `order_flow`; `adv_remaining(spec, chain)`
is the budget the chain left. Nothing else reaches a later portfolio; a run has no other side.

The declaration is what the schedule is derived from. A chain-reading *constraint* couples its
portfolio through its `scope` alone (`scope ∩ tradable`), which is what lets a book that shares one
universe still solve as a graph rather than a line; a chain-reading *term* is opaque to the schedule
and couples every portfolio through its whole tradable set. Use the chain to *limit* the coupled side
and only that side — the engine's guarantee that the schedule never changes the answer rests on the
chain reaching nothing else.

## 6. Test

For a term, assert `value` against a hand-computed number and solve a two- or three-asset case whose
optimum you can compute by hand; `tests/engine/test_solve.py` shows the pattern. For a constraint,
perturb a feasible solution past the bound and assert `residual` reports it, then render `to_cvxpy`
in a solve and check the verifier passes the answer; `tests/domain/test_constraints.py` and
`tests/engine/test_check.py` are the models.

```bash
uv run pytest tests/domain/test_constraints.py tests/engine/test_solve.py tests/engine/test_check.py
```
