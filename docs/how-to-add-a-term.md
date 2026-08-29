# How to add an objective term or a constraint

Terms and constraints are what the optimizer minimizes and respects. This guide adds a term that reads a
signal column and a constraint that uses earlier portfolios' results, and keeps both auditable.

## Prerequisites

- The data the term needs is in the per-portfolio spec. `ProblemSpec` carries fixed arrays aligned to
  `spec.security_ids` (`w0`, `price`, `w_target`, `tax_per_dollar`, `tcost_per_dollar`, `lb`, `ub`,
  `adv_capacity`, sector matrix and bounds) plus every extra column of the universe frame, exported by
  name: numeric ones as float64 in `spec.column(name)`, boolean ones as `np.bool_` masks in
  `spec.flag(name)`.
- If the signal is not in the universe yet, attach it with an assembly step or a rule first (see
  [how to add security analytics](how-to-add-security-analytics.md)); the term then reads it with
  `spec.column("my_signal")`.

## 1. Write the function in `src/portfolio_optimizer/terms.py` — or in your package

A term or constraint in an installed package is named `my_firm.terms:signal_tilt` in the config. It runs
and is recorded like a shipped one, but has no numpy twin unless you add one here (step 3), so the
manifest lists it as `unverified`.

An objective term returns `ObjectiveTerm`; a constraint returns `ConstraintSet`. Both take
`x: DecisionVars` (`x.w`, `x.buy`, `x.sell` — weights and the non-negative trade split, as fractions of
NAV) and `spec: ProblemSpec`.

```python
class SignalParams(WeightedParams):
    column: str = "my_signal"


def signal_tilt(x: DecisionVars, spec: ProblemSpec, params: SignalParams) -> ObjectiveTerm:
    """``-weight · sᵀw``: reward exposure to the signal."""
    return ObjectiveTerm("signal_tilt", scale(float(params.weight), dot(-spec.column(params.column), x.w)))


def max_names_traded(x: DecisionVars, spec: ProblemSpec, params: TurnoverParams) -> ConstraintSet:
    """Two-way turnover per name capped at ``limit``."""
    return ConstraintSet("max_names_traded", (at_most(plus(x.buy, x.sell), float(params.limit)),))
```

Write the math through the atoms in `portfolio_optimizer.cvx.adapter` (`sum_squares`, `norm1`, `total`,
`dot`, `matvec`, `scale`, `plus`, `minus`, `shifted`, `equals`, `at_most`, `at_least`). They are the whole
cvxpy surface the template exposes, and every one preserves DCP convexity when the inputs do. A term that
is not convex is rejected when the problem is built, before the solver runs.

Read numbers from `spec` or from `params`; never from a file or a global. The spec is hashed into the
manifest, so anything the term used is recorded.

One modeling rule the engine relies on: **a term must never reward selling** — a negative cost per unit
of `x.sell` that no other term outweighs. After a solve the engine replaces the solver's buy/sell split
with the canonical one (`buy = max(w − w0, 0)`, `sell = max(w0 − w, 0)`) and the verifier recomputes
the objective on that split. With a term that pays for a round trip, the solver sells and rebuys the
same name, the canonical split removes the round trip, and the recomputed objective no longer matches
the solver's: the portfolio fails verification. The shipped `tax_cost` refuses to run with a loss-harvest
incentive and no transaction cost for exactly this reason, and a realistic transaction cost is rarely
enough. If the model needs a harvest reward, price the round trip explicitly with a constraint or a
term the verifier can mirror.

## 2. Name it in the config

```json
"objective": {"terms": ["tracking_error", {"name": "signal_tilt", "params": {"weight": "0.5", "column": "momentum"}}]},
"constraints": ["long_only", "max_weight", "cash_bounds", {"name": "max_names_traded", "params": {"limit": "0.02"}}]
```

The engine minimizes the sum of the terms. What `buy` and `sell` mean is not a constraint you list: the
run's `sides` supplies the trade identity to every solve.

## 3. Give the verifier a twin, or accept "unverified"

After every solve the engine recomputes each shipped term and constraint in plain numpy
(`src/portfolio_optimizer/engine/check.py`) and compares with the solver. A custom step has no twin, so
the manifest lists it under `unverified` and the objective comparison is skipped. To keep the run fully
verifiable, add a numpy twin keyed by the step's qualified name:

```python
def _signal_tilt(spec: ProblemSpec, sol: Solution, params: Mapping[str, object]) -> float:
    return -param(params, "weight", 1.0) * float((spec.column(str(params.get("column", "my_signal"))) * sol.w).sum())


TERM_TWINS = {**TERM_TWINS, "portfolio_optimizer.terms:signal_tilt": _signal_tilt}
```

A constraint twin returns `(name, residual_array)` pairs where a positive residual is a violation. The
test `test_every_shipped_term_and_constraint_has_a_twin` pins the shipped set; extend it with yours.

## 4. If the step needs earlier portfolios' results

Add `chain: ChainState`. It carries `bought_shares` per security — what higher-priority portfolios
*bought*, aligned to `spec.security_ids`, and zero wherever this portfolio cannot buy — plus the
`predecessors` it was folded from; `cumulative_adv_participation` is the worked example. Sells never
reach a later portfolio: portfolios couple through buys only. Declaring `chain` is what makes a
portfolio wait for every higher-priority portfolio that can buy a security it can buy too; with no
chain-aware step in the config, nothing waits. Use it to *limit* buys, and only buys — the engine's
guarantee that the schedule never changes the answer rests on the chain reaching nothing else.

## 5. Test

For a term, a two- or three-asset case whose optimum you can compute by hand is worth more than any
number of random ones; `tests/engine/test_solve.py` shows the pattern. For a constraint, additionally
perturb a feasible solution past the constraint and assert the verifier's twin reports it
(`tests/engine/test_check.py`).

```bash
uv run pytest tests/engine/test_solve.py tests/engine/test_check.py
```
