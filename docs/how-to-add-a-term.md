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
`x: DecisionVars` and `spec: ProblemSpec`. `x.w` is the target weight and `x.trade` the amount traded,
as fractions of NAV; `x.buy` and `x.sell` are the non-negative split, and exist only on a side the run
has — read `x.sell` in a buy-only run and `validate-config` refuses the config, naming the side. Write
"the amount traded" as `x.trade` and the term runs under every `sides`; write `x.buy` or `x.sell` only
when the term means that side specifically.

```python
class SignalParams(WeightedParams):
    column: str = "my_signal"


def signal_tilt(x: DecisionVars, spec: ProblemSpec, params: SignalParams) -> ObjectiveTerm:
    """``-weight · sᵀw``: reward exposure to the signal."""
    return ObjectiveTerm("signal_tilt", scale(float(params.weight), dot(-spec.column(params.column), x.w)))


def max_names_traded(x: DecisionVars, spec: ProblemSpec, params: TurnoverParams) -> ConstraintSet:
    """Turnover per name capped at ``limit``; two-way where the run has two sides."""
    return ConstraintSet("max_names_traded", (at_most(x.trade, float(params.limit)),))
```

Write the math through the atoms in `portfolio_optimizer.cvx.adapter` (`sum_squares`, `norm1`, `total`,
`dot`, `matvec`, `scale`, `plus`, `minus`, `shifted`, `shortfall`, `equals`, `at_most`, `at_least`). They are the whole
cvxpy surface the template exposes, and every one preserves DCP convexity when the inputs do. A term that
is not convex is rejected when the problem is built, before the solver runs.

Read numbers from `spec` or from `params`; never from a file or a global. The spec is hashed into the
manifest, so anything the term used is recorded.

One modeling rule a two-sided run relies on: **a term must never reward selling** — a negative cost per
unit of `x.sell` that no other term outweighs. After a solve the engine replaces the solver's buy/sell split
with the canonical one (`buy = max(w − w0, 0)`, `sell = max(w0 − w, 0)`) and the verifier recomputes
the objective on that split. With a term that pays for a round trip, the solver sells and rebuys the
same name, the canonical split removes the round trip, and the recomputed objective no longer matches
the solver's: the portfolio fails verification. The shipped `tax_cost` refuses to run with a loss-harvest
incentive and no transaction cost for exactly this reason, and a realistic transaction cost is rarely
enough. If the model needs a harvest reward, price the round trip explicitly with a constraint or a
term the verifier can mirror. A one-sided run has no such rule to keep: with one vector there is no
round trip to reward.

## 2. Name a term in the config, a constraint in the data

A term is config — the engine minimizes the sum of the objective's terms:

```json
"objective": {"terms": ["tracking_error", {"name": "signal_tilt", "params": {"weight": "0.5", "column": "momentum"}}]}
```

A **constraint is data**, one row per portfolio in the `constraints` dataset, under the convention the
shipped `cvxpy` solve step reads:

```csv
portfolio_id,name,label,params
P1,long_only,,
P1,max_weight,,
P1,max_names_traded,,"{""limit"": ""0.02""}"
```

So a constraint that applies to one account is one row, not a config edit — which is the point of
loading them. What `buy` and `sell` mean is not a constraint you write: the run's `sides` supplies the
trade identity to every solve, and a row naming `trade_balance` is refused. To use one constraint
function twice for a portfolio with different params, give each row a `label`, since the verifier's
report and the manifest key on it and labels must be unique within a portfolio.

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

A constraint twin has the signature `(spec, sol, chain, params, profile) -> list[tuple[str, F64]]`
(`ConstraintTwin` in `engine/check.py`): the spec, the `Solution` in place of `x`, the `ChainState`,
the params as a mapping, and the side profile that made `x` — `profile.coupled(sol)` is the numpy twin
of `x.coupled`. It returns `(name, residual_array)` pairs where a positive residual is a violation; the
verifier reports each under the constraint's label and holds it to `violation_tol`. Register it in
`CONSTRAINT_TWINS` by qualified name. The test `test_every_shipped_term_and_constraint_has_a_twin` pins
the shipped set; extend it with yours.

## 4. If the step needs earlier portfolios' results

Add `chain: ChainState`. It carries `traded_shares` per security — what higher-priority portfolios
*traded on the side the run couples through* (bought under `both` and `buy`, sold under `sell`),
aligned to `spec.security_ids`, and zero wherever this portfolio cannot trade the name on that side —
plus the `predecessors` it was folded from; `cumulative_adv_participation` is the worked example, and
`x.coupled` is the decision vector on that side, so a constraint written as `at_most(x.coupled, ...)`
runs under every `sides`. Nothing else reaches a later portfolio: a two-sided run's sells reach no
one. Declaring `chain` is what makes a portfolio wait for every higher-priority portfolio whose
tradable set overlaps its own; with no chain-aware step in the config, nothing waits. Use it to *limit*
the coupled side, and only that side — the engine's guarantee that the schedule never changes the
answer rests on the chain reaching nothing else.

## 5. Test

For a term, a two- or three-asset case whose optimum you can compute by hand is worth more than any
number of random ones; `tests/engine/test_solve.py` shows the pattern. For a constraint, additionally
perturb a feasible solution past the constraint and assert the verifier's twin reports it
(`tests/engine/test_check.py`).

```bash
uv run pytest tests/engine/test_solve.py tests/engine/test_check.py
```
