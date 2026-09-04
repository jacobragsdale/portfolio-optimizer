# How to replace the cvxpy solve with your own function or library

The engine builds each portfolio's problem as data, folds the chain, and hands *one* configured step
the job of deciding the weights. By default that step is `cvxpy`, which renders the typed terms and the
portfolio's typed constraint rows through their own `to_cvxpy` and solves the problem. This guide swaps
it for a step of your own: a plain numpy function for a side that needs no optimizer, or an adapter
over a library your firm already uses to build the problem. Everything around the step — the build,
the dependency graph, verification, rounding, the manifest — is unchanged, and the verifier is what
makes the step's answer trustworthy: it holds the weights to every typed constraint row the portfolio
carries, whether or not the step rendered it.

## Prerequisites

- A run config that works with the default step ([the tutorial](tutorial-first-run.md) gets you one).
- The function's package importable on the client and on every worker; a qualified step name is
  imported by name, not pickled, and must be in `PORTFOLIO_OPTIMIZER_STEP_PACKAGES` when that
  allowlist is set.

## 1. Write the function

A solve step takes a `SolveRequest` and returns a `SolveResult`. Everything it may use is on the
request: the `ProblemSpec` (every input the solver would see, as numpy arrays aligned to
`spec.security_ids`), the `ChainState` (what higher-priority portfolios traded on the side the run
couples through, masked to what this one can trade there), the order-flow profile, the typed terms, this
portfolio's constraint rows, and the run's extra datasets.

```python
import numpy as np

from portfolio_optimizer.domain.constraints import adv_remaining
from portfolio_optimizer.solving import SolveRequest, SolveResult


def buy_the_best_alpha(request: SolveRequest) -> SolveResult:
    """Spend the cash above the floor in proportion to each name's alpha, capped by its bound and the ADV budget left."""
    spec, chain = request.spec, request.chain
    budget = (1.0 - float(spec.w0.sum())) - spec.scalar("cash_lb")
    room = np.maximum(np.minimum(spec.ub - spec.w0, adv_remaining(spec, chain)), 0.0)
    want = np.maximum(spec.column("alpha"), 0.0)
    buy = np.minimum(room, budget * want / want.sum()) if want.sum() > 0 else np.zeros(spec.n)
    return SolveResult(w=spec.w0 + buy)
```

The rules of the contract:

- **Return weights and nothing else.** `w` must be aligned to `spec.security_ids`; the order-flow profile
  derives the trade from it. Leave `objective` unset unless you minimized one — then the verifier
  compares it with the terms' own `value`, as it does for cvxpy.
- **Raise to refuse.** A book the function cannot handle (cash below the floor, say) is an exception
  with a message; the engine records the portfolio as failed at stage `solve` with that message and
  does not try to explain it. `spec.scalar`, `spec.column`, and `adv_remaining` raise
  `MissingSpecColumnError` for a spec that lacks what they read (`adv_capacity` needs the universe's
  `adv_shares`); the shipped `pro_rata_fill` catches that one and fills without a budget.
- **Touch nothing outside the request.** No files, no clock, no other portfolio. A step that reads the
  chain gets it from `request.chain`; nothing else about other portfolios is visible, by design. Because
  the engine cannot see *whether* a step of yours reads the chain, every step other than the shipped
  one is opaque to the schedule: each portfolio couples through its whole tradable set.
- **Optional params.** Add `params: MyParams` (a `Params` subclass) to take settings from the config,
  exactly as a rule does. The shipped step's `solver`, `options`, `time_limit_s`, and `verbose` are
  its params, not the engine's; a step that wants a solver name declares its own field.

The shipped `pro_rata_fill` in `src/portfolio_optimizer/solvers.py` is this example finished — with
the cap's excess redistributed — and the shape to copy.

### What `SolveResult` carries

Every field but `w` has a default, so a pure function sets `w` and nothing else (`solving.py`):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `w` | `F64 \| None` | — | The weights, aligned to `spec.security_ids`; the order-flow profile derives the trade from them. `None` is only acceptable with a status that is not optimal. |
| `status` | `SolveStatus` | `OPTIMAL` | `optimal` or `optimal_inaccurate` are accepted and verified; `infeasible` raises `InfeasibleError` with an arithmetic diagnosis; `unbounded` raises `UnboundedError`; `solver_error` raises `SolverFailureError` carrying `detail`. |
| `objective` | `float \| None` | `None` | The value the step minimized, when it minimized one; the verifier compares it with the terms' recomputed sum. Leave it unset and the comparison is skipped. |
| `iterations` | `int \| None` | `None` | Recorded in the manifest's `solve` record. |
| `solve_time_s` | `float` | `0.0` | Recorded in the manifest's `solve` record. |
| `solver` | `str \| None` | `None` | What produced the answer; the engine records the step's qualified name when unset. |
| `solver_version` | `str \| None` | `None` | Its version; the engine records the step's package version when unset. |
| `detail` | `str` | `""` | Free text: the solver's own status, or what a function did (`pro_rata_fill` reports what it invested). Quoted in the failure message when the status is not optimal. |
| `duals` | `Mapping[str, float]` | `{}` | Per constraint name, the largest dual value the solver reported — the shadow price; recorded in the manifest's `solve.duals`. |

### Constraints are yours to interpret

`request.constraints` is a **DataFrame**: this portfolio's constraint rows exactly as its loader
returned them and its rules left them. The engine reads `portfolio_id` and, where the frame has one,
the `kind` column's declarations for the schedule — nothing else — so whatever vocabulary your desk
writes its constraints in arrives intact, and interpreting it is the solve step's job. It is empty when
the run declares no `constraints` dataset, which is the shape to handle if your step needs none.

```python
from portfolio_optimizer.domain.constraints import parse_constraints


def with_our_library(request: SolveRequest, params: OurParams) -> SolveResult:
    parsed = parse_constraints(request.constraints)  # typed rows as their models; None for a frame in another vocabulary
    limits = our_library.limits(request.constraints if parsed is None else parsed.typed)
    weights, value = our_library.solve(request.spec, limits, solver=params.solver)
    return SolveResult(w=weights, objective=value, solver=params.solver)
```

The step reports nothing about the constraints, because what is verified is not the step's to say. The
engine stamps the portfolio's typed rows on the solution itself and re-checks the weights through each
model's own `residual` — a kind a package publishes is checked exactly like a shipped one — so a step
that skips a row fails on that row rather than passing quietly. Rows in a vocabulary of your own are
yours to interpret and are not independently checked; if that matters, express them as typed rows.

The shipped `solvers.cvxpy` is the worked example: it parses the typed rows, renders each through
`to_cvxpy`, adds the order-flow profile's identity, and reports the solver's duals back. Replacing that
one function is how a desk brings its own syntax without touching the engine.

### Runtime parameters arrive the same way

`request.extras` is every extra dataset the run carried — any name the engine does not know — as the
rules left it, each frame reduced to this portfolio's rows where it has a `portfolio_id` column. It is
where a setting that is not a per-security column belongs: a risk aversion, a name count, a cash
buffer. Nothing shipped interprets it, which is the point — the engine loads, hashes and records these
frames and hands them over untouched.

```python
def with_our_library(request: SolveRequest, params: OurParams) -> SolveResult:
    settings = our_library.settings(request.extras["global_parameters"])  # a name/value frame, loaded and hashed like any input
    ...
```

Because they are loaded rather than configured, changing one changes the manifest's dataset hashes and
not the config hash, so `diff-manifests` reports "the data changed" — which is what a parameter change
is.

Set `solver` and `solver_version` on the result when your library can say what solved the problem;
otherwise the engine records the step's qualified name and its package version.

## 2. Name it in the config

```json
"solve": "mypkg.fills:buy_the_best_alpha"
```

or, with params, `{"name": "mypkg.fills:with_our_library", "params": {"solver": "CLARABEL"}}`. `solve`
is a step like any other: a bare name resolves in `solvers.py` or among the steps installed packages
publish in the group `portfolio_optimizer.solve`, a qualified name anywhere the engine can import. The
cvxpy solver's own settings go with the shipped step and leave with it.

## 3. Check it before a run

```bash
uv run portfolio-optimizer validate-config configs/my_run.json
```

The resolver imports the function, checks its signature against the contract (`request` annotated
`SolveRequest`, returning `SolveResult`), and validates its params. A signature that does not match is
reported with the other config failures:

```text
config rejected: 1 config resolution failure(s): solve: mypkg.fills:buy_the_best_alpha: missing required parameter 'request'
```

The step itself is not run here — the dummy problem `validate-config` renders terms against is not one
worth solving, and a firm's step may reach a service. Under a step that is not the shipped one the
terms are not dry-rendered either, and the objective may be empty.

## 4. Read the result of a run

A step that minimizes nothing shows up in the manifest with its qualified name as `solver`, its
package version, and no `objective_value`; the verifier's report still carries the configured terms'
values, so a heuristic can be compared with the optimizer on the same book, and `check.active` names
the limits the answer sits against. If the step's answer violates a constraint it reported, the
portfolio fails at stage `solve` with `VerificationError` naming the check — the same refusal a wrong
solver answer gets. That is the contract working: a step is free to compute the weights any way it
likes, and is held to the same limits as the solver.
