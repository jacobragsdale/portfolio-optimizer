# How to replace the cvxpy solve with your own function or library

The engine builds each portfolio's problem as data, folds the chain, and hands *one* configured step
the job of deciding the weights. By default that step is `cvxpy`, which builds and solves a cvxpy
problem from the configured terms and constraints. This guide swaps it for a step of your own: a plain
numpy function for a side that needs no optimizer, or an adapter over a library your firm already uses
to build the problem. Everything around the step — the build, the dependency graph, verification,
rounding, the manifest — is unchanged, and the verifier is what makes the step's answer trustworthy.

## Prerequisites

- A run config that works with the default step ([the tutorial](tutorial-first-run.md) gets you one).
- The function's package importable on the client and on every worker; a qualified step name is
  imported by name, not pickled.

## 1. Write the function

A solve step takes a `SolveRequest` and returns a `SolveResult`. Everything it may use is on the
request: the `ProblemSpec` (every input the solver would see, as numpy arrays aligned to
`spec.security_ids`), the `ChainState` (what higher-priority portfolios traded on the side the run
couples through, masked to what this one can trade there), the side profile, the resolved terms, this
portfolio's constraint rows, and the `solver` block.

```python
import numpy as np

from portfolio_optimizer.solving import SolveRequest, SolveResult
from portfolio_optimizer.terms import adv_remaining


def buy_the_underweights(request: SolveRequest) -> SolveResult:
    """Spend the cash above the floor on the names furthest below target, capped by the ADV budget left."""
    spec, chain = request.spec, request.chain
    budget = (1.0 - float(spec.w0.sum())) - spec.cash_lb
    room = np.maximum(np.minimum(spec.ub - spec.w0, adv_remaining(spec, chain)), 0.0)
    want = np.maximum(spec.w_target - spec.w0, 0.0)
    buy = np.minimum(room, budget * want / want.sum()) if want.sum() > 0 else np.zeros(spec.n)
    return SolveResult(w=spec.w0 + buy)
```

The rules of the contract:

- **Return weights and nothing else.** `w` must be aligned to `spec.security_ids`; the side profile
  derives the trade from it. Leave `objective` unset unless you minimized one — then the verifier
  compares it with the terms' numpy twins, as it does for cvxpy.
- **Raise to refuse.** A book the function cannot handle (cash below the floor, say) is an exception
  with a message; the engine records the portfolio as failed at stage `solve` with that message and
  does not try to explain it.
- **Touch nothing outside the request.** No files, no clock, no other portfolio. A step that reads the
  chain gets it from `request.chain`; nothing else about other portfolios is visible, by design.
- **Optional params.** Add `params: MyParams` (a `Params` subclass) to take settings from the config,
  exactly as a rule or a term does.

The shipped `pro_rata_fill` in `src/portfolio_optimizer/solvers.py` is this example finished — with
the cap's excess redistributed — and the shape to copy.

### What `SolveResult` carries

Every field but `w` has a default, so a pure function sets `w` and nothing else (`solving.py`):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `w` | `F64 \| None` | — | The weights, aligned to `spec.security_ids`; the side profile derives the trade from them. `None` is only acceptable with a status that is not optimal. |
| `status` | `SolveStatus` | `OPTIMAL` | `optimal` or `optimal_inaccurate` are accepted and verified; `infeasible` raises `InfeasibleError` with an arithmetic diagnosis; `unbounded` raises `UnboundedError`; `solver_error` raises `SolverFailureError` carrying `detail`. |
| `objective` | `float \| None` | `None` | The value the step minimized, when it minimized one; the verifier compares it with the terms' numpy twins. Leave it unset and the comparison is skipped. |
| `iterations` | `int \| None` | `None` | Recorded in the manifest's `solve` record. |
| `solve_time_s` | `float` | `0.0` | Recorded in the manifest's `solve` record. |
| `solver` | `str \| None` | `None` | What produced the answer; the engine records the step's qualified name when unset. |
| `solver_version` | `str \| None` | `None` | Its version; the engine records the step's package version when unset. |
| `detail` | `str` | `""` | Free text: the solver's own status, or what a function did (`pro_rata_fill` reports what it invested). Quoted in the failure message when the status is not optimal. |

### Constraints are yours to interpret

`request.constraints` is a **DataFrame**: this portfolio's constraint rows exactly as its loader
returned them and its rules left them. The engine reads one column, `portfolio_id`, and nothing else —
so whatever vocabulary your desk writes its constraints in arrives intact, and interpreting it is the
solve step's job. It is empty when the run declares no `constraints` dataset, which is the shape to
handle if your step needs none.

```python
def with_our_library(request: SolveRequest) -> SolveResult:
    limits = our_library.parse(request.constraints)  # your columns, your syntax
    weights, value = our_library.solve(request.spec, limits, solver=request.solver.name, options=request.solver.options)
    return SolveResult(w=weights, objective=value, solver=request.solver.name, constraints=our_library.applied(limits))
```

Report what you applied on `SolveResult.constraints`, a tuple of `StepRef` (`qualname`, JSON-safe
`params`, `label`). The verifier re-checks every ref it has a numpy twin for — every shipped
constraint has one — and reports the rest as `unverified` in the manifest rather than passing them
silently; the identity and solution checks run either way. Leave it empty and none of your constraints
are independently checked, which is honest but worth choosing deliberately.

The shipped `solvers.cvxpy` is the worked example: `interpret_constraints` reads the template's
`name`/`label`/`params` convention and resolves each row to a function in `terms.py`. Replacing that
one function is how a desk brings its own syntax without touching the engine.

Set `solver` and `solver_version` on the result when your library can say what solved the problem;
otherwise the engine records the step's qualified name and its package version.

## 2. Name it in the config

```json
"solve": "mypkg.fills:buy_the_underweights",
"solver": {"name": "CLARABEL"}
```

`solve` is a step like any other: a bare name resolves in `solvers.py`, a qualified name anywhere the
engine can import. The `solver` block stays; it is cvxpy's options and a step that is not cvxpy may
ignore it.

## 3. Check it before a run

```bash
uv run portfolio-optimizer validate-config configs/my_run.json
```

The resolver imports the function, checks its signature against the contract (`request` annotated
`SolveRequest`, returning `SolveResult`), and validates its params. A signature that does not match is
reported with the other config failures:

```text
config rejected: 1 config resolution failure(s): solve: mypkg.fills:buy_the_underweights: missing required parameter 'request'
```

The step itself is not run here — the dummy problem `validate-config` constructs terms against is not
one worth solving, and a firm's step may reach a service.

## 4. Read the result of a run

A step that minimizes nothing shows up in the manifest with its qualified name as `solver`, its
package version, and no `objective_value`; the verifier's report still carries the configured terms'
values, so a heuristic can be compared with the optimizer on the same book. If the step's answer
violates a configured constraint, the portfolio fails at stage `solve` with `VerificationError` naming
the check — the same refusal a wrong solver answer gets. That is the contract working: a step is free
to compute the weights any way it likes, and is held to the same limits as the solver.
