# portfolio-optimizer

A template repository for a JSON-driven, auditable portfolio-optimization engine built on pandas and cvxpy.
Clone it (or click **Use this template**), keep the engine, and write your own loaders, rules, objective
terms, constraints, and sinks as ordinary Python functions.

A run is one JSON file. The engine loads the data it names, assembles it, applies your rules per
portfolio, solves one problem per portfolio, re-verifies every answer without cvxpy, rounds to
whole-share orders, publishes them, and writes a manifest that lets anyone reproduce and audit the run.

## The run config, block by block

The quickest way to see what the engine does is to read the run it ships with,
[`configs/example_run.json`](configs/example_run.json):

```json
{
  "$schema": "./run-config.schema.json",
  "run": {"name": "example_rebalance", "as_of": "2026-08-28T00:00:00Z", "tags": {"desk": "template"}},
  "portfolios": {"name": "csv", "params": {"path": "portfolios.csv"}},
  "datasets": {
    "holdings": {"loader": {"name": "csv_per_portfolio", "params": {"directory": "holdings"}}, "scope": "per_portfolio", "batch_size": 1},
    "universe": {"loader": {"name": "csv", "params": {"path": "universe.csv"}}},
    "details": {"loader": {"name": "csv_per_portfolio", "params": {"directory": "details"}}, "scope": "per_portfolio", "batch_size": 1},
    "constraints": {"loader": {"name": "json_constraints", "params": {"path": "constraints.json"}}},
    "targets": {"loader": {"name": "csv", "params": {"path": "targets.csv"}}},
    "prices": {"loader": {"name": "csv", "params": {"path": "prices.csv", "dtypes": {"security_id": "string", "price": "decimal"}}}}
  },
  "assembly": [
    {"name": "join", "params": {"into": "universe", "source": "prices", "on": ["security_id"], "cardinality": "one_to_one", "require_all_matched": true}},
    {"name": "drop", "params": {"datasets": ["prices"]}}
  ],
  "rules": [{"name": "restrict_low_liquidity", "params": {"min_adv_shares": 1000}}, "add_zero_alpha", "attach_universe_columns"],
  "objective": {
    "sense": "minimize",
    "terms": [
      {"name": "tracking_error", "params": {"weight": "1.0"}},
      {"name": "tax_cost", "params": {"weight": "1.0"}},
      {"name": "transaction_cost", "params": {"weight": "1.0"}}
    ]
  },
  "constraints": ["long_only", "max_weight", "cash_bounds", "turnover_cap", "sector_bounds", "cumulative_adv_participation"],
  "solver": {"name": "CLARABEL", "options": {"max_iter": 200}, "time_limit_s": 60.0, "verbose": false},
  "post_solve": {"violation_tol": 1e-6, "objective_rel_tol": 1e-5, "objective_abs_tol": 1e-9},
  "sink": {"name": "orders_to_parquet", "params": {"subdir": "orders"}},
  "execution": {"on_error": "fail_fast"}
}
```

Most values are *steps*: a function named either by a bare name, looked up in the template module for
that kind of step, or by `package.module:function`, with optional `params` (see
[the one convention](#the-one-convention)). Top to bottom:

- **`$schema`** — for your editor, not the engine: live validation and completion of key and step names
  from the generated schema. The engine ignores it.
- **`run`** — the run's identity. `name` and `tags` go into the manifest; `as_of` is the timezone-aware
  instant the run is *as of* — every loader receives it, and it decides whether each tax lot is long- or
  short-term.
- **`portfolios`** — the one loader that runs first and alone. It returns the portfolio ids and,
  optionally, a `solve_order` priority; every other loader is told which ids to fetch.
- **`datasets`** — everything else to load, all at once. Five names belong to the engine and are
  validated against fixed schemas: `holdings`, `universe`, `details`, `targets`, and `constraints` (each
  portfolio's style limits). Any other name — `prices` here — is an extra dataset the engine does not
  interpret: assembly steps see it, and whatever survives assembly reaches each portfolio's rules as
  `data.extras`. Each entry also says how its loader is called. `universe`, `targets`, `prices`, and
  `constraints` say nothing and are `global`: one call for the whole book, and the only datasets
  assembly sees. `holdings` and `details` are `per_portfolio` with `batch_size: 1`, so the engine calls
  their loaders once per account — a custodian and an account master answer one portfolio at a time —
  which is also why a portfolio whose own inputs are missing fails alone instead of stopping the run.
  An input from a throttled source adds a `rate_limit`.
- **`assembly`** — steps that run once over all loaded datasets to make the tables the build expects.
  Here: join prices into the universe, checking every security matched exactly once, then drop the price
  file. This is where per-security analytics get attached to `holdings` and `universe`.
- **`rules`** — business logic, run per portfolio in order: each takes one portfolio's validated bundle
  and returns a modified one, and never sees other portfolios. The example restricts illiquid names,
  adds a zero `alpha` column, and copies the universe's analytics columns onto holdings — the
  attachment an assembly `join` would do if holdings were global.
- **`objective`** — the sum of the listed terms, always minimized; a reward is a negative term. Each term
  is a function returning a convex expression, and its `weight` is a string so the manifest records an
  exact `Decimal`.
- **`constraints`** — *which* constraints apply; *how tight* they are comes from the data (the style
  object in the `constraints` dataset, and per-security columns in the universe). The last one is
  chain-aware — it reads what higher-priority portfolios have already traded — and that is the only
  reason one portfolio ever waits for another.
- **`solver`** — the cvxpy solver and its options. Its presence is checked when the config resolves, its
  version is recorded in the manifest, and there is no silent fallback to another solver.
- **`post_solve`** — tolerances for the verifier that re-checks every solution in plain numpy without
  cvxpy: each constraint's residual and the gap between the recomputed and reported objective.
- **`sink`** — where the orders go, called once with every solved portfolio's orders.
- **`execution`** — what one failed portfolio does to the rest: `fail_fast` or `continue`. *Where* the
  work runs and how many workers there are is an environment setting, so a laptop run and a cluster run
  of one config hash identically.

Four keys the example leaves at their defaults: `sides` (`both`; `buy` or `sell` runs a one-sided
problem a third the size), `solve` (`cvxpy`; a heuristic or your own library can replace it),
`solve_order` (a step that computes each portfolio's priority from the data instead of the column), and
`rate_limits` (named pools shared by inputs on one backend).

Numbers — positions, prices, caps, bands, tax rates — are never in the config; they live in the data.
Behavior is never in the config either; it lives in the functions the config names.
[Reading a run config](docs/explanation-run-config.md) explains each block in depth, and
[the reference](docs/reference-run-config.md) lists every key with its type and default.

## Quick start

```bash
uv sync --locked
cp .env.example .env
uv run --env-file .env portfolio-optimizer validate-config configs/example_run.json
uv run --env-file .env portfolio-optimizer run configs/example_run.json
```

The example runs two portfolios over three securities with a hand-checkable optimum; see
[the tutorial](docs/tutorial-first-run.md) for what to expect at each step.

## The one convention

A step is an ordinary function in a designated module; the JSON run config names it.

```python
# src/portfolio_optimizer/rules.py
class CapSingleNameParams(Params):
    max_weight: Decimal = Field(gt=0, le=1)


def cap_single_name(data: PortfolioData, params: CapSingleNameParams) -> PortfolioData: ...
```

```json
"rules": [{"name": "cap_single_name", "params": {"max_weight": "0.05"}}]
```

No decorators, no registries. The function may live in the template modules or in any package installed
in the environment: name it `my_firm.rules:cap_single_name` and the engine imports it from there — which
is how a firm shares loaders, rules, and sinks across desks, with the package's version recorded in every
manifest. Before any data loads, the engine imports every named function, checks its
signature, validates its `params` against the function's own model, and records its source hash in the
run manifest. Loaders, assembly steps, solve-order steps, objective terms, constraints, solve steps, and
sinks follow the same rule.

Run configs are validated three ways: live in your editor through `"$schema": "./run-config.schema.json"`,
by `portfolio-optimizer validate-config` (which also imports and checks every step), and by any JSON Schema
validator against [`configs/run-config.schema.json`](configs/run-config.schema.json).

## Layout

| Path | Role |
|---|---|
| `src/portfolio_optimizer/{loaders,assembly,rules,solve_order,terms,solvers,sinks}.py` | **Yours to edit.** Each ships worked, tested examples. Shared steps live in your own installed package instead and are named `package.module:function`. |
| `src/portfolio_optimizer/engine/` | Loading and assembly, the rule pipeline, build, the solve stage, cvxpy-free verification, orders, the per-portfolio tasks, the derived dependency graph, the Dask cluster each run provisions for itself, manifest. Rarely edited. |
| `src/portfolio_optimizer/domain/` | Frame schemas, the per-portfolio data bundle and its optimizer frame, the pure-data problem spec and results, and `sides.py`: the side profiles — what a buy-only, sell-only, or two-sided run means, in numpy. |
| `src/portfolio_optimizer/config/` | The run-config models and the step resolver. |
| `src/portfolio_optimizer/cvx/` | `adapter.py`, the only module that imports cvxpy, and `sides.py`, the side profiles' cvxpy half: each side's decision variables and trade identity. |
| `src/portfolio_optimizer/solving.py` | The solve step's contract: `SolveRequest` in, `SolveResult` out. |
| `src/portfolio_optimizer/ratelimit.py` | Rate-limit pools loaders draw from, and `fan_out` for sources that answer one portfolio per call. |
| `configs/example_run.json`, `configs/run-config.schema.json`, `examples/data/` | The shipped example and the generated JSON Schema. |
| `benchmarks/profile_portfolio.py` | Times one portfolio through the pipeline stage by stage at a chosen book size and side; the numbers in `IDEAS.md` come from it. |
| `docs/` | Tutorial, how-to guides, reference, and explanation. |
| `IDEAS.md` | Threads that are not yet decisions, and known defects waiting to be fixed. |

## Documentation

- [Tutorial: your first run](docs/tutorial-first-run.md)
- [Explanation: reading a run config](docs/explanation-run-config.md) — the walkthrough above, in depth
- [How to add a rule](docs/how-to-add-a-rule.md)
- [How to add a loader or a sink](docs/how-to-add-a-loader-or-sink.md)
- [How to add security analytics columns to holdings and the universe](docs/how-to-add-security-analytics.md)
- [How to add an objective term or a constraint](docs/how-to-add-a-term.md)
- [How to set the solve order](docs/how-to-set-the-solve-order.md)
- [How to run one side: a buy-only or sell-only run](docs/how-to-run-one-side.md)
- [How to replace the cvxpy solve with your own function or library](docs/how-to-write-a-solve-step.md)
- [How to run on a cluster](docs/how-to-run-on-a-cluster.md)
- [Reference: the run config](docs/reference-run-config.md)
- [Reference: the per-portfolio bundle and the optimizer frame](docs/reference-portfolio-data.md)
- [Reference: outputs, the manifest, and the CLI](docs/reference-manifest.md)
- [Explanation: how the engine is built and why](docs/explanation-architecture.md)
- [Explanation: the life of a run](docs/explanation-run-lifecycle.md)

## Development

```bash
uv sync --locked
uv run pre-commit run --all-files   # ruff format, ruff check --fix, ty, uv lock, prettier for JSON
uv run pytest                       # unit, property, and smoke tests; warnings are errors
```

Every run provisions its own Dask cluster and tears it down when it ends: `PORTFOLIO_OPTIMIZER_CLUSTER=local`
on a laptop needs nothing beyond the locked environment; on Kubernetes the `kubernetes` extra
(`uv sync --locked --extra kubernetes`) and the Dask operator. See [how to run on a cluster](docs/how-to-run-on-a-cluster.md).

Python 3.12 or newer. Dependencies are locked in `uv.lock`; solver versions are recorded in every manifest.
