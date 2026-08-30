# portfolio-optimizer

A template repository for a JSON-driven, auditable portfolio-optimization engine built on pandas and cvxpy.
Clone it (or click **Use this template**), keep the engine, and write your own loaders, rules, objective
terms, constraints, and sinks as ordinary Python functions.

A run is one JSON file. The engine loads the data it names, assembles it, applies your rules per
portfolio, solves one problem per portfolio, re-verifies every answer without cvxpy, rounds to
whole-share orders, publishes them, and writes a manifest that lets anyone reproduce and audit the run.

## The run config, block by block

The quickest way to see what the engine does is to read the run it ships with,
[`configs/example_run.json`](configs/example_run.json), annotated here:

```jsonc
// The run that ships with the template, annotated. The real file is strict JSON with no
// comments — these lines are stripped and the rest compared against it by a test, so this
// copy cannot drift.
{
  // `as_of_date` is the only field here that changes results: every loader gets it, and it
  // decides whether a tax lot is long- or short-term. It must carry a zone.
  "run": {
    "name": "example_rebalance",
    "as_of_date": "2026-08-28T00:00:00Z",
    "tags": {"desk": "template"}
  },

  // Every input the run loads. Each dataset starts the moment the datasets its `depends_on` names
  // have loaded — with no dependencies, immediately — so the stage costs its longest chain rather
  // than its sum. Each loader stands in for a service — a custodian, a security master, an account
  // master — and waits as long as that source would; nothing here is a file read in a real run.
  "datasets": {
    // The book of record: the portfolio ids and their solve priorities. A dataset like any other —
    // only the inputs that ask for its ids wait on it. A fixed book can skip the loader and be
    // written inline: "portfolios": ["P1", "P2"].
    "portfolios": {"loader": "load_portfolios"},

    // `per_portfolio` is the engine's fan-out: one call per batch of accounts, `batch_size: 1`
    // being a call each — the shape of a custodian that answers one at a time — and it implies
    // `depends_on: ["portfolios"]`. A hundred calls at once is more than a vendor allows, so this
    // input names a pool.
    "holdings": {
      "loader": "load_holdings",
      "scope": "per_portfolio",
      "batch_size": 1,
      "rate_limit": "custodian"
    },

    // No `scope` means `global`: one call for the book, and the only datasets assembly sees. No
    // `depends_on` means it starts at once — this one is the run's long pole, and it no longer
    // waits for the book of record to answer first.
    "universe": {"loader": "load_universe"},

    // The account master: NAV, cash, tax rates, style limits. `batch_size: 25` hands the loader
    // twenty-five ids per call — the shape of a source that takes a list — and the bound is inline
    // because the firm's own database is nobody else's budget to share.
    "details": {
      "loader": "load_details",
      "scope": "per_portfolio",
      "batch_size": 25,
      "rate_limit": {"max_in_flight": 4}
    },

    // Which constraints bind each account and how tight they are. The engine reads only
    // `portfolio_id`; the solve step interprets every other column. `depends_on` hands the loader
    // the book's ids as `request.portfolio_ids`, so compliance is asked about the book, not the firm.
    "constraints": {"loader": "load_constraints", "depends_on": ["portfolios"]},

    // A name the engine does not know is an extra: carried untouched to the rules and on to the
    // solve step, which is where runtime parameters belong. One loader, two sets, each named by
    // its dataset.
    "global_parameters": {"loader": "load_parameters"},

    // The same shape, read earlier: `restrict_low_liquidity` takes its `min_adv_shares` here.
    "buy_universe_parameters": {"loader": "load_parameters"}
  },

  // Named pools inputs share. `holdings` is the only one here that names this pool, but a second
  // input on the same vendor would draw from the same budget rather than a second one.
  "rate_limits": {"custodian": {"requests_per_second": 20, "burst": 20, "max_in_flight": 8}},

  // Applied in order to each portfolio's bundle; a rule never sees another portfolio.
  "rules": ["restrict_low_liquidity"],

  // The sum is minimized, so a reward is a negative term — `alpha` is one. Weights are
  // strings so the manifest records an exact Decimal.
  "objective": {
    "sense": "minimize",
    "terms": [
      {"name": "alpha", "params": {"weight": "1.0"}},
      {"name": "tax_cost", "params": {"weight": "1.0"}},
      {"name": "transaction_cost", "params": {"weight": "1.0"}}
    ]
  },

  // Checked when the config resolves and on every worker; no fallback to another solver.
  "solver": {
    "name": "CLARABEL",
    "options": {"max_iter": 200},
    "time_limit_s": 60.0,
    "verbose": false
  },

  // Tolerances for the independent, cvxpy-free re-verification of every solution.
  "post_solve": {"violation_tol": 1e-6, "objective_rel_tol": 1e-5, "objective_abs_tol": 1e-9},

  "sink": {"name": "orders_to_parquet", "params": {"subdir": "orders"}},

  // `fail_fast` stops at the first failure; `continue` isolates it. `dependencies` says
  // whether portfolios wait for each other — declared, since constraints are opaque data.
  "execution": {"on_error": "fail_fast", "dependencies": "overlap"}
}
```

Most values are *steps*: a function named either by a bare name, looked up in the template module for
that kind of step, or by `package.module:function`, with optional `params` (see
[the one convention](#the-one-convention)). Top to bottom:

- **`run`** — the run's identity. `name` and `tags` go into the manifest; `as_of_date` is the timezone-aware
  instant the run is *as of* — every loader receives it, and it decides whether each tax lot is long- or
  short-term.
- **`datasets`** — everything the run loads, every one a frame, scheduled as the dependency DAG the
  entries declare: a dataset starts the moment the datasets its `depends_on` names have loaded, and
  one with no dependencies starts immediately, so the stage costs its longest chain rather than its
  sum. `portfolios` is the required first fact — the portfolio ids and, optionally, a `solve_order`
  priority — loaded like any dataset or written inline as a list of ids (the written order is the
  solve order); an input that wants the book's ids as `request.portfolio_ids` names it in
  `depends_on`, and a dependency's frames reach the loader as `request.inputs`. Three more names are required and
  validated against fixed schemas: `holdings`, `universe`, and `details` (the account's facts *and* its
  style limits); `constraints` is engine-known but optional. Any other name is an extra dataset
  the engine does not interpret: assembly steps see it, and whatever survives assembly reaches each
  portfolio's rules as `data.extras`. Each entry also says how its loader is called.
  `universe` and `constraints` say nothing and are `global`: one call for the
  whole book, and the only datasets assembly sees. `holdings` and `details` are `per_portfolio`, so the engine calls their loaders per
  account rather than once for the book, and `batch_size` says how finely: `1` is a call per account,
  the shape of a custodian that answers one at a time; `25` hands the loader twenty-five ids per call,
  the shape of an account master that takes a list. It is also why a portfolio whose own inputs are
  missing fails alone instead of stopping the run. An input whose source will not take a hundred calls
  at once adds a `rate_limit`: a named pool it shares with the other inputs on that backend, or a bound
  of its own.
- **`rules`** — business logic, run per portfolio in order: each takes one portfolio's validated bundle
  and returns a modified one, and never sees other portfolios. The example runs one, freezing names too
  illiquid to trade at a threshold it reads from `buy_universe_parameters` rather than from the config;
  `rules.py` also ships a rule that copies the universe's analytics columns onto holdings — the
  attachment an assembly `join` would do if holdings were global.
- **`objective`** — the sum of the listed terms, always minimized; a reward is a negative term. Each term
  is a function returning a convex expression, and its `weight` is a string so the manifest records an
  exact `Decimal`.
- **`constraints`** — *which* constraints apply and *how tight* they are, both from the data: a row per
  constraint per account, its numbers either on the row itself (a sector band's `lower` and `upper`) or
  in the style limits on the account's `details` row and the per-security columns of the universe. One
  of them is chain-aware — it reads what higher-priority portfolios have already traded — and that is
  the only reason one portfolio ever waits for another.
- **`solver`** — the cvxpy solver and its options. Its presence is checked when the config resolves, its
  version is recorded in the manifest, and there is no silent fallback to another solver.
- **`post_solve`** — tolerances for the verifier that re-checks every solution in plain numpy without
  cvxpy: each constraint's residual and the gap between the recomputed and reported objective.
- **`sink`** — where the orders go, called once with every solved portfolio's orders.
- **`execution`** — what one failed portfolio does to the rest: `fail_fast` or `continue`. *Where* the
  work runs and how many workers there are is an environment setting, so a laptop run and a cluster run
  of one config hash identically.

Four keys the example leaves at their defaults: `assembly` (steps that reshape the loaded datasets
before the engine-known frames are validated — a `join` that attaches per-security analytics to
`universe`, a `drop` for the dataset that supplied them), `sides` (`both`; `buy` or `sell` runs a
one-sided problem a third the size), `solve` (`cvxpy`; a heuristic or your own library can replace it),
and `solve_order` (a step that computes each portfolio's priority from the data instead of the column).

Numbers — positions, prices, caps, bands, tax rates, a liquidity threshold — are never in the config;
they live in the data, including the run's own parameters.
Behavior is never in the config either; it lives in the functions the config names.
[Reading a run config](docs/explanation-run-config.md) explains each block in depth, and
[the reference](docs/reference-run-config.md) lists every key with its type and default.

## The solve schedule is a derived graph

The state of the art in production rebalancing is parallel builds and a *sequential* solve, because a
chain-aware constraint seems to force one: when account *j* may only take what is left of a name's
daily volume after the accounts ahead of it took theirs, the safe reading of that dependency is a
line — *N* solves end to end. The line is the worst case, not the truth. Two accounts that cannot
trade a security in common cannot affect each other's feasible set, whatever their constraints read.
So the engine derives the real object: every portfolio builds in parallel and reports its tradable
set, and a portfolio waits only for the higher-priority portfolios whose tradable set intersects its
own. Wall clock is the graph's critical path times a solve, not *N* times a solve, independent
components solve at once, and the head of the book is solving while the tail is still building.

![Eight accounts: the line a chain-aware book seems to force, and the graph the engine derives from mandate overlap — three components, critical path three, same orders either way](docs/images/derived-schedule.svg)

What makes that a result rather than a claim: `execution.dependencies: "all"` runs the same book as
the strict line, and produces **byte-identical orders and chain hashes** — a property test asserts
it, and `benchmarks/run_book.py` runs both schedules over synthetic books on a real cluster.
Measured 2026-08-30 (8 local workers, Clarabel; the manifest's own `schedule` and `timing` blocks):

| Book | Derived: edges · components · critical path | Wall clock |
|---|---|---:|
| 100 accounts, 10 disjoint mandates, 2,000 names (~0.14s solves) | 450 · 10 · 10 | **6.9s** |
| — the same book as the line (`dependencies: all`) | 4,950 · 1 · 100 | 14.9s |
| 12 accounts, 4 disjoint mandates, 30,000 names (~2s solves) | 12 · 4 · 3 | **14.0s** |
| — the same book as the line | 66 · 1 · 12 | 24.6s |
| 1,000 accounts, 10 disjoint mandates | 49,500 · 10 · 100 | 34.1s — capacity-bound, 6.1× parallel on 8 workers |

The honest counterweight: overlap is on *any* shared tradable name, so the win belongs to books
partitioned by mandate, universe, or restriction list (`restrict_to_mandate` is the shipped shape).
A book whose accounts all buy from one universe is a complete DAG — the shipped example is
deliberately that case, and its own manifest says so: `edges 4950, critical_path 100` — and a single
sector shared between neighbouring mandates is enough to chain the book back into a line (1,450
edges, critical path 100 again). Narrowing coupling from "any shared name" to the names a constraint
can actually bind on is the open thread in `IDEAS.md`. How the graph is derived and why it is exact
is in [the architecture explanation](docs/explanation-architecture.md#a-run-couples-through-its-one-side-so-the-schedule-is-a-graph);
`portfolio-optimizer timeline` draws where any run's wall clock went.

## Quick start

```bash
uv sync --locked
cp .env.example .env
uv run --env-file .env portfolio-optimizer validate-config configs/example_run.json
uv run --env-file .env portfolio-optimizer run configs/example_run.json
```

The example runs a hundred accounts over three securities, and takes about half a minute of which
almost all is the shipped loaders pretending to be the services they stand in for. Its first two
accounts have a hand-checkable optimum; see [the tutorial](docs/tutorial-first-run.md) for what to
expect at each step.

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

Run configs are validated by `portfolio-optimizer validate-config`, which imports and checks every step,
and by any JSON Schema validator against [`configs/run-config.schema.json`](configs/run-config.schema.json).
Adding `"$schema": "./run-config.schema.json"` to a config gets the same completion and validation live in
an editor; the engine accepts the key and ignores it.

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
| `configs/example_run.json`, `configs/run-config.schema.json`, `examples/data/` | The shipped example — a hundred accounts over three securities, one CSV table per source — and the generated JSON Schema. |
| `benchmarks/` | `profile_portfolio.py` times one portfolio through the pipeline stage by stage at a chosen book size and side; `run_book.py` runs a synthetic book of *N* portfolios on a local cluster and reports the derived schedule and the timing spans. The numbers in `IDEAS.md` come from them. |
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
uv run pre-commit run --all-files   # ruff format and check, ty, uv lock, prettier for JSON
uv run pytest                       # unit, property, and smoke tests; warnings are errors
```

Every run provisions its own Dask cluster and tears it down when it ends: `PORTFOLIO_OPTIMIZER_CLUSTER=local`
on a laptop needs nothing beyond the locked environment; on Kubernetes the `kubernetes` extra
(`uv sync --locked --extra kubernetes`) and the Dask operator. See [how to run on a cluster](docs/how-to-run-on-a-cluster.md).

Python 3.12 or newer. Dependencies are locked in `uv.lock`; solver versions are recorded in every manifest.
