# portfolio-optimizer

A template for a JSON-driven, auditable portfolio-optimization engine on pandas and cvxpy. Keep the
engine; write your own loaders, rules, term and constraint kinds, solve steps, and sinks as ordinary
Python.

A run is one JSON file, an order flow, and an as-of date. The engine loads the data the file names,
applies your rules per portfolio, builds one problem per portfolio, solves along a dependency graph it
derives from the data, re-verifies every answer without cvxpy, rounds to executable quantities and
re-verifies the book they leave, publishes them, and writes a manifest that lets anyone reproduce the
run. A run is an inflow, an outflow, or a rebalance: it buys, it sells, or it does either to bring a
book back inside its bounds. A desk's order flows are separate runs over one snapshot, and the template
ships one of each.

## The run config, block by block

The quickest way to learn the engine is the inflow it ships with,
[`configs/example_inflow.json`](configs/example_inflow.json), annotated here:

```jsonc
// The inflow the template ships, annotated. The real file is strict JSON; a test strips these
// comments and compares the rest to it. The outflow changes the name, `order_flow`, and adds a
// `tax_cost` term on what is sold; the rebalance changes the first two.
{
  // Recorded in the manifest, kept out of the config hash. The as-of instant is `run --as-of`,
  // so one wiring runs every day under one hash and `diff-manifests` compares Monday with Tuesday.
  "run": {"name": "example_inflow", "tags": {"desk": "template"}},

  // `inflow` buys, `outflow` sells, `rebalance` may do either. One decision variable per name, so
  // a term or constraint that reads a side the run lacks is refused before any data loads.
  "order_flow": "inflow",

  // Every input the run loads, each a frame. A dataset starts the moment its `depends_on` have
  // loaded, so the stage costs its longest chain, not its sum. The four engine-known datasets are
  // validated against fixed schemas; any further column is exported to the problem by name.
  "datasets": {
    // The book of record: portfolio ids and solve priorities. A fixed book can be written inline.
    "portfolios": {"loader": "load_portfolios"},

    // `per_portfolio` fans out: one call per `batch_size` accounts, at most `max_in_flight` open.
    // A call each is the shape of a custodian that answers one account at a time.
    "holdings": {
      "loader": "load_holdings",
      "scope": "per_portfolio",
      "batch_size": 1,
      "max_in_flight": 8
    },

    // The security master: price, sector, volume, increment, and the restricted flag. No `scope`
    // means one call for the book; no `depends_on` means it starts at once.
    "universe": {"loader": "load_universe"},

    // The research store: alpha and cost per name. No one service knows a name's price and its
    // alpha, so the universe arrives in two parts and the `assembly` step below makes one of them.
    "signals": {"loader": "load_signals"},

    // The account master: NAV, cash, tax rates, style limits, and any other column the desk keeps.
    "details": {
      "loader": "load_details",
      "scope": "per_portfolio",
      "batch_size": 25,
      "max_in_flight": 4
    },

    // Which constraints bind each account and how tight: one typed row each. One kind reads what
    // higher-priority portfolios have already traded, and that is the only reason a portfolio ever
    // waits for another. `depends_on` hands the loader the book's ids.
    "constraints": {"loader": "load_constraints", "depends_on": ["portfolios"]},

    // The blotter, for the wash-sale rule below. A name the engine does not know is an extra
    // dataset, carried untouched to the rules and the solve step.
    "trades": {"loader": "load_trades", "depends_on": ["portfolios"]},

    // Runtime parameters are data too.
    "global_parameters": {"loader": "load_parameters"},
    "buy_universe_parameters": {"loader": "load_parameters"}
  },

  // Once per run, after the loaders and before the schemas are checked. The join is a claim the
  // engine enforces: every universe name has exactly one signal row, or the run stops naming the rest.
  "assembly": [
    {
      "name": "join",
      "params": {
        "into": "universe",
        "source": "signals",
        "on": ["security_id"],
        "cardinality": "one_to_one",
        "require_all_matched": true
      }
    },
    // The source has done its job; dropped, it is not carried into every account's bundle.
    {"name": "drop", "params": {"datasets": ["signals"]}}
  ],

  // Applied in order to each portfolio; a rule never sees another portfolio. Drop illiquid names,
  // then freeze every name the account traded in the last thirty days.
  "rules": ["restrict_low_liquidity", "restrict_recent_trades"],

  // The sum is minimized, so a reward has a negative weight. `linear` is `weight · columnᵀvector`
  // over any per-security column and a decision vector.
  "objective": [
    {"kind": "linear", "name": "alpha", "column": "alpha", "weight": "-1"},
    {
      "kind": "linear",
      "name": "transaction_cost",
      "column": "tcost_per_dollar",
      "vector": "trade"
    }
  ],

  // The step that decides the weights. The solver is checked when the config resolves; no fallback.
  "solve": {
    "name": "cvxpy",
    "params": {"solver": "CLARABEL", "options": {"max_iter": 200}, "time_limit_s": 60.0}
  },

  // Tolerances for the cvxpy-free re-verification of every solution.
  "post_solve": {"violation_tol": 1e-6, "objective_rel_tol": 1e-5, "objective_abs_tol": 1e-9},

  // Where the orders go.
  "sink": {"name": "orders_to_parquet", "params": {"subdir": "orders"}},

  // Business rules proven on the orders that went out, against the data as the rules first saw it:
  // each records passed, failed, or not_exercised (the book never reached the rule) under its label.
  "checks": [
    {"name": "restricted_never_traded", "label": "restricted_never_traded"},
    {"name": "no_trades_inside_wash_window", "label": "wash_sale_window", "params": {"window_days": 30}}
  ],

  // What one failure does to the rest, and whether the schedule is the derived graph or the strict
  // line: the same orders either way. Where the work runs is an environment setting, not config.
  "execution": {"on_error": "fail_fast", "dependencies": "overlap"}
}
```

Most values are *steps*: a function named bare, looked up in the template module for that kind of step
or among the steps installed packages publish, or as `package.module:function`, with optional `params`
([the one convention](#the-one-convention)). Numbers never live in the config: positions, prices, caps,
bands, tax rates, and the run's own parameters are data. The instant is a run argument. Behavior is the
functions and kinds the config names.

Two keys the example leaves at their defaults: `build` (`standard`, the step that turns a bundle into a
problem, replaceable for tax lots or a factor block) and `solve_order` (a step that computes each
portfolio's priority from the data instead of reading the column).

A failed run is retried with `run CONFIG --retry-of MANIFEST`: any config, over exactly the portfolios
the manifest recorded as failed at the stages `--retry-stages` names, tagged `retry_of`.

[Reading a run config](docs/explanation-run-config.md) goes block by block in depth,
[the reference](docs/reference-run-config.md) covers what the schema cannot say, and
`configs/run-config.schema.json`, generated from the models, lists every key with its type, default,
and description.

## The solve schedule is a derived graph

A chain-aware constraint seems to force a sequential solve: when account *j* may take only what is
left of a name's daily volume after the accounts ahead of it took theirs, the safe schedule is a line,
*N* solves end to end. The line is the worst case, not the truth. Two accounts that cannot trade a
security in common cannot affect each other's feasible set, whatever their constraints read. So every
portfolio builds in parallel and reports its tradable set, and a portfolio waits only for the
higher-priority portfolios whose tradable set intersects what its own chain-reading constraints
consume. Wall clock is the graph's critical path times a solve, not *N* times a solve; independent
components solve at once; the head of the book is solving while the tail is still building. A run in
which nothing reads the chain is told so before any build, and nothing waits at all.

![Eight accounts: the line a chain-aware book seems to force, and the graph the engine derives from mandate overlap — three components, critical path three, same orders either way](docs/images/derived-schedule.svg)

What makes that a result rather than a claim: `execution.dependencies: "all"` runs the same book as the
strict line and produces **byte-identical orders and chain hashes**. A property test asserts it, and
`benchmarks/run_book.py` runs both schedules over synthetic books on a real cluster. Measured
2026-08-30, 8 local workers, Clarabel, from the manifest's own `schedule` and `timing` blocks:

| Book | Derived: edges · components · critical path | Wall clock |
|---|---|---:|
| 100 accounts, 10 disjoint mandates, 2,000 names (~0.14s solves) | 450 · 10 · 10 | **6.9s** |
| — the same book as the line (`dependencies: all`) | 4,950 · 1 · 100 | 14.9s |
| 12 accounts, 4 disjoint mandates, 30,000 names (~2s solves) | 12 · 4 · 3 | **14.0s** |
| — the same book as the line | 66 · 1 · 12 | 24.6s |
| 1,000 accounts, 10 disjoint mandates | 49,500 · 10 · 100 | 34.1s — capacity-bound, 6.1× parallel on 8 workers |

The counterweight: overlap is on *any* shared tradable name, so the win belongs to books partitioned
by mandate, universe, or restriction list (`restrict_to_mandate` is the shipped shape). A book whose
accounts all trade one universe is a complete DAG; the shipped example is deliberately that case, and
its manifest says so: `edges 4950, critical_path 100`. One sector shared between neighbouring mandates
chains the book back into a line. Typed constraint rows narrow the reading: a row declares whether it
reads the chain and, through its `scope`, which names it couples through, so a portfolio whose
constraints read no chain waits for nobody. How the graph is derived and why it is exact is in
[the architecture explanation](docs/explanation-architecture.md#a-run-couples-through-its-one-side-so-the-schedule-is-a-graph);
the `trace.json` beside every manifest shows where the run's wall clock went.

## Quick start

```bash
uv sync --locked
uv run portfolio-optimizer validate-config configs/example_inflow.json
uv run portfolio-optimizer run configs/example_inflow.json --data-root examples/data --as-of 2026-08-28T00:00:00Z
uv run portfolio-optimizer run configs/example_outflow.json --data-root examples/data --as-of 2026-08-28T00:00:00Z
uv run portfolio-optimizer run configs/example_rebalance.json --data-root examples/data --as-of 2026-08-28T00:00:00Z
```

No settings are needed. The default cluster is `inline`, every task in this process one after another,
which is also where a rule is stepped through under a debugger. The example is a hundred accounts over
three securities; each order flow takes about half a minute, nearly all of it the shipped loaders
pretending to be the services they stand in for. The inflow puts each account's cash to work inside the
book's shared liquidity; the outflow harvests the lots held at a loss; the rebalance does the inflow's
buys and sells the negative-alpha name down to its sector floor in one solve. The first two accounts
have a hand-checkable optimum under each; [the tutorial](docs/tutorial-first-run.md) says what to
expect at each step. `portfolio-optimizer steps` lists every step, term, and constraint kind the
environment can name. Every run ends with the proof a QA tester reads: each typed constraint verified
on every portfolio with its margin, and each `checks` entry — a business rule proven on the orders
that went out — as `passed`, `failed`, or `not_exercised`; [how to QA a deployment](docs/how-to-qa-a-deployment.md)
reads it.

## The one convention

A step is an ordinary function in a designated module; the config names it.

```python
# src/portfolio_optimizer/rules.py
class CapSingleNameParams(Params):
    max_weight: Decimal = Field(gt=0, le=1)


def cap_single_name(data: PortfolioData, params: CapSingleNameParams) -> PortfolioData: ...
```

```json
"rules": [{"name": "cap_single_name", "params": {"max_weight": "0.05"}}]
```

No decorators, no registries. Name it `my_firm.rules:cap_single_name` and the engine imports it from
any installed package, or publish it as an entry point in the group `portfolio_optimizer.rule` and name
it bare: that is how a firm shares loaders, rules, and sinks across desks, with the package's version in
every manifest. Before any data loads, the engine imports every named function, checks its signature,
validates its `params` against the function's own model, and records its source hash in the manifest.
Loaders, assembly steps, rules, solve-order steps, the build step, solve steps, sinks, and checks all
follow this rule.

Terms and constraints are *kinds*: strict pydantic models that carry both halves of themselves,
`to_cvxpy` for the shipped solve step and `value` or `residual` in numpy for the verifier, so nothing
the solver was told goes unchecked. A kind published in the group `portfolio_optimizer.term` or
`portfolio_optimizer.constraint` is known everywhere a shipped one is: the resolver, the schedule, the
solve step, the verifier, the JSON Schema.

`portfolio-optimizer validate-config` imports and checks every step and kind; any JSON Schema validator
checks the shape against [`configs/run-config.schema.json`](configs/run-config.schema.json). Adding
`"$schema": "./run-config.schema.json"` to a config gives an editor live completion and validation; the
engine ignores the key.

## Layout

| Path | Role |
|---|---|
| `src/portfolio_optimizer/{loaders,assembly,rules,solve_order,solvers,sinks,checks}.py` | **Yours to edit.** Each ships worked, tested examples. Shared steps live in your own installed package, named `package.module:function` or published as entry points. |
| `src/portfolio_optimizer/engine/` | Loading and assembly, the rule pipeline, the standard build step, the solve stage, cvxpy-free verification, orders, the per-portfolio tasks, the derived dependency graph, the inline and Dask backends, the manifest. Rarely edited. |
| `src/portfolio_optimizer/domain/` | Frame schemas, the per-portfolio bundle and its optimizer frame, the pure-data problem spec and results, the typed constraint and term kinds and their registry, and the order-flow profiles. |
| `src/portfolio_optimizer/config/` | The run-config models, the step resolver, and the JSON Schema generator. |
| `src/portfolio_optimizer/cvx/` | `adapter.py`, the only module that imports cvxpy, and `order_flow.py`, each order flow's decision variable and trade identity in cvxpy. |
| `src/portfolio_optimizer/solving.py` | The solve step's contract: `SolveRequest` in, `SolveResult` out; the solver table. |
| `configs/`, `examples/data/` | The four example configs over one book of a hundred accounts and three securities, one CSV per source (`outflow_orders.csv` is what the outflow wrote), and the generated JSON Schema. |
| `benchmarks/` | `profile_portfolio.py` times one portfolio stage by stage at a chosen book size and order flow; `run_book.py` and `run_state_book.py` run a synthetic book of *N* portfolios on a local cluster and report the derived schedule and timing. The numbers in `IDEAS.md` come from them. |
| `docs/` | Tutorial, how-to guides, reference, and explanation. |
| `IDEAS.md` | Threads that are not yet decisions, and known defects waiting to be fixed. |

## Documentation

- [Tutorial: your first run](docs/tutorial-first-run.md)
- [Explanation: reading a run config](docs/explanation-run-config.md) — the walkthrough above, in depth
- [How to add a rule](docs/how-to-add-a-rule.md)
- [How to add a loader or a sink](docs/how-to-add-a-loader-or-sink.md)
- [How to add security analytics columns to holdings and the universe](docs/how-to-add-security-analytics.md)
- [How to add an objective term or a constraint kind](docs/how-to-add-a-term.md)
- [How to set the solve order](docs/how-to-set-the-solve-order.md)
- [How to run an order flow: inflow, outflow, rebalance](docs/how-to-run-an-order-flow.md)
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

`PORTFOLIO_OPTIMIZER_CLUSTER=local` provisions Dask worker processes on this machine for the run and
tears them down when it ends; a Dask Gateway address needs the `gateway` extra
(`uv sync --locked --extra gateway`). See [how to run on a cluster](docs/how-to-run-on-a-cluster.md).

Python 3.12 or newer. Dependencies are locked in `uv.lock`; solver versions are recorded in every manifest.
