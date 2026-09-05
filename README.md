# portfolio-optimizer

**Turn portfolio data and investment rules into verified trade orders, with a record of how each result was produced.**

`portfolio-optimizer` is a Python template for teams building a portfolio construction or rebalancing
workflow across many accounts. It brings data loading, account-specific rules, optimization, order
rounding, verification, and audit records into one configurable pipeline. A JSON file describes the
workflow; ordinary Python functions connect it to your data and investment process.

The project uses pandas for tabular data and CVXPY with Clarabel for the shipped optimization step.
You supply the holdings, prices, signals, constraints, and business logic. The engine handles the
execution around them, from checking the configuration to writing orders and a run manifest.

This is an early-stage, extensible template. The examples use local CSV inputs and write order files;
connecting live data sources and a trading system requires your own loaders and sink.

## Why use it?

- **Spend more time on investment logic.** Reuse the loading, scheduling, verification, and reporting
  machinery while changing rules, objectives, constraints, or the solver to fit your process.
- **Apply different mandates across one book.** Each account gets its own holdings, parameters, and
  constraint rows. Shipped constraint kinds cover position weights, group and factor exposures, cash,
  turnover, and participation in daily trading volume.
- **Check what the orders actually leave you holding.** The engine independently checks the solved
  portfolio with NumPy, converts weights into quantities respecting trading increments and minimum
  denominations, then checks the resulting portfolio again before sending its orders to the sink.
- **Coordinate accounts that share liquidity.** Higher-priority accounts' trades consume capacity
  available to later accounts. The engine derives dependencies from tradable securities and
  constraint scope, allowing independent accounts to solve concurrently on Dask.
- **Make results easier to explain and investigate.** Saved problem specifications, solutions,
  constraint margins, rule audits, data and code hashes, package versions, and timing traces help
  answer what ran, which limits bound, and where a failure occurred.
- **Use the same workflow locally and on a cluster.** Debug in a single process, then select local
  Dask workers, an existing scheduler, or Dask Gateway through environment settings. The investment
  configuration stays the same.

## What you can run

Each run has one order flow and a timezone-aware as-of instant:

| Flow | Allowed trades | Example use |
|---|---|---|
| **Inflow** | Buy | Invest available cash within account limits and shared liquidity. |
| **Outflow** | Sell | Reduce holdings or apply a tax-aware selling objective. |
| **Rebalance** | Buy and sell | Adjust existing positions toward the configured objective and limits. |

The repository includes examples of all three over **100 accounts and three securities**, plus an
[inflow following an outflow](configs/example_inflow_after_outflow.json) that reads the earlier
orders. The examples combine expected-return signals and transaction costs; the outflow also includes
a tax-cost term. See [how to run an order flow](docs/how-to-run-an-order-flow.md).

## Quick start

From a clone of this repository, with **uv** installed and **Python 3.12 or newer** available:

```bash
uv sync --locked
uv run portfolio-optimizer validate-config configs/example_inflow.json
uv run portfolio-optimizer run configs/example_inflow.json --data-root examples/data --as-of 2026-08-28T00:00:00Z
```

No environment configuration is required. The default backend is `inline`, which runs build and solve
tasks in the current process. The sample loaders deliberately simulate service latency, so allow
roughly half a minute for the example.

The CLI prints the manifest path, each account's outcome and binding constraints, and the configured
business-check results. A successful example writes these artifacts under `out/<run_id>/`:

| Artifact | What it gives you |
|---|---|
| `orders/orders.parquet` | The combined trade orders from successful accounts. |
| `manifest.json` | Configuration, provenance, per-account outcomes, verification reports, and timing. |
| `problem_specs/`, `solutions/`, `executed/`, `chain/` | Saved problems, solved and rounded portfolios, and predecessor trading state for re-verification. |
| `trace.json` | A timeline of loading, building, solving, and other stages. |

For a guided walkthrough with hand-checkable orders and saved-solution verification, follow
[your first run](docs/tutorial-first-run.md). To try the other flows:

```bash
uv run portfolio-optimizer run configs/example_outflow.json --data-root examples/data --as-of 2026-08-28T00:00:00Z
uv run portfolio-optimizer run configs/example_rebalance.json --data-root examples/data --as-of 2026-08-28T00:00:00Z
```

## How a run works

1. **Resolve the configuration.** Import the named steps, validate their signatures and parameters,
   and check solver availability before loading data.
2. **Load and assemble the inputs.** Load datasets according to their dependencies, with bounded
   concurrency and optional per-account batching. Join related sources and validate the assembled data.
3. **Build each account's problem.** Apply its rules, construct the optimization inputs, and identify
   the securities and constraints that connect it to other accounts.
4. **Schedule and solve.** Respect account priority wherever earlier trades affect later limits;
   submit independent work concurrently when using a parallel backend.
5. **Verify and round.** Check the solver's result independently, round to order quantities, and
   reject the account's result if the rounded portfolio fails verification.
6. **Publish and record.** Send successful accounts' orders to the configured sink, run the business
   checks on published orders, and record outcomes and artifacts in the manifest.

Business checks run **after publication** and report `passed`, `failed`, or `not_exercised`; they are
an audit of the published orders. Constraint verification of solved and rounded portfolios happens
before publication. See [how to QA a deployment](docs/how-to-qa-a-deployment.md) for reading both.

### Parallelism follows the portfolio dependencies

If two accounts can trade the same security and share a volume limit, their solves may need to run
in priority order. Accounts with disjoint trading universes can often run independently. The engine
derives that graph from the built problems, and records its edges, independent components, and
critical path in the manifest.

![A sequential account schedule compared with a dependency graph that allows independent mandates to solve concurrently](docs/images/derived-schedule.svg)

The benefit depends on the book: distinct mandates can expose parallel work, while accounts sharing
the same constrained universe can form a fully sequential chain. `execution.dependencies: "all"`
provides a strict sequential dependency schedule for comparison. The repository includes equivalence
tests and synthetic-book benchmarks; see the
[architecture explanation](docs/explanation-architecture.md) and [benchmark scripts](benchmarks/).

## Adapt it to your process

The JSON config selects behavior; datasets carry positions, prices, signals, account parameters, and
constraint rows. The `--as-of` argument selects the snapshot instant. Environment settings control
where the workflow runs and writes its output. This separation lets one workflow operate over new
data or on different infrastructure without rewriting its investment logic.

### The one convention

A step is an ordinary Python function, named in the config. Use a shipped name, a qualified name such
as `my_firm.rules:cap_single_name`, or an installed package's entry point. The resolver validates the
function's parameters and records its source hash.

For example, this entry selects the shipped position-cap rule:

```json
{"name": "cap_single_name", "params": {"max_weight": "0.05"}}
```

Add it to a config's `rules` list to tighten the configured single-name cap to at most 5%. The same
naming convention covers loaders, assembly steps, rules, solve-order steps, builders, solvers, sinks,
and checks.

Objective terms and constraints are typed kinds with a solver representation and an independent
numeric evaluator. Custom kinds extend both optimization and verification. You can also replace the
CVXPY solve step with a function accepting `SolveRequest` and returning `SolveResult`.

Start with the extension point matching your change:

| Your goal | Guide |
|---|---|
| Connect a data source or publish orders elsewhere | [Add a loader or sink](docs/how-to-add-a-loader-or-sink.md) |
| Change account eligibility or restrictions | [Add a rule](docs/how-to-add-a-rule.md) |
| Bring in security analytics | [Add analytics columns](docs/how-to-add-security-analytics.md) |
| Change the objective or available constraints | [Add a term or constraint kind](docs/how-to-add-a-term.md) |
| Change account priority | [Set the solve order](docs/how-to-set-the-solve-order.md) |
| Use another optimization implementation | [Write a solve step](docs/how-to-write-a-solve-step.md) |
| Audit a business rule on published orders | [Add a check](docs/how-to-add-a-check.md) |
| Run with more compute | [Run on a cluster](docs/how-to-run-on-a-cluster.md) |

Run `uv run portfolio-optimizer steps` to list the available steps and kinds.
[`.env.example`](.env.example) documents runtime settings; load a customized `.env` explicitly with
`uv run --env-file .env ...`.

## Documentation

- **Get started:** [Your first run](docs/tutorial-first-run.md).
- **Understand configuration:** [Reading a run config](docs/explanation-run-config.md),
  [config reference](docs/reference-run-config.md), and [JSON Schema](configs/run-config.schema.json).
- **Understand execution:** [Architecture](docs/explanation-architecture.md) and
  [the life of a run](docs/explanation-run-lifecycle.md).
- **Inspect inputs and results:** [Portfolio data](docs/reference-portfolio-data.md),
  [outputs, manifests, and CLI commands](docs/reference-manifest.md), and
  [deployment QA](docs/how-to-qa-a-deployment.md).

## Development

The main extension modules live directly under `src/portfolio_optimizer/` (`loaders.py`, `rules.py`,
`solvers.py`, `sinks.py`, and related steps). `engine/` owns orchestration, `domain/` owns data contracts
and typed kinds, `config/` owns configuration and resolution, and `cvx/` contains the CVXPY adapter.
Worked inputs are in `examples/data/`; tests and performance harnesses are in `tests/` and `benchmarks/`.

```bash
uv sync --locked
uv run pre-commit run --all-files
uv run pytest
```

Dependencies are locked in `uv.lock`. [IDEAS.md](IDEAS.md) tracks design discussions and known issues.

## The run config, block by block

The complete inflow config is preserved here as a reference. The annotations explain the wiring;
[`configs/example_inflow.json`](configs/example_inflow.json) is the runnable, strict JSON version.
A test checks that the annotated copy matches it.

<details>
<summary>Expand the annotated inflow configuration</summary>

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
  // loaded, with concurrency bounded by the loader settings. The four engine-known datasets are
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

    // Business parameters can be loaded as data too.
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

  // Business checks on published orders, against the data as the rules first saw it:
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

</details>
