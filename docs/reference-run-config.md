# Reference: the run config

A run config is one JSON document validated by `portfolio_optimizer.config.models.RunConfig`. Unknown
keys are rejected everywhere. Money, weights, and rates are written as JSON strings (`"0.05"`) and become
exact `Decimal` values; solver tolerances are JSON numbers.

**Every key, its type, its default, and its description are in the generated JSON Schema**
(`configs/run-config.schema.json`, below) — including the parameters of every shipped step. This page
carries only what the schema cannot say: the signature each kind of step must have, how datasets and
their rate limits behave at load time, what an account's constraint rows and style limits look like
(neither is part of the config), and the environment variables. For what each block *means* to the
engine and when it is consumed, see [reading a run config](explanation-run-config.md).

## JSON Schema

`configs/run-config.schema.json` is a draft 2020-12 JSON Schema generated from the models
(`uv run portfolio-optimizer schema`), so it cannot disagree with what the engine accepts; a test fails
when the checked-in file is stale. It carries every field's description, a definition per kind of step,
and — for every shipped loader, rule, term, constraint, and sink — the exact parameter schema, applied
by `if`/`then` on the step's `name`. Custom (qualified) step names are allowed with any parameters,
which the engine validates at resolution time instead.

Ways to validate a config:

| Method | What it checks |
|---|---|
| `"$schema": "./run-config.schema.json"` at the top of the file | Live validation and completion in editors that honor `$schema` (VS Code, JetBrains). The key is accepted and ignored by the engine. |
| `uv run portfolio-optimizer validate-config CONFIG` | Everything the config can be checked for without data: the models, plus importing every step, checking signatures, validating params (including custom steps), checking the solver, and constructing every term once under the run's side profile. Constraints are loaded data, so they are not checked here. Prints how dependencies between portfolios will be derived, then one line per resolved step. The same resolution runs at the start of `run` and on every worker. |
| Any draft 2020-12 validator (`check-jsonschema`, `jsonschema`, `ajv`) against the schema file | The schema alone — suitable for CI pipelines that do not install the engine. |

The schema cannot express one rule the models enforce: `as_of_date` must carry a time zone.
`validate-config` reports it.

## Step references

A step is either a bare string or an object:

```json
"cap_single_name"
{"name": "cap_single_name", "params": {"max_weight": "0.05"}}
```

`name` is a bare identifier resolved in the template module for its kind (`loaders.py`, `assembly.py`,
`rules.py`, `solve_order.py`, `terms.py`, `solvers.py`, `sinks.py`), or a qualified `package.module:function`. `params` (default `{}`) is validated
against the function's `params` annotation; a function without a `params` argument rejects any params.

| Kind | Signature |
|---|---|
| dataset loader (`portfolios` included) | `(request: LoadRequest[, params]) -> pd.DataFrame`, plain or `async def` |
| assembly step | `(frames: Frames[, params]) -> Frames` |
| rule | `(data: PortfolioData[, params]) -> PortfolioData` |
| solve-order step | `(data: PortfolioData[, params]) -> Decimal` — finite; lower solves first |
| objective term | `(x: DecisionVars, spec: ProblemSpec[, params][, chain: ChainState]) -> ObjectiveTerm` |
| constraint | `(x: DecisionVars, spec: ProblemSpec[, params][, chain: ChainState]) -> ConstraintSet` |
| solve step | `(request: SolveRequest[, params]) -> SolveResult` |
| sink | `(orders: pd.DataFrame, io: IoContext[, params]) -> tuple[Artifact, ...]` |

Optional arguments are recognized by name and must carry exactly the annotation shown. Only loaders may
be `async def`; every other kind runs synchronously. Declaring `chain` on a term or constraint is what
makes a portfolio wait for higher-priority portfolios that can trade a security it can trade too, on the
side the run couples through (buys under `both` and `buy`, sells under `sell`).

## Datasets

### `depends_on`, `scope`, and `batch_size`

The engine starts each dataset the moment its `depends_on` dependencies have fully loaded (a
per-portfolio dependency: every batch, concatenated) and hands their frames to the loader as
`request.inputs`. Declaring `portfolios` is what fills `request.portfolio_ids` with the book's
surviving ids. Unknown names, self-dependencies, and cycles are rejected at validation, the cycle
named. A `per_portfolio` dataset implies `depends_on: ["portfolios"]` and is never passed to assembly
— attach its columns in a rule.

`portfolios` is required and always global: it is the dataset that produces the ids a `per_portfolio`
dataset is partitioned by, and it cannot depend on one. For a per-portfolio batch, an input frame in
`request.inputs` with a `portfolio_id` column is reduced to the batch's rows; one without is passed
whole. A dataset downstream of a failed dataset is skipped, never called, and named beside the failure
in the run's rejection.

A `per_portfolio` batch that fails fails only its own portfolios, which are recorded at stage `load`
and skipped; the rest of the book runs. A dataset *no* batch of which came back is the source being
down, and rejects the run. So does a required dataset that is missing, or one that violates its schema
— structural problems reject, coverage problems fail a portfolio.

### `rate_limit` and `rate_limits`

Every loaded entry of `datasets` (`portfolios` included; an inline book has no source to bound) may carry a `rate_limit`, which the loader
receives as `request.rate_limiter`. It is written one of two ways:

- **An inline bound**, private to that input: `"rate_limit": {"requests_per_second": 5, "max_in_flight": 2}`.
  Use this when sources scale differently — a fragile vendor API on one input, a database that takes
  32 concurrent queries on another.
- **The name of a shared pool** declared under the top-level `rate_limits`: `"rate_limit": "vendor_api"`.
  Inputs naming the same pool share its budget, which is what you want when two datasets come from the
  same backend.

A bound sets `requests_per_second` (a continuously refilled token bucket, with `burst` allowed at
once), `max_in_flight` (simultaneous requests across every loader drawing from it), or both — at least
one is required. Naming an undeclared pool is a config error.

## `execution`

There is no execution mode. Every portfolio builds in a worker at once; solves are submitted with their
predecessors' contributions as dependencies and run where the build lives; outcomes are classified in
solve order. Under `dependencies: none`, no portfolio waits for another. `dependencies` is declared,
not inferred: constraints are loaded data the engine does not interpret, so it cannot tell whether
yours read the chain — though a row with a typed `kind` column does declare it, and narrows the graph
accordingly. The workers are the Dask cluster the run provisions for itself — local worker processes on
a laptop, pods on Kubernetes, or a scheduler someone else runs — sized by the settings below; they are
recorded in the manifest's `settings` block and never affect the config hash.

## Shipped steps

Loaders: `load_portfolios`, `load_holdings` (one call per account, fanned out under the input's rate
limit), `load_universe`, `load_details` (a plain `def`: one query per batch of ids, run in a worker
thread), `load_constraints`, `load_parameters` (`set_name`, default the dataset's own name). Every one
of them stands in for a service and takes `min_latency_s` and `max_latency_s`, which override the wait
that source is pretended to take; a real loader has neither. Assembly steps: `join`, `union`, `select`, `drop`. Rules:
`cap_single_name` (`max_weight`), `add_zero_alpha`, `restrict_low_liquidity` (`dataset`, `key`; reads its
threshold from a `name`/`value` extra dataset, by default `buy_universe_parameters`/`min_adv_shares`),
`attach_universe_columns` (`columns`; copies per-security columns from the universe onto holdings,
matched on `security_id` — default every column the universe carries beyond its schema).
Solve-order steps: `most_uninvested_first`. Terms: `alpha` (`column`), `tax_cost`,
`transaction_cost` (`cost_bps`), each with `weight` (default `"1"`). Constraints:
`long_only`, `max_weight`, `cash_bounds`, `sector_bound` (`sector`, `lower`, `upper`, `tolerance`), `turnover_cap`,
`cumulative_adv_participation` (`chain`: `trade ≤ adv_capacity` and `coupled ≤ adv_capacity − predecessors' trades on the side the run couples through`).
Solve steps: `cvxpy` (default), `pro_rata_fill`. Sinks: `orders_to_parquet`, `orders_to_csv` (`subdir`, default `orders`).

## Constraints (the `constraints` dataset)

Which constraints bind an account is data, not config: there is no `constraints` key in a run config.
The dataset is per portfolio, like `holdings`, and optional — a run whose solve step needs none
declares no such dataset.

The engine validates one column and carries the rest untouched:

| Column | Type | Description |
|---|---|---|
| `portfolio_id` | string | The account the row applies to. The only column the engine reads. |

Every other column is the desk's own and reaches the solve step, which is the only thing that
interprets them. The shipped `cvxpy` step reads this convention:

| Column | Type | Description |
|---|---|---|
| `name` | string | A step in `terms.py` or a qualified `package.module:function`. |
| `label` | string, optional | Unique among the portfolio's rows; the verifier's report and the manifest key on it. Defaults to the bare name. |
| `params` | string, optional | A JSON object validated against the function's own `Params` model. Money and weights are strings inside it, as in a config. |

A row naming `trade_balance` is refused: the trade identity comes from `sides`. Nothing about a row is
checked until the solve step uses it, so a bad row fails that portfolio at stage `solve` and the rest
of the book runs.

A limit that is not a per-account scalar carries its numbers on the row. `sector_bound` is the shipped
example: one row per sector, each with its own `label`, and `params` naming the `sector` and its
`lower` and `upper` band (`tolerance` is the slack the verifier allows). The spec supplies the sector's
membership — one sparse row of the matrix the build derives from `universe.sector` — and the row
supplies the band, so bounding a second sector is a second row rather than a schema change.

```csv
portfolio_id,name,label,params
P1,sector_bound,tech,"{""sector"": ""TECH"", ""lower"": ""0.5"", ""upper"": ""1""}"
```

## Style limits (columns of `details`)

Every bounded constraint reads its limits from the data, not from the config. The per-account scalars
are columns of the `details` frame:

| Column | Type | Description |
|---|---|---|
| `max_weight` | decimal in (0, 1] | Single-name cap. |
| `max_turnover` | decimal in [0, 2] | Two-way turnover as a fraction of NAV. |
| `max_adv_participation` | decimal in [0, 1] | Fraction of each name's ADV the portfolio may trade. |
| `min_trade_notional` | decimal ≥ 0 | Orders below this notional are dropped. Not a constraint: the order step applies it after the solve. |
| `cash_lb` | decimal in [0, 1] | Lower bound on `1 − Σw`. |
| `cash_ub` | decimal in [0, 1] | Upper bound on `1 − Σw`; `cash_lb = cash_ub = 0` is full investment. Must be at least `cash_lb`. |

A limit that is not one scalar per account — a per-sector band — lives on its constraint row instead;
see the section above.

## Environment

All required unless stated; no defaults; an unknown `PORTFOLIO_OPTIMIZER_*` variable is an error. `run
--data-root`, `run --output`, and `run --max-workers` override the corresponding setting for one run.

| Variable | Values | Description |
|---|---|---|
| `PORTFOLIO_OPTIMIZER_OUTPUT_DIR` | path | Where `<run_id>/` directories are written. |
| `PORTFOLIO_OPTIMIZER_DATA_ROOT` | path | `request.data_root` for the shipped file loaders. |
| `PORTFOLIO_OPTIMIZER_LOG_LEVEL` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` | |
| `PORTFOLIO_OPTIMIZER_CLUSTER` | `local` \| `kubernetes` \| `auto` \| `tcp://host:port` \| `tls://host:port` | The Dask cluster the run provisions for itself (`local`: worker processes on this machine; `kubernetes`: pods through the Dask operator) or a scheduler to connect to. `auto` becomes `kubernetes` when `KUBERNETES_SERVICE_HOST` is set and `local` otherwise; the manifest records the resolved value. |
| `PORTFOLIO_OPTIMIZER_MIN_WORKERS` | integer ≥ 1, ≤ max | Workers provisioned before the load stage. |
| `PORTFOLIO_OPTIMIZER_MAX_WORKERS` | integer ≥ 1 | Workers after assembly. Every build and every solve is submitted at once; the scheduler runs what is ready. |
| `PORTFOLIO_OPTIMIZER_CLUSTER_TIMEOUT_S` | number > 0 | How long to wait, after assembly, for the first worker. |
| `PORTFOLIO_OPTIMIZER_WORKER_IMAGE` | image reference | Required when the cluster resolves to `kubernetes`: the image worker pods run, normally this run's own. |
| `PORTFOLIO_OPTIMIZER_IMAGE_DIGEST` | string | Optional; set by the platform. Part of every process's environment fingerprint and forwarded to worker pods. |

See [how to run on a cluster](how-to-run-on-a-cluster.md).
