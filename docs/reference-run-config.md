# Reference: the run config

A run config is one JSON document validated by `portfolio_optimizer.config.models.RunConfig`. Unknown
keys are rejected everywhere. Money, weights, and rates are written as JSON strings (`"0.05"`) and become
exact `Decimal` values; solver tolerances are JSON numbers. For what each block means to the engine and
when it is consumed, see [reading a run config](explanation-run-config.md).

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
| `uv run portfolio-optimizer validate-config CONFIG` | Everything: the models, plus importing every step, checking signatures, validating params (including custom steps), and the execution-mode rules. |
| Any draft 2020-12 validator (`check-jsonschema`, `jsonschema`, `ajv`) against the schema file | The schema alone — suitable for CI pipelines that do not install the engine. |

The schema cannot express two rules the models enforce: `as_of` must carry a time zone, and chain-aware
steps (those declaring `ctx`/`chain`) are only allowed under the sequential modes with `fail_fast`.
`validate-config` reports both.

## Top level

| Key | Type | Required | Description |
|---|---|---|---|
| `run` | object | yes | Run identity: `name` (non-empty), `as_of` (timezone-aware ISO-8601 timestamp), `tags` (string map, default `{}`). |
| `portfolios` | step or input | yes | Loader returning the `portfolios` frame (`portfolio_id`, `solve_order`); a bare step, or `{"loader": step[, "rate_limit": ...]}` to bound its source. Solve order is ascending `solve_order`. |
| `datasets` | object | yes | Named inputs, each `{"loader": step[, "rate_limit": name or bound]}`. Must include `holdings`, `universe`, `details`, `constraints`, `targets`; `covariance` is optional and enables the `risk` term; any other name is an extra dataset for `assembly.joins`. All dataset loaders run concurrently. |
| `rate_limits` | object | no | Named pools that inputs on the same backend share; see below. Default `{}`. |
| `assembly` | object | no | `portfolio_key` (default `portfolio_id`), `security_key` (default `security_id`), `joins` (default `[]`). |
| `rules` | step list | no | Business-logic rules, run in order. Default `[]`. |
| `objective` | object | yes | `sense` (only `minimize`), `terms` (step list, at least one). |
| `constraints` | step list | no | Constraint functions. Default `[]`. |
| `solver` | object | no | `name` (default `CLARABEL`; must be installed in cvxpy), `options` (map of solver options passed verbatim to `Problem.solve`, default `{}`), `time_limit_s` (number > 0 or absent; mapped to `time_limit` for `CLARABEL`, `OSQP`, and `HIGHS` and to `time_limit_secs` for `SCS`; any other solver rejects it), `verbose` (default `false`). |
| `post_solve` | object | no | `violation_tol` (default `1e-6`), `objective_rel_tol` (`1e-5`), `objective_abs_tol` (`1e-9`); all > 0. |
| `sink` | step | yes | Where orders go. |
| `execution` | object | yes | See below. |

## Step references

A step is either a bare string or an object:

```json
"cap_single_name"
{"name": "cap_single_name", "params": {"max_weight": "0.05"}}
```

`name` is a bare identifier resolved in the template module for its kind (`loaders.py`, `rules.py`,
`terms.py`, `sinks.py`), or a qualified `package.module:function`. `params` (default `{}`) is validated
against the function's `params` annotation; a function without a `params` argument rejects any params.

| Kind | Signature |
|---|---|
| portfolios, dataset loader | `(request: LoadRequest[, params]) -> pd.DataFrame`, plain or `async def` |
| constraints loader | `(request: LoadRequest[, params]) -> dict[str, dict[str, object]]`, plain or `async def` |
| rule | `(data: PortfolioData[, params][, ctx: SolveContext]) -> PortfolioData` |
| objective term | `(x: DecisionVars, spec: ProblemSpec[, params][, chain: ChainState]) -> ObjectiveTerm` |
| constraint | `(x: DecisionVars, spec: ProblemSpec[, params][, chain: ChainState]) -> ConstraintSet` |
| sink | `(orders: pd.DataFrame, io: IoContext[, params]) -> tuple[Artifact, ...]` |

Optional arguments are recognized by name and must carry exactly the annotation shown. Only loaders may
be `async def`; every other kind runs synchronously.

## Rate limits

Every input — `portfolios` and each entry of `datasets` — may carry a `rate_limit`, which the loader
receives as `request.rate_limiter`. It is written one of two ways:

- **An inline bound**, private to that input: `"rate_limit": {"requests_per_second": 5, "max_in_flight": 2}`.
  Use this when sources scale differently — a fragile vendor API on one input, a database that takes
  32 concurrent queries on another.
- **The name of a shared pool** declared under the top-level `rate_limits`: `"rate_limit": "vendor_api"`.
  Inputs naming the same pool share its budget, which is what you want when two datasets come from the
  same backend.

A bound, inline or in a pool, has these keys:

| Key | Type | Description |
|---|---|---|
| `requests_per_second` | number > 0 | Sustained rate, refilled continuously (a token bucket). Omit for no rate bound. |
| `burst` | integer ≥ 1 | Requests allowed at once before the rate applies. Default: `requests_per_second` rounded up. Requires `requests_per_second`. |
| `max_in_flight` | integer ≥ 1 | Simultaneous requests across every loader drawing from the bound. Omit for no concurrency bound. |

At least one of `requests_per_second` and `max_in_flight` is required. Naming an undeclared pool is a
config error.

## `assembly.joins[]`

| Key | Type | Required | Description |
|---|---|---|---|
| `into` | `holdings` \| `universe` \| `targets` | yes | Frame that receives the columns. |
| `source` | string | yes | A declared dataset other than `into` and `constraints`. |
| `on` | string list | yes | Join keys; their dtypes are aligned to `into` before merging. |
| `how` | `left` \| `inner` | no | Default `left`. |
| `cardinality` | `one_to_one` \| `one_to_many` \| `many_to_one` | yes | Enforced; a violation aborts the run. |
| `require_all_matched` | bool | no | Default `false`. When true, every row of `into` must find a match; unmatched keys are reported. |

Columns already present in `into` are never overwritten; such a join is rejected.

## `execution`

| Key | Type | Required | Description |
|---|---|---|---|
| `mode` | `sequential` \| `parallel_build_sequential_solve` \| `parallel` | yes | See table. |
| `executor` | `process` \| `thread` | no | Default `process`. `parallel` requires `process`. |
| `max_workers` | integer ≥ 1 | no | Default `1`. |
| `on_error` | `fail_fast` \| `continue` | no | Default `fail_fast`. |

| Mode | Rules and build | Solve | Chain-aware steps allowed |
|---|---|---|---|
| `sequential` | main process, solve order, live context | main process | rules (`ctx`), terms and constraints (`chain`) |
| `parallel_build_sequential_solve` | executor, no context | main process, solve order | terms and constraints (`chain`) only |
| `parallel` | executor, whole pipeline per portfolio | in the worker | none |

Config-load errors: a chain-aware step under `parallel`; a `ctx` rule under
`parallel_build_sequential_solve`; `executor: thread` with `parallel`; any chain-aware step with
`on_error: continue`.

## Shipped steps

Loaders: `csv` (`path`, `decimal_columns`, `utc_datetime_columns`, `dtypes`), `csv_per_portfolio`
(`directory`, `decimal_columns`, `utc_datetime_columns`, `dtypes`; reads `<directory>/<portfolio_id>.csv`
per portfolio under the input's rate limit), `parquet` (`path`, `decimal_columns`), `json_constraints`
(`path`). The column-typing params apply to extra datasets only; engine-known datasets are typed by
their schema. Rules: `cap_single_name` (`max_weight`),
`add_zero_alpha`, `restrict_low_liquidity` (`min_adv_shares`), `avoid_cross_portfolio_wash_sales`
(`ctx`). Terms: `tracking_error`, `risk`, `alpha` (`column`), `tax_cost`, `transaction_cost` (`cost_bps`),
each with `weight` (default `"1"`). Constraints: `trade_balance`, `long_only`, `max_weight`,
`cash_bounds`, `sector_bounds` (`tolerance`), `turnover_cap`, `cumulative_adv_participation` (`chain`).
Sinks: `orders_to_parquet`, `orders_to_csv` (`subdir`, default `orders`).

## Style constraints (the `constraints` dataset)

Per portfolio id, an object validated into `StyleConstraints`:

| Key | Type | Description |
|---|---|---|
| `max_weight` | decimal in (0, 1] | Single-name cap. |
| `max_turnover` | decimal in [0, 2] | Two-way turnover as a fraction of NAV. |
| `min_trade_notional` | decimal ≥ 0 | Orders below this notional are dropped. |
| `cash_bounds` | `[low, high]`, 0 ≤ low ≤ high ≤ 1 | Bounds on `1 − Σw`; `["0", "0"]` is full investment. |
| `max_adv_participation` | decimal in [0, 1] | Fraction of each name's ADV the portfolio may trade. |
| `sector_bounds` | map of sector → `[low, high]` | Default `{}`; every sector must exist in the universe. |
| `long_only` | `true` | Shorting is a non-goal of the template. |

## Environment

`PORTFOLIO_OPTIMIZER_OUTPUT_DIR`, `PORTFOLIO_OPTIMIZER_DATA_ROOT`, `PORTFOLIO_OPTIMIZER_LOG_LEVEL`
(`DEBUG` \| `INFO` \| `WARNING` \| `ERROR`). All required; no defaults; an unknown
`PORTFOLIO_OPTIMIZER_*` variable is an error. `run --data-root` and `run --output` override the first two.
