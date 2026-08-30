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
| `uv run portfolio-optimizer validate-config CONFIG` | Everything: the models, plus importing every step, checking signatures, and validating params (including custom steps). Prints how dependencies between portfolios will be derived. |
| Any draft 2020-12 validator (`check-jsonschema`, `jsonschema`, `ajv`) against the schema file | The schema alone — suitable for CI pipelines that do not install the engine. |

The schema cannot express one rule the models enforce: `as_of` must carry a time zone.
`validate-config` reports it.

## Top level

| Key | Type | Required | Description |
|---|---|---|---|
| `run` | object | yes | Run identity: `name` (non-empty), `as_of` (timezone-aware ISO-8601 timestamp), `tags` (string map, default `{}`). |
| `portfolios` | step or input | yes | Loader returning the `portfolios` frame (`portfolio_id`, optional `solve_order`); a bare step, or `{"loader": step[, "rate_limit": ...]}` to bound its source. `solve_order` is a priority: lower solves first, ties break on `portfolio_id`, values may repeat. |
| `datasets` | object | yes | Named inputs, each `{"loader": step[, "rate_limit": name or bound]}`. `constraints` is always required; `holdings`, `universe`, `details`, and `targets` are required unless `assembly` is non-empty, in which case they may be produced by a step and are checked after assembly. Any other name is an extra dataset: visible to every assembly step, and carried into each portfolio's bundle as `data.extras` unless dropped. All dataset loaders run concurrently. |
| `rate_limits` | object | no | Named pools that inputs on the same backend share; see below. Default `{}`. |
| `assembly` | step list | no | Assembly steps, run in order over every loaded dataset before schema validation. Default `[]`. See below. |
| `rules` | step list | no | Business-logic rules, run in order on each portfolio's bundle; they never see other portfolios. Default `[]`. |
| `solve_order` | step | no | A solve-order step evaluated on each ruled bundle; its `Decimal` key replaces the `solve_order` column. Lower solves first. |
| `sides` | string | no | Which side the run trades: `both` (default), `buy`, or `sell`. Selects the side profile that supplies the decision variables, the trade identity, the tradable set, and the chain. `both`: `w`, `buy`, `sell` all variables, coupling through buys. `buy`: `w` alone with `w ≥ w0`, `buy = w − w0`, no `sell`; coupling through buys. `sell`: `w` alone with `w ≤ w0`, `sell = w0 − w`, no `buy`; coupling through sells. A term or constraint reading a side the run lacks is refused at `validate-config`. |
| `objective` | object | yes | `sense` (only `minimize`), `terms` (step list, at least one). |
| `constraints` | constraint list | no | Constraints. Default `[]`. Each is a step (bare name or `{"name", "params"}`) with two optional keys: `kind` (`function`, the only kind today) and `label` (unique among the run's constraints; defaults to the bare name; the verifier's report and the manifest key on it). The trade identity is not a constraint; `sides` supplies it, and `trade_balance` is refused by name. |
| `solve` | step | no | The solve step from `solvers.py`: `(request: SolveRequest[, params]) -> SolveResult`. Default `cvxpy`. A qualified name plugs in a firm's library or a pure function; see [how to replace the cvxpy solve](how-to-write-a-solve-step.md). |
| `solver` | object | no | `name` (default `CLARABEL`; one of `CLARABEL`, `OSQP`, `SCS`, `HIGHS`, `PIQP`, and installed — checked when the config resolves, on the client and on every worker), `options` (map of solver options passed verbatim to `Problem.solve`, default `{}`), `time_limit_s` (number > 0 or absent; mapped to `time_limit` for `CLARABEL`, `OSQP`, and `HIGHS` and to `time_limit_secs` for `SCS`; `PIQP` rejects it at resolve), `verbose` (default `false`). |
| `post_solve` | object | no | `violation_tol` (default `1e-6`), `objective_rel_tol` (`1e-5`), `objective_abs_tol` (`1e-9`); all > 0. |
| `sink` | step | yes | Where orders go. |
| `execution` | object | no | See below. Which cluster the run provisions and how many workers it has are settings, not config. |

## Step references

A step is either a bare string or an object:

```json
"cap_single_name"
{"name": "cap_single_name", "params": {"max_weight": "0.05"}}
```

`name` is a bare identifier resolved in the template module for its kind (`loaders.py`, `assembly.py`,
`rules.py`, `solve_order.py`, `terms.py`, `sinks.py`), or a qualified `package.module:function`. `params` (default `{}`) is validated
against the function's `params` annotation; a function without a `params` argument rejects any params.

| Kind | Signature |
|---|---|
| portfolios, dataset loader | `(request: LoadRequest[, params]) -> pd.DataFrame`, plain or `async def` |
| constraints loader | `(request: LoadRequest[, params]) -> dict[str, dict[str, object]]`, plain or `async def` |
| assembly step | `(frames: Frames[, params]) -> Frames` |
| rule | `(data: PortfolioData[, params]) -> PortfolioData` |
| solve-order step | `(data: PortfolioData[, params]) -> Decimal` — finite; lower solves first |
| objective term | `(x: DecisionVars, spec: ProblemSpec[, params][, chain: ChainState]) -> ObjectiveTerm` |
| constraint | `(x: DecisionVars, spec: ProblemSpec[, params][, chain: ChainState]) -> ConstraintSet` |
| sink | `(orders: pd.DataFrame, io: IoContext[, params]) -> tuple[Artifact, ...]` |

Optional arguments are recognized by name and must carry exactly the annotation shown. Only loaders may
be `async def`; every other kind runs synchronously. Declaring `chain` on a term or constraint is what
makes a portfolio wait for higher-priority portfolios that can buy a security it can buy too.

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

## `assembly[]`

Each entry is a step of kind `assembly`: `(frames: Frames[, params]) -> Frames`, where `Frames` is an
immutable mapping of dataset name to frame (see [the bundle reference](reference-portfolio-data.md)).
Steps run in order, once per run, after every loader has returned. A step that raises `ValueError` or
`KeyError` rejects the run as `assembly[i] <qualname>: <message>`. After the last step, `holdings`,
`universe`, `details`, and `targets` must exist and satisfy their schemas; every other dataset still
present is carried into each portfolio's bundle as an extra. The manifest records each step's
`rows_in`, `rows_out`, and `columns_added`.

### `join`

| Key | Type | Required | Description |
|---|---|---|---|
| `into` | string | yes | Dataset that receives the columns; any dataset. |
| `source` | string | yes | Dataset the columns come from; any dataset other than `into`. |
| `on` | string list | yes | Join keys present in both; their dtypes are aligned to `into` before merging. |
| `how` | `left` \| `inner` | no | Default `left`. |
| `cardinality` | `one_to_one` \| `one_to_many` \| `many_to_one` | yes | Enforced; a violation aborts the run. |
| `require_all_matched` | bool | no | Default `false`. When true, every row of `into` must find a match; unmatched keys are reported. |
| `columns` | string list | no | Source columns to bring, besides the keys. Default: every non-key column. |
| `rename` | object | no | Source column → name in `into`, applied to brought columns. Default `{}`. |
| `overwrite` | bool | no | Default `false`: a brought column that `into` already has is rejected. `true` replaces it. |

Unmatched rows of a Decimal (`object`) column are `None`.

### `union`

| Key | Type | Required | Description |
|---|---|---|---|
| `into` | string | yes | Name of the stacked result. If it already exists it must be one of `sources`. |
| `sources` | string list | yes | Datasets stacked in order. Shared columns must agree on dtype; a column some lack is null there, with `bool`/`int64`/`float64` promoted to `boolean`/`Int64`/`Float64`. |
| `source_column` | string | no | Column recording each row's source. Default: none. |
| `keep_sources` | bool | no | Default `false`: sources other than `into` are dropped. |

### `select`

| Key | Type | Required | Description |
|---|---|---|---|
| `dataset` | string | yes | Dataset to trim. |
| `columns` | string list | no | Keep exactly these, in this order. Exclusive with `drop`. |
| `drop` | string list | no | Columns to remove. Exclusive with `columns`. Default `[]`. |
| `rename` | object | no | Old name → new, applied after `columns`/`drop`. Default `{}`. |

### `drop`

| Key | Type | Required | Description |
|---|---|---|---|
| `datasets` | string list | yes | Datasets to discard; each must exist. |

## `execution`

| Key | Type | Required | Description |
|---|---|---|---|
| `on_error` | `fail_fast` \| `continue` | no | Default `fail_fast`: every lower-priority portfolio is recorded `skipped` after the first failure. `continue`: only the portfolios that depended on the failure are skipped, naming it. |
| `dependencies` | `overlap` \| `all` | no | Default `overlap`: a portfolio waits for every higher-priority portfolio whose buyable securities overlap its own. `all`: every higher-priority portfolio is a predecessor — the same answer, one line, for diagnosis. |

There is no execution mode. Every portfolio builds in a worker at once; solves are submitted with their
predecessors' contributions as dependencies and run where the build lives; outcomes are classified in
solve order. With no chain-aware term or constraint, no portfolio waits for another. The workers are the
Dask cluster the run provisions for itself — local worker processes on a laptop, pods on Kubernetes, or
a scheduler someone else runs — sized by the settings below; they are recorded in the manifest's
`settings` block and never affect the config hash.

## Shipped steps

Loaders: `csv` (`path`, `decimal_columns`, `utc_datetime_columns`, `dtypes`), `csv_per_portfolio`
(`directory`, `decimal_columns`, `utc_datetime_columns`, `dtypes`; reads `<directory>/<portfolio_id>.csv`
per portfolio under the input's rate limit), `parquet` (`path`, `decimal_columns`), `json_constraints`
(`path`). The column-typing params apply to extra datasets only; engine-known datasets are typed by
their schema. Assembly steps: `join`, `union`, `select`, `drop` (parameters above). Rules:
`cap_single_name` (`max_weight`), `add_zero_alpha`, `restrict_low_liquidity` (`min_adv_shares`).
Solve-order steps: `furthest_from_target_first`. Terms: `tracking_error`, `alpha` (`column`), `tax_cost`,
`transaction_cost` (`cost_bps`), each with `weight` (default `"1"`). Constraints:
`long_only`, `max_weight`, `cash_bounds`, `sector_bounds` (`tolerance`), `turnover_cap`,
`cumulative_adv_participation` (`chain`: `buy + sell ≤ adv_capacity` and `buy ≤ adv_capacity − predecessors' buys`).
Solve steps: `cvxpy` (default), `pro_rata_fill`. Sinks: `orders_to_parquet`, `orders_to_csv` (`subdir`, default `orders`).

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

All required unless stated; no defaults; an unknown `PORTFOLIO_OPTIMIZER_*` variable is an error. `run
--data-root`, `run --output`, and `run --max-workers` override the corresponding setting for one run.

| Variable | Values | Description |
|---|---|---|
| `PORTFOLIO_OPTIMIZER_OUTPUT_DIR` | path | Where `<run_id>/` directories are written. |
| `PORTFOLIO_OPTIMIZER_DATA_ROOT` | path | `request.data_root` for the shipped file loaders. |
| `PORTFOLIO_OPTIMIZER_LOG_LEVEL` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` | |
| `PORTFOLIO_OPTIMIZER_CLUSTER` | `local` \| `kubernetes` \| `auto` \| `tcp://host:port` | The Dask cluster the run provisions for itself (`local`: worker processes on this machine; `kubernetes`: pods through the Dask operator) or a scheduler to connect to. `auto` becomes `kubernetes` when `KUBERNETES_SERVICE_HOST` is set and `local` otherwise; the manifest records the resolved value. |
| `PORTFOLIO_OPTIMIZER_MIN_WORKERS` | integer ≥ 1, ≤ max | Workers provisioned before the load stage. |
| `PORTFOLIO_OPTIMIZER_MAX_WORKERS` | integer ≥ 1 | Workers after assembly. Every build and every solve is submitted at once; the scheduler runs what is ready. |
| `PORTFOLIO_OPTIMIZER_CLUSTER_TIMEOUT_S` | number > 0 | How long to wait, after assembly, for the first worker. |
| `PORTFOLIO_OPTIMIZER_WORKER_IMAGE` | image reference | Required when the cluster resolves to `kubernetes`: the image worker pods run, normally this run's own. |
| `PORTFOLIO_OPTIMIZER_IMAGE_DIGEST` | string | Optional; set by the platform. Part of every process's environment fingerprint and forwarded to worker pods. |

See [how to run on a cluster](how-to-run-on-a-cluster.md).
