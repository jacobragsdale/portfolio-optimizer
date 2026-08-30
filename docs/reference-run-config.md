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
| `uv run portfolio-optimizer validate-config CONFIG` | Everything the config can be checked for without data: the models, plus importing every step, checking signatures, validating params (including custom steps), checking the solver, and constructing every term once under the run's side profile. Constraints are loaded data, so they are not checked here. Prints how dependencies between portfolios will be derived, then one line per resolved step. The same resolution runs at the start of `run` and on every worker. |
| Any draft 2020-12 validator (`check-jsonschema`, `jsonschema`, `ajv`) against the schema file | The schema alone — suitable for CI pipelines that do not install the engine. |

The schema cannot express one rule the models enforce: `as_of_date` must carry a time zone.
`validate-config` reports it.

## Top level

| Key | Type | Required | Description |
|---|---|---|---|
| `run` | object | yes | Run identity: `name` (non-empty), `as_of_date` (timezone-aware ISO-8601 timestamp), `tags` (string map, default `{}`). |
| `portfolios` | step or input | yes | Loader returning the `portfolios` frame (`portfolio_id`, optional `solve_order`); a bare step, or `{"loader": step[, "rate_limit": ...]}` to bound its source. `solve_order` is a priority: lower solves first, ties break on `portfolio_id`, values may repeat. |
| `datasets` | object | yes | Named inputs, each `{"loader": step[, "scope": ..., "batch_size": n, "rate_limit": name or bound]}`. `holdings`, `universe`, and `details` are required unless `assembly` is non-empty, in which case they may be produced by a step and are checked after assembly. `constraints` is engine-known but optional. Any other name is an extra dataset: visible to every assembly step, and carried into each portfolio's bundle as `data.extras` unless dropped. All dataset loaders run concurrently. |
| `rate_limits` | object | no | Named pools that inputs on the same backend share; see below. Default `{}`. |
| `assembly` | step list | no | Assembly steps, run in order over every loaded dataset before schema validation. Default `[]`. See below. |
| `rules` | step list | no | Business-logic rules, run in order on each portfolio's bundle; they never see other portfolios. Default `[]`. |
| `solve_order` | step | no | A solve-order step evaluated on each ruled bundle; its `Decimal` key replaces the `solve_order` column. Lower solves first. |
| `sides` | `both` \| `buy` \| `sell` | no | Which side the run trades; default `both`. `both`: `w`, `buy`, `sell` all variables, `w = w0 + buy − sell`, coupling through buys. `buy`: `w` alone with `w ≥ w0`, `buy = w − w0`, no `sell`; coupling through buys. `sell`: `w` alone with `w ≤ w0`, `sell = w0 − w`, no `buy`; coupling through sells. A term reading a side the run lacks is refused at `validate-config`; a constraint that does fails its portfolio at `solve`. See [how to run one side](how-to-run-one-side.md). |
| `objective` | object | yes | `sense` (only `minimize`), `terms` (step list, at least one). |
| `solve` | step | no | The solve step from `solvers.py`: `(request: SolveRequest[, params]) -> SolveResult`. Default `cvxpy`. A qualified name plugs in a firm's library or a pure function; see [how to replace the cvxpy solve](how-to-write-a-solve-step.md). |
| `solver` | object | no | `name` (default `CLARABEL`; one of `CLARABEL`, `OSQP`, `SCS`, `HIGHS`, `PIQP`, and installed — checked when the config resolves, on the client and on every worker), `options` (map of solver options passed verbatim to `Problem.solve`, default `{}`), `time_limit_s` (number > 0 or absent; mapped to `time_limit` for `CLARABEL`, `OSQP`, and `HIGHS` and to `time_limit_secs` for `SCS`; `PIQP` rejects it at resolve), `verbose` (default `false`). |
| `post_solve` | object | no | `violation_tol` (default `1e-6`; the one tolerance every residual is held to, identity checks and constraints alike), `objective_rel_tol` (`1e-5`), `objective_abs_tol` (`1e-9`); all > 0. |
| `sink` | step | yes | Where orders go. |
| `execution` | object | no | See below. Which cluster the run provisions and how many workers it has are settings, not config. |

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
| portfolios, dataset loader | `(request: LoadRequest[, params]) -> pd.DataFrame`, plain or `async def` |
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

## Rate limits

### `scope` and `batch_size`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `scope` | `"global"` \| `"per_portfolio"` | `"global"` | `global`: one call for the whole book, and the dataset is what the assembly steps see. `per_portfolio`: the engine cuts the ids into batches and calls the loader once per batch. A per-portfolio dataset is not passed to assembly — attach its columns in a rule. |
| `batch_size` | integer ≥ 1 | every id in one call | How many portfolios one call of a `per_portfolio` loader receives as `request.portfolio_ids`. `1` is a call per portfolio. Rejected on a `global` dataset, which never receives ids. |

`portfolios` is always global: it is the loader that produces the ids everything else is partitioned by.

A `per_portfolio` batch that fails fails only its own portfolios, which are recorded at stage `load`
and skipped; the rest of the book runs. A dataset *no* batch of which came back is the source being
down, and rejects the run. So does a required dataset that is missing, or one that violates its schema
— structural problems reject, coverage problems fail a portfolio.

### `rate_limit` and `rate_limits`

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
`universe`, and `details` must exist and satisfy their schemas; every other dataset still
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
| `dependencies` | `overlap` \| `all` \| `none` | no | Default `overlap`: a portfolio waits for every higher-priority portfolio whose tradable set — the securities it can trade on the side the run couples through: buyable (`ub > w0`) under `both` and `buy`, sellable (held, `lb < w0`) under `sell` — overlaps its own. `all`: every higher-priority portfolio is a predecessor — the same answer, one line, for diagnosis. `none`: nothing waits and the whole book solves at once, which is right when no constraint reads what others traded. Declared, not inferred: constraints are loaded data the engine does not interpret, so it cannot tell whether yours read the chain. |

There is no execution mode. Every portfolio builds in a worker at once; solves are submitted with their
predecessors' contributions as dependencies and run where the build lives; outcomes are classified in
solve order. Under `dependencies: none`, no portfolio waits for another. The workers are the
Dask cluster the run provisions for itself — local worker processes on a laptop, pods on Kubernetes, or
a scheduler someone else runs — sized by the settings below; they are recorded in the manifest's
`settings` block and never affect the config hash.

## Shipped steps

Loaders: `csv` (`path`, `dtypes`), `csv_per_portfolio` (`directory`, `dtypes`; reads
`<directory>/<portfolio_id>.csv` per portfolio under the input's rate limit), `parquet` (`path`,
`dtypes`). `dtypes` maps a column name to one kind — `string`, `Int64`,
`Float64`, `bool`, `decimal` (an exact `Decimal`), or `datetime_utc` (a timezone-aware timestamp) —
and applies to extra datasets only; engine-known datasets are typed by their schema, and a column no
kind is declared for arrives as pandas inferred it. Assembly steps: `join`, `union`, `select`, `drop` (parameters above). Rules:
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
