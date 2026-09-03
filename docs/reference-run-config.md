# Reference: the run config

A run config is one JSON document validated by `portfolio_optimizer.config.models.RunConfig`. Unknown
keys are rejected everywhere. Money, weights, and rates are written as JSON strings (`"0.05"`) and become
exact `Decimal` values; solver tolerances are JSON numbers. The instant the run is *as of* is not in the
document — it is `run --as-of` — and the config hash covers the wiring alone: `run.name`, `run.tags`,
and the `$schema` pointer are left out of it.

**Every key, its type, its default, and its description are in the generated JSON Schema**
(`configs/run-config.schema.json`, below) — including the parameters of every shipped step and the
fields of every shipped term kind. This page carries only what the schema cannot say: the signature each
kind of step must have and where a name is looked up, how datasets and their in-flight bounds behave at
load time, what an account's constraint rows and style limits look like (neither is part of the config),
and the environment variables. For what each block *means* to the engine and when it is consumed, see
[reading a run config](explanation-run-config.md).

## JSON Schema

`configs/run-config.schema.json` is a draft 2020-12 JSON Schema generated from the models
(`uv run portfolio-optimizer schema`), so it cannot disagree with what the engine accepts; a test fails
when the checked-in file is stale. It carries every field's description, a definition per kind of step
with the exact parameter schema of every step the generating environment can name — applied by
`if`/`then` on the step's `name` — and the objective as the union of every known term kind's own
schema. The checked-in file is generated over the template alone; an environment with installed step
packages generates a wider one. Custom (qualified) step names are allowed with any parameters, which
the engine validates at resolution time instead.

Ways to validate a config:

| Method | What it checks |
|---|---|
| `"$schema": "./run-config.schema.json"` at the top of the file | Live validation and completion in editors that honor `$schema` (VS Code, JetBrains). The key is accepted and ignored by the engine. |
| `uv run portfolio-optimizer validate-config CONFIG` | Everything the config can be checked for without data: the models, plus importing every step, checking signatures, validating params (including custom steps), parsing every objective term as its kind, checking the cvxpy solver, and rendering every term once against a one-security dummy spec under the run's order-flow profile. Constraints are loaded data, checked per portfolio at build. Prints the config hash, how dependencies between portfolios will be derived, one line per resolved step, and one per term as `name (Kind)`. The same resolution runs at the start of `run` and on every worker. |
| Any draft 2020-12 validator (`check-jsonschema`, `jsonschema`, `ajv`) against the schema file | The schema alone — suitable for CI pipelines that do not install the engine. |

`uv run portfolio-optimizer steps` lists every step a bare name can resolve to, by kind and with its
parameter names, and every term and constraint kind with its fields.

## Step references

A step is either a bare string or an object:

```json
"cap_single_name"
{"name": "cap_single_name", "params": {"max_weight": "0.05"}}
```

A bare `name` is looked up in the template module for its kind — `loaders.py`, `assembly.py`,
`rules.py`, `solve_order.py`, `engine/build.py`, `solvers.py`, `sinks.py` — and then among the steps
installed packages publish as entry points in the group `portfolio_optimizer.<kind>`
(`portfolio_optimizer.rule`, `portfolio_optimizer.loader`, ...); the template module wins a name both
have. A qualified `package.module:function` is imported from anywhere the engine and every worker can
import — or, when `PORTFOLIO_OPTIMIZER_STEP_PACKAGES` names an allowlist, from those top-level packages
alone (the template and published entry points are always allowed). `params` (default `{}`) is
validated against the function's `params` annotation; a function without a `params` argument rejects
any params.

| Kind | Signature |
|---|---|
| dataset loader (`portfolios` included) | `(request: LoadRequest[, params]) -> pd.DataFrame`, plain or `async def` |
| assembly step | `(frames: Frames[, params]) -> Frames` |
| rule | `(data: PortfolioData[, params]) -> PortfolioData` |
| solve-order step | `(data: PortfolioData[, params]) -> Decimal` — finite; lower solves first |
| build step | `(data: PortfolioData[, params]) -> ProblemSpec`; `standard` is the default, and its one param is `hold_breached_starts` (default `false`): a name already past a bound is held where it is, its bound moved to the current weight, instead of failing the portfolio as a start the order flow cannot trade out of |
| solve step | `(request: SolveRequest[, params]) -> SolveResult`; `cvxpy` is the default |
| sink | `(orders: pd.DataFrame, io: IoContext[, params]) -> tuple[Artifact, ...]` |

Engine arguments are recognized by name and must carry exactly the annotation shown. Only loaders may
be `async def`; every other kind runs synchronously. Objective terms and constraints are not steps but
*kinds* — strict pydantic models, below — and a kind declares on its class whether it reads the chain.

## `objective`

A list of term records, each an object whose `kind` names a model; the engine minimizes their sum.
Every kind carries `name` (unique among the run's terms; what the verifier's report and the manifest
key on) and `weight` (a string, default `"1"`; negative for a reward). The shipped kind:

| Kind | Fields | Meaning |
|---|---|---|
| `linear` | `column` (optional), `vector` (`w` default, `trade`, or a side the run has — `buy` or `sell`; both under `rebalance`, where they are convex and a reward on any of the three is refused) | `weight · columnᵀvector` over a per-security column of the spec — the exported `alpha`, the derived `tax_per_dollar` or `tcost_per_dollar`, any exported universe column. Omitted, every name counts once, so `trade` alone is a turnover penalty. |

A kind an installed package publishes in the entry-point group `portfolio_optimizer.term` is accepted
by name like a shipped one. Under the shipped `cvxpy` step the objective needs at least one term, and a
term that reads a decision vector the run's `order_flow` lacks, rewards a convex one under `rebalance`, or
is not convex, is refused — at resolve where the config shows it, at solve where only the data does.

## `solve`

The solve step and its own parameters. The shipped `cvxpy` step takes:

| Param | Type | Default | Meaning |
|---|---|---|---|
| `solver` | string | `CLARABEL` | The cvxpy solver: `CLARABEL`, `OSQP`, `SCS`, `HIGHS` (installed with cvxpy) or `PIQP` (the `piqp` extra). Must be one the adapter knows *and* installed — checked at resolve, on every worker, with no fallback. |
| `options` | object | `{}` | Passed verbatim to `Problem.solve(**options)`, e.g. `{"max_iter": 200}`. |
| `time_limit_s` | number > 0 | none | Wall-clock limit per solve, translated to the solver's own option; rejected at resolve for a solver without one (`PIQP`). |
| `verbose` | bool | `false` | The solver's own iteration log. |

`pro_rata_fill`, the other shipped step, takes no params; a step of your own takes whatever its
`Params` model declares. Any solve step other than the shipped one is opaque to the schedule: it may
read `request.chain` however it likes, so every portfolio couples through its whole tradable set.

## Datasets

### `depends_on`, `scope`, `batch_size`, and `max_in_flight`

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

### `max_in_flight`

A `per_portfolio` dataset may carry `max_in_flight`: how many of the batches `batch_size` cut the book
into the engine runs at once. A slot is held for the whole call, so `"batch_size": 1, "max_in_flight": 8`
is eight concurrent single-account requests and the rest queued behind them. Omitted, every batch runs
at once. It is per dataset — two inputs on one backend each get their own bound, which is the only
budget arithmetic the config does — and it is rejected on a `global` dataset, which is one call.

There is no rate limit beyond it: a source that needs pacing rather than a concurrency cap gets it in
the loader, where the client's own retry and backoff live.

## `execution`

`on_error` is `fail_fast` (default) or `continue`. `dependencies` is `overlap` (default) or `all`.
Under `overlap` a portfolio waits for the higher-priority portfolios whose tradable set, on the side the
run couples through, intersects what its own chain-reading constraint rows consume — the scopes of its
`participation_limit` rows, or its whole tradable set when anything opaque might read the chain (a
chain-aware term, a solve step other than the shipped one, a constraint frame with no `kind` column).
Under `all` every higher-priority portfolio is a predecessor: one line, the same answer, for diagnosis.
There is no `none`: a run in which nothing reads the chain is recognized from the data before any
build, no portfolio waits, and the manifest records `schedule.coupling: "none"`. There is no execution
mode either: every portfolio builds at once; solves are submitted with their predecessors'
contributions as dependencies and run where the build lives; outcomes are classified in solve order.
Where the work runs — this process, or a Dask cluster the run provisions for itself — is a setting
(below), recorded in the manifest's `settings` and `cluster` blocks and never part of the config hash.

## Shipped steps

Loaders: `load_portfolios`, `load_holdings` (one request per account in the batch, run together),
`load_universe`, `load_details` (a plain `def`: one query per batch of ids, run in a worker thread),
`load_constraints`, `load_mandates`, `load_trades`, `load_parameters` (`set_name`, default the dataset's own name).
Every one of them stands in for a service and takes `min_latency_s` and `max_latency_s`, which override
the wait that source is pretended to take; a real loader has neither. Assembly steps: `join`, `union`,
`select`, `drop`. Rules: `cap_single_name` (`max_weight`), `add_zero_alpha`, `restrict_low_liquidity`
(`dataset`, `key`; reads its threshold from a `name`/`value` extra dataset, by default
`buy_universe_parameters`/`min_adv_shares`), `restrict_to_mandate` (`dataset`, default `mandates`;
freezes every name whose sector is outside the account's mandate rows), `restrict_recent_trades`
(`dataset`, default `trades`; `window_days`, default 30; freezes every name the account traded within
the window of the run's as-of instant), `attach_universe_columns`
(`columns`; copies per-security columns from the universe onto holdings, matched on `security_id` —
default every column the universe carries beyond its schema). Solve-order steps:
`most_uninvested_first`. Build steps: `standard`. Solve steps: `cvxpy` (default), `pro_rata_fill`.
Sinks: `orders_to_parquet`, `orders_to_csv` (`subdir`, default `orders`).

## Constraints (the `constraints` dataset)

Which constraints bind an account is data, not config: there is no `constraints` key in a run config.
The dataset is per portfolio, like `holdings`, and optional — a run whose solve step needs none
declares no such dataset, and every portfolio gets an empty frame.

| Column | Type | Description |
|---|---|---|
| `portfolio_id` | string | The account the row applies to. |
| `kind` | string | The typed constraint model the row is: one of the kinds below, or one an installed package publishes in the group `portfolio_optimizer.constraint`. The engine reads it for the declaration it schedules by — whether the kind reads the chain and, through `scope`, which securities it couples through. |
| `label` | string | The constraint's `name`, unique among the account's rows; the verifier's report and the manifest key on it. May instead be given as `name` inside `params`, or in a `name` column. |
| `params` | string | A JSON object with the kind's fields. Money and weights are strings inside it, as in a config. |

A frame with no `kind` column is in a vocabulary the engine does not know: the shipped `cvxpy` step
refuses it, and a custom solve step reads it its own way. Where the column exists, every row is parsed
as its kind when the portfolio builds, after its rules; a malformed row, or one that names a column,
flag, scalar, or group the spec does not carry, fails that portfolio at stage `build`, before any solve
is scheduled on it, and the rest of the book runs.

Every kind carries:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | identifier | — | Unique per account. |
| `direction` | `<=` or `>=` | — | Which side of the bound the expression must stay on (`<` and `>` bind identically); an equality is two rows. |
| `scope` | string | none | A boolean flag column of the universe; set, the constraint touches only flagged securities, and a chain-reading kind couples only through them. |
| `allow_current_weight` | bool | `false` | The start policy: a bound the book already breaches loosens to the current value — hold it, do not worsen it — instead of failing the portfolio. Applies to `w`-shaped rows; a bound on `buy`, `sell`, or `trade` starts at zero. |
| `tolerance` | decimal ≥ 0 | `"0"` | Slack the verifier allows on the bound; the solver is held to the bound itself. |

A **bound** is written one of three ways: a literal (`"0.05"`); a per-account scalar the spec carries,
`{"scalar": "cash_ub"}` — any numeric column of the account's `details` row, the style limits included;
or, where the kind allows a per-security bound, a column of the spec, `{"column": "ub"}` — one of the
spec's own vectors or an exported universe column. The numbers stay in the data; the row says where.

| Kind | Fields | Bounds | Meaning |
|---|---|---|---|
| `weight_limit` | `vector` (default `w`), `bounds` | literal, scalar, or column | Per scoped security, `vector` against the bound: `{"direction": "<=", "bounds": {"scalar": "max_weight"}}` is a single-name cap; `{"direction": "<=", "vector": "buy", "bounds": "0", "scope": "excluded"}` is no new positions in the flagged names. |
| `group_limit` | `column`, `vector` (default `w`), `bounds` | one bound for every group, or a mapping of group to bound | The summed `vector` over each group of a string universe column, which the spec carries as a grouping: sector bands, country caps. A group the mapping does not name is unbounded by this row. |
| `exposure_limit` | `column`, `vector` (default `w`), `bounds` | literal or scalar | `column · vector` against the bound: a beta, a duration, a score. |
| `cash_limit` | `bounds` | literal or scalar | The cash left after the run, `1 − Σw`. `>=` on `cash_lb` and `<=` on `cash_ub` are the style's floor and cap. Takes no `scope`. |
| `turnover_limit` | `vector` (default `trade`), `bounds` | literal or scalar | The summed `vector` over the scope; `trade` against `{"scalar": "max_turnover"}` is the style's two-way turnover cap. |
| `participation_limit` | `bounds` (default `"1"`) | a literal multiple of `adv_capacity` | **Chain-aware.** Own trade in each scoped name within `bounds × adv_capacity` (the style's `max_adv_participation` times the day's volume, as a fraction of NAV), and the coupled side within what higher-priority portfolios' trades on that side left of it. Needs the universe's `adv_shares`; `direction` must be `<=`; no start policy. |

The per-security box `lb ≤ w ≤ ub` — the bounds the build derives from the style's `max_weight`, the
universe's optional `min_weight`/`max_weight` columns, and the `restricted` flag — is part of every
solve's trade identity, not a row; so is what the trade means under the run's `order_flow` — `w ≥ w0` with
`buy = w − w0`, `w ≤ w0` with `sell = w0 − w`, or `w` free in the box with `buy = max(w − w0, 0)` and
`sell = max(w0 − w, 0)` under `rebalance`. The box's own start policy is the build's:
`{"name": "standard", "params": {"hold_breached_starts": true}}` holds a name already past a bound
where it is, so an inflow over a name's cap is a feasible run that buys none of it rather than an
infeasible start.

The shipped example's rows for one account (`examples/data/constraints.csv`):

```csv
portfolio_id,kind,label,params
P1,cash_limit,cash_floor,"{""direction"": "">="", ""bounds"": {""scalar"": ""cash_lb""}}"
P1,cash_limit,cash_cap,"{""direction"": ""<="", ""bounds"": {""scalar"": ""cash_ub""}}"
P1,turnover_limit,turnover,"{""direction"": ""<="", ""bounds"": {""scalar"": ""max_turnover""}}"
P1,group_limit,sector_floor,"{""direction"": "">="", ""column"": ""sector"", ""bounds"": {""TECH"": ""0.5"", ""HEALTH"": ""0""}}"
P1,group_limit,sector_cap,"{""direction"": ""<="", ""column"": ""sector"", ""bounds"": {""TECH"": ""1"", ""HEALTH"": ""0.5""}}"
P1,participation_limit,adv,"{""direction"": ""<=""}"
```

Two rows of one kind produce residuals of the same name, so the verifier reports each as
`label/residual` (`sector_floor/group_limit`); the manifest records every row the solve applied, per
portfolio, as its JSON record.

## Style limits (columns of `details`)

Every bounded constraint reads its limits from the data, not from the config. The per-account scalars
are columns of the `details` frame, and the build exports every numeric column of the account's row —
these, `nav`, `cash`, the tax rates, and any further column the desk keeps on an account — as a spec
scalar a constraint row can name with `{"scalar": ...}`:

| Column | Type | Description |
|---|---|---|
| `max_weight` | decimal in (0, 1] | Single-name cap, folded into the spec's per-security `ub`. |
| `max_turnover` | decimal in [0, 2] | Two-way turnover as a fraction of NAV; what a `turnover_limit` row names. |
| `max_adv_participation` | decimal in [0, 1] | Fraction of each name's ADV the portfolio may trade; folded into the spec's `adv_capacity` column, which `participation_limit` scales. |
| `min_trade_notional` | decimal ≥ 0 | Orders below this notional are dropped. Not a constraint: the order step applies it after the solve. |
| `cash_lb` | decimal in [0, 1] | Lower bound on `1 − Σw`; what a `cash_limit` `>=` row names. |
| `cash_ub` | decimal in [0, 1] | Upper bound on `1 − Σw`; `cash_lb = cash_ub = 0` is full investment. Must be at least `cash_lb`. |

A limit that is not one scalar per account — a per-sector band — lives on its constraint row instead
(`group_limit` above).

## Environment

Every setting has a default a laptop can run with; an unknown `PORTFOLIO_OPTIMIZER_*` variable is an
error. `run --data-root`, `run --output`, and `run --max-workers` override the corresponding setting
for one run. A `.env` file is read only with `uv run --env-file .env ...`.

| Variable | Values | Default | Description |
|---|---|---|---|
| `PORTFOLIO_OPTIMIZER_OUTPUT_DIR` | path | `out` | Where `<run_id>/` directories are written. |
| `PORTFOLIO_OPTIMIZER_DATA_ROOT` | path | `.` | `request.data_root` for the shipped file loaders. |
| `PORTFOLIO_OPTIMIZER_LOG_LEVEL` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` | `INFO` | |
| `PORTFOLIO_OPTIMIZER_CLUSTER` | `inline` \| `local` \| `http(s)://gateway` \| `tcp://host:port` \| `tls://host:port` | `inline` | Where the work runs: this process, one task after another (`inline`); a Dask cluster the run provisions for itself (`local`: worker processes on this machine; an `http(s)://` address: a cluster asked of the Dask Gateway there); or a scheduler to connect to. |
| `PORTFOLIO_OPTIMIZER_MIN_WORKERS` | integer ≥ 1, ≤ max | `1` | Workers provisioned before the load stage (`local` and gateway). |
| `PORTFOLIO_OPTIMIZER_MAX_WORKERS` | integer ≥ 1 | `1` | Workers after assembly. Every build, and every solve whose predecessors are known, is submitted; the scheduler runs what is ready. |
| `PORTFOLIO_OPTIMIZER_CLUSTER_TIMEOUT_S` | number > 0 | `120` | How long to wait, after assembly, for the first worker. |
| `PORTFOLIO_OPTIMIZER_STEP_PACKAGES` | comma-separated package names | unset | The only top-level packages a qualified step name (`pkg.module:function`) may import from; unset, any importable module. The template's modules and steps published as entry points are always allowed. Recorded in the manifest's `settings` and applied on every worker. |
| `PORTFOLIO_OPTIMIZER_WORKER_IMAGE` | image reference | unset | Required for a gateway: the image its scheduler and worker pods run, normally this run's own. |
| `PORTFOLIO_OPTIMIZER_GATEWAY_PASSWORD` | string | unset | Required for a gateway: the password its simple authenticator accepts. Recorded in the manifest as `**********`. |
| `PORTFOLIO_OPTIMIZER_GATEWAY_PROXY_ADDRESS` | `tls://host:port` | unset | Where the gateway publishes scheduler traffic when that is not its own host and port. Unset, `dask-gateway` assumes the gateway's. |

See [how to run on a cluster](how-to-run-on-a-cluster.md).
