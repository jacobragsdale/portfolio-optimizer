# Reference: outputs, the manifest, and the CLI

## Command line

`portfolio-optimizer <command>`; every command exits with one of four codes.

| Code | Meaning |
|---|---|
| `0` | Every portfolio solved (or, for `verify`/`diff-manifests`, the check passed / no differences). |
| `1` | At least one portfolio failed, or nothing solved; for `verify`, verification failed; for `diff-manifests`, differences found. |
| `2` | Inputs rejected before anything ran: settings, config, resolution, schema, or usage errors. |
| `3` | Infrastructure: a file could not be read or written, the sink failed, or the cluster never produced a worker. |

| Command | Arguments | Description |
|---|---|---|
| `run CONFIG` | `--as-of DATETIME` (required: an ISO 8601 instant with a zone, e.g. `2026-08-28T00:00:00Z`), `--data-root PATH`, `--output PATH`, `--max-workers N` | Load, solve, verify, publish, and write the manifest. Prints the run id, the manifest path, and one line per portfolio naming the constraints that bind (`binding: ...`). |
| `validate-config CONFIG` | | Validate and resolve a config without loading data; prints the config hash and how dependencies will be derived, lists every resolved step with `[external]` markers, and every objective term as `name (Kind)` with a `[chain]` marker on a chain-aware kind. Constraints are loaded data and are not listed. |
| `verify` | `--manifest PATH --portfolio ID` | Reload the persisted spec, solution, and chain state and recompute every constraint the solve applied and every term in numpy — each through its kind's own residual or value — holding every residual to the `check.tolerance` the run used. Reads `sides` from the manifest's resolved config to pick the side profile whose identity checks apply, and the terms from the manifest's `terms`. Marks the checks the answer sits against `[binding]`. Never imports cvxpy. |
| `diff-manifests LEFT RIGHT` | | Print the first stage at which two runs diverge, overall and per portfolio. |
| `schema` | | Print the JSON Schema for run configs, over every step and term kind this environment can name; `> configs/run-config.schema.json` regenerates the checked-in file. |
| `steps` | | List every step a bare name can resolve to, by kind and with its parameter names — the template's and what installed packages publish — and every term and constraint kind with its fields. |

## Output directory

`<OUTPUT_DIR>/<run_id>/`

| Path | Content |
|---|---|
| `manifest.json` | The run manifest (below). |
| `orders/orders.parquet` | Written by the `orders_to_parquet` sink: every solved portfolio's orders, sorted by `(portfolio_id, security_id)`. Absent when no portfolio solved. |
| `problem_specs/<portfolio_id>.npz` | The `ProblemSpec` as arrays plus JSON metadata; no pickle. |
| `solutions/<portfolio_id>.npz` | The solve step's `w` and the profile's `buy`/`sell` split, with the objective (null when the step minimized nothing), status, solver and its version, solve time, iterations, spec hash, the records of the typed constraints the step applied, and the solver's duals. |
| `chain/<portfolio_id>.npz` | The chain state the solve depended on: `traded_shares`, the shares its predecessors traded on the side the run couples through (bought under `buy`, sold under `sell`), per security, zero where the portfolio cannot trade the name on that side; the metadata names the predecessors. Chain files written before 2026-08-29 carry the key `bought_shares` and no longer load; their hash is unchanged. |
| `failures/<portfolio_id>.txt` | The traceback of the exception that failed a portfolio, with the `run_id`, stage, and error above it. Written only for a failure an exception produced: a portfolio skipped after another's, a worker refused for its environment, and an input simply absent have no traceback and no file. A failure the run itself owns (`portfolio_id` `*`) is named for its stage — `failures/sink.txt`, `failures/cluster.txt`. This is normally the only surviving record of *where* a failure happened, since a worker's own stderr goes nowhere the run can read. |
| `trace.json` | The manifest's `timing` spans in the Chrome trace format: a row per worker process, a lane per portfolio, one complete event per span. Opens in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev). |

Files are written to a sibling temp file and renamed, so a crash leaves no partial output.

## Orders frame

| Column | dtype | Description |
|---|---|---|
| `portfolio_id`, `security_id` | `string` | Unique together. |
| `side` | `string` | `BUY` or `SELL`. |
| `quantity` | `Int64` > 0 | Whole shares, a multiple of the security's `lot_size`. |
| `reference_price` | `Decimal` | The universe price used to size the order. |
| `notional` | `Decimal` | `quantity × reference_price`, exact. |
| `target_weight` | `Float64` | The solved weight. |
| `unrounded_shares` | `Float64` | The signed share delta before rounding. |
| `spec_hash` | `string` | Hash of the problem the order came from. |
| `run_id` | `string` | |
| `as_of_date` | `datetime64[ns, UTC]` | |

Rounding: nearest whole share (half-even), then down to a lot multiple, then a sell is clamped to the
shares held and a buy to the room under the security's upper bound; orders below the style's
`min_trade_notional` are dropped.

## Manifest

`manifest.json` validates as `portfolio_optimizer.engine.manifest.RunManifest`. `manifest_sha256` is
the hash of the rest of the document; `load_manifest` refuses a document whose content does not match it.

| Field | Description |
|---|---|
| `run_id`, `run_name`, `tags`, `created_at_utc`, `as_of_date` | Identity. `run_name` and `tags` are the config's `run` block, kept out of the config hash; `as_of_date` is the run's `--as-of`; `created_at_utc` comes from the injected clock. |
| `git_sha`, `git_dirty` | The code revision, or `unknown` outside a repository. |
| `schedule` | The dependency graph the run derived: `coupling` (`none` when nothing read the chain — derived from the data and the steps, never configured — `overlap`, or `all`), `portfolios`, `edges`, `components` (independent groups that never waited on each other), `largest_component`, `critical_path` (the longest chain of solves that had to run one after another). Absent when the cluster never came up. |
| `cluster` | The backend's lifetime: `kind` (`inline` — this process, the default — `local`, `gateway`, `address`), `min_workers`, `max_workers`, `workers_ready` (workers joined when the first task could run; `1` under `inline`), `scheduler_address` (`null` under `inline`), `provision_started_at` (before the load stage), `first_worker_ready_at` (after assembly), `closed_at`. |
| `versions` | `python`, `cvxpy`, `numpy`, `pandas`, `solver`, `solver_version`, `packages`: the installed version of every distribution that supplied a step named outside the template modules (`{"my-firm-quant": "1.4.2"}`; a module no distribution provides is listed under its own name as `unknown`), and `workers[]`: every distinct environment that executed a task (`environment` — interpreter, libraries, solver, step packages, git sha, image digest — with `hosts` and `portfolios`). Normally one entry, equal to the run's own environment. |
| `config` | `path`, `sha256` of the canonical resolved config (the `run` block and `$schema` excluded), and `resolved` (the full config). |
| `settings` | Every setting the run used, including the worker counts and the step-package allowlist when one is set, with `cluster` resolved. |
| `terms` | Per configured objective term, in order, its record: `kind` and every field (`name`, `weight`, and the kind's own, `column` and `vector` for `linear`). `verify` parses these back through the kind registry, and every term's value in the report is keyed by its `name`. Constraints are recorded per portfolio instead, since they are loaded data a rule may change. |
| `datasets[]` | `name`, `loader_qualname` (`config` for a book written inline), `loader_source_sha256`, `params_sha256` (for an inline book, both hash the literal ids), `rows`, `columns`, `content_sha256`, `depends_on` (the effective dependencies, `portfolios` included for a per-portfolio dataset), `started_s` (seconds after the load stage began that the loader started: its wait on dependencies), `load_time_s` (wall-clock seconds the loader took), `batches` (calls the engine made: 1 for a global dataset, one per batch for a per-portfolio one, 0 for an inline book), `rejected` (portfolios whose batch failed, and which are failed at stage `load`). |
| `assembly[]` | Per assembly step, in order: `qualname`, `source_sha256`, `params_sha256`, `rows_in` and `rows_out` (rows per dataset before and after), `columns_added` (per dataset, the columns the step introduced). |
| `portfolios[]` | See below. A `sink` failure appears as an extra record with `portfolio_id: "*"`. |
| `artifacts[]` | `path`, `sha256`, `size_bytes` of every file written. |
| `timing[]` | Wall-clock spans over the run's stages: `name` (`load`, `dataset:<name>`, `assembly`, `cluster` — provisioning to first worker — `build`, `solve`, `sink`, with sub-phases as `build:slice`/`build:rules`/`build:spec` and `solve:chain`/`solve:solve`/`solve:verify`/`solve:orders`), `portfolio_id` (`null` for a run-scoped stage), `worker` (`host:pid`), `started_at_s` (Unix wall clock), `duration_s`. Observability, never identity: `diff-manifests` does not compare them, and instants are each process's own wall clock, so cross-host skew is visible rather than corrected. Written beside the manifest as `trace.json`, which opens in `chrome://tracing` or Perfetto. |
| `exit_code` | The code the run returned. |

### `portfolios[]`

| Field | Description |
|---|---|
| `portfolio_id`, `status` | `solved` or `failed`. Records are in solve order. |
| `solve_order` | The portfolio's solve-order key as a decimal string: the `solve_order` step's value, else the column's, else `0`. |
| `predecessors` | How many higher-priority portfolios this one waited for and folded into its chain. |
| `rules[]` | `qualname`, `source_sha256`, `params_sha256`, `rows_in`, `rows_out` per rule; the row counts cover `holdings`, `universe`, and every extra dataset in the bundle by name. |
| `constraints[]` | The typed constraints the solve step applied to this portfolio, after its rules, each as its record: `kind` and every field (`name`, `direction`, `scope`, `allow_current_weight`, `tolerance`, and the kind's own — `bounds`, `column`, `vector`). What `verify` re-checks. Empty for a step that reported none. |
| `problem_spec_sha256` | Hash of every array, flag, grouping, and scalar the solver saw. |
| `chain_inputs_sha256` | Hash of the chain state — the security ids and the shares predecessors traded on the coupled side, never which predecessors — so `overlap` and `all` runs hash alike. |
| `solve` | `solver`, `solver_version`, `status`, `iterations`, `objective_value`, `solve_time_s`, `duals` (per constraint name the step rendered, the largest dual value the solver reported — the shadow price of the limit, zero where it did not bind; `no_sells` under `buy` or `no_buys` under `sell` is the side profile's identity, the box included). For the `cvxpy` step, the cvxpy solver's name and the version of its distribution; the cvxpy version itself is in `versions.cvxpy`. A solve step that is not cvxpy records its qualified name as `solver` and its package version as `solver_version` unless it named a solver itself, `objective_value` is `null` when it minimized nothing, and `duals` is empty unless it reported some. |
| `check` | `tolerance` (the `violation_tol` every residual was held to), `max_violation`, `violated` (check names), `active` (the checks that bind — their residual within the tolerance of the bound — in report order, as `label/residual` where a constraint's residual name differs from its label: `ub`, `cash_floor/cash_limit`, `adv/cumulative_participation`), `objective_gap`, `objective_passed`, `passed`. |
| `drift` | `max_weight_error`, `tolerance`, `dropped_orders`, `passed` — the effect of rounding to shares. |
| `orders` | `count`, `sha256` (content hash excluding `run_id`), `gross_notional`. |
| `failure_stage`, `error` | For failed portfolios: `load` (a batch that failed, or no `details` row), `slice`, `build` (the bundle could not become a problem, or a constraint row or term could not apply to it), `solve`, `worker` (the worker died, or its environment fingerprint differed from the run's), `skipped` (the error names the failed predecessor, or the `fail_fast` cut-off), `sink`, or `cluster` (on the `*` record: no worker came up), and the error. The traceback behind the error, where there was an exception, is in `failures/` above — the manifest keeps the one-line summary. |

### Content hashes

Frame hashes are independent of row order (given the schema key), column order, and index; Decimals are
normalized; timestamps are compared as UTC instants; every column's dtype is part of the hash. Spec
hashes cover each vector's, column's, flag's, and grouping's name, shape, dtype, and bytes (with `-0.0`
normalized) plus the metadata — ids, `nav`, `as_of_date`, and every scalar. Source hashes are of the
function's source text.

### `diff-manifests` stages

Checked in order: `config`, `code` (git sha), `versions` (libraries, solver, step packages, and the set of worker environments — not the hosts they ran on), `datasets` (per dataset), `assembly` (the steps, their source and params hashes, and their `rows_out` and `columns_added`), then per portfolio
`status`, `rules`, `spec`, `solve` (objective value), `orders`. Only the first divergence per portfolio
is reported. The `schedule` record, `solve_order`, `predecessors`, and `timing` are never compared: the
schedule and the clock must not change results, and a `dependencies` change already shows at `config`.

Manifests written before the derived schedule existed carry an `execution_mode` field and no longer
load; ones written before the `timing` block hash without it and no longer load either.
