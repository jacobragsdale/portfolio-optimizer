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
| `run CONFIG` | `--data-root PATH`, `--output PATH`, `--max-workers N` | Load, solve, verify, publish, and write the manifest. Prints the run id, the manifest path, and one line per portfolio. |
| `validate-config CONFIG` | | Validate and resolve a config without loading data; prints how dependencies will be derived and lists every resolved step with `[external]` and `[chain]` markers. |
| `verify` | `--manifest PATH --portfolio ID` | Reload the persisted spec, solution, and chain state and recompute every shipped constraint and term in numpy, holding every residual to the `check.tolerance` the run used. Reads `sides` from the manifest's resolved config to pick the side profile whose identity checks apply. Never imports cvxpy. |
| `diff-manifests LEFT RIGHT` | | Print the first stage at which two runs diverge, overall and per portfolio. |
| `schema` | | Print the JSON Schema for run configs; `> configs/run-config.schema.json` regenerates the checked-in file. |

## Output directory

`<OUTPUT_DIR>/<run_id>/`

| Path | Content |
|---|---|
| `manifest.json` | The run manifest (below). |
| `orders/orders.parquet` | Written by the `orders_to_parquet` sink: every solved portfolio's orders, sorted by `(portfolio_id, security_id)`. Absent when no portfolio solved. |
| `problem_specs/<portfolio_id>.npz` | The `ProblemSpec` as arrays plus JSON metadata; no pickle. |
| `solutions/<portfolio_id>.npz` | The solve step's `w` and the profile's `buy`/`sell` split, with the objective (null when the step minimized nothing), status, solver and its version, solve time, iterations, and spec hash. |
| `chain/<portfolio_id>.npz` | The chain state the solve depended on: `traded_shares`, the shares its predecessors traded on the side the run couples through (bought under `both` and `buy`, sold under `sell`), per security, zero where the portfolio cannot trade the name on that side; the metadata names the predecessors. Chain files written before 2026-08-29 carry the key `bought_shares` and no longer load; their hash is unchanged. |

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
shares held; orders below the style's `min_trade_notional` are dropped.

## Manifest

`manifest.json` validates as `portfolio_optimizer.engine.manifest.RunManifest`. `manifest_sha256` is
the hash of the rest of the document; `load_manifest` refuses a document whose content does not match it.

| Field | Description |
|---|---|
| `run_id`, `run_name`, `created_at_utc`, `as_of_date` | Identity. `created_at_utc` comes from the injected clock. |
| `git_sha`, `git_dirty` | The code revision, or `unknown` outside a repository. |
| `schedule` | The dependency graph the run derived: `coupling` (`none` when nothing read the chain, `overlap`, or `all`), `portfolios`, `edges`, `components` (independent groups that never waited on each other), `largest_component`, `critical_path` (the longest chain of solves that had to run one after another). Absent when the cluster never came up. |
| `cluster` | The cluster's lifetime: `kind` (`local`, `kubernetes`, `address`), `min_workers`, `max_workers`, `workers_ready` (workers joined when the first task could run), `scheduler_address`, `provision_started_at` (before the load stage), `first_worker_ready_at` (after assembly), `closed_at`. |
| `versions` | `python`, `cvxpy`, `numpy`, `pandas`, `solver`, `solver_version`, `packages`: the installed version of every distribution that supplied a step named outside the template modules (`{"my-firm-quant": "1.4.2"}`; a module no distribution provides is listed under its own name as `unknown`), and `workers[]`: every distinct environment that executed a task (`environment` — interpreter, libraries, solver, step packages, git sha, image digest — with `hosts` and `portfolios`). Normally one entry, equal to the run's own environment. |
| `config` | `path`, `sha256` of the canonical resolved config, and `resolved` (the full config). |
| `settings` | Every setting the run used, including the worker counts, with `cluster` resolved. |
| `terms`, `constraints` | Per configured step, in order: `qualname`, `params` (JSON-safe), and `label` — a term's bare name, a constraint's configured `label` (default its bare name). `verify` uses these, and every check in `check` is reported under its step's label. |
| `datasets[]` | `name`, `loader_qualname`, `loader_source_sha256`, `params_sha256`, `rows`, `columns`, `content_sha256`, `load_time_s` (wall-clock seconds the loader took), `batches` (calls the engine made: 1 for a global dataset, one per batch for a per-portfolio one), `rejected` (portfolios whose batch failed, and which are failed at stage `load`). |
| `assembly[]` | Per assembly step, in order: `qualname`, `source_sha256`, `params_sha256`, `rows_in` and `rows_out` (rows per dataset before and after), `columns_added` (per dataset, the columns the step introduced). |
| `portfolios[]` | See below. A `sink` failure appears as an extra record with `portfolio_id: "*"`. |
| `artifacts[]` | `path`, `sha256`, `size_bytes` of every file written. |
| `exit_code` | The code the run returned. |

### `portfolios[]`

| Field | Description |
|---|---|
| `portfolio_id`, `status` | `solved` or `failed`. Records are in solve order. |
| `solve_order` | The portfolio's solve-order key as a decimal string: the `solve_order` step's value, else the column's, else `0`. |
| `predecessors` | How many higher-priority portfolios this one waited for and folded into its chain. |
| `rules[]` | `qualname`, `source_sha256`, `params_sha256`, `rows_in`, `rows_out` per rule; the row counts cover `holdings`, `universe`, `targets`, and every extra dataset in the bundle by name. |
| `problem_spec_sha256` | Hash of every array and scalar the solver saw. |
| `chain_inputs_sha256` | Hash of the chain state — the security ids and the shares predecessors traded on the coupled side, never which predecessors — so `overlap` and `all` runs hash alike. |
| `solve` | `solver`, `solver_version`, `status`, `iterations`, `objective_value`, `solve_time_s`. For the `cvxpy` step, the cvxpy solver's name and the version of its distribution; the cvxpy version itself is in `versions.cvxpy`. A solve step that is not cvxpy records its qualified name as `solver` and its package version as `solver_version` unless it named a solver itself, and `objective_value` is `null` when it minimized nothing. |
| `check` | `tolerance` (the `violation_tol` every residual was held to), `max_violation`, `violated` (check names), `objective_gap`, `objective_passed`, `unverified`, `passed`. |
| `drift` | `max_weight_error`, `tolerance`, `dropped_orders`, `passed` — the effect of rounding to shares. |
| `orders` | `count`, `sha256` (content hash excluding `run_id`), `gross_notional`. |
| `failure_stage`, `error` | For failed portfolios: `slice`, `build`, `solve`, `worker` (the worker died, or its environment fingerprint differed from the run's), `skipped` (the error names the failed predecessor, or the `fail_fast` cut-off), `sink`, or `cluster` (on the `*` record: no worker came up), and the error. |

### Content hashes

Frame hashes are independent of row order (given the schema key), column order, and index; Decimals are
normalized; timestamps are compared as UTC instants; every column's dtype is part of the hash. Spec
hashes cover each array's name, shape, dtype, and bytes (with `-0.0` normalized) plus the scalar
metadata. Source hashes are of the function's source text.

### `diff-manifests` stages

Checked in order: `config`, `code` (git sha), `versions` (libraries, solver, step packages, and the set of worker environments — not the hosts they ran on), `datasets` (per dataset), `assembly` (the steps, their source and params hashes, and their `rows_out` and `columns_added`), then per portfolio
`status`, `rules`, `spec`, `solve` (objective value), `orders`. Only the first divergence per portfolio
is reported. The `schedule` record, `solve_order`, and `predecessors` are never compared: the schedule
must not change results, and a `dependencies` change already shows at `config`.

Manifests written before the derived schedule existed carry an `execution_mode` field and no longer load.
