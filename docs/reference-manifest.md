# Reference: outputs, the manifest, and the CLI

## Command line

`portfolio-optimizer <command>`; every command exits with one of four codes.

| Code | Meaning |
|---|---|
| `0` | Every portfolio solved (or, for `verify`/`diff-manifests`, the check passed / no differences). |
| `1` | At least one portfolio failed, or nothing solved; for `verify`, verification failed; for `diff-manifests`, differences found. |
| `2` | Inputs rejected before anything ran: settings, config, resolution, schema, or usage errors. |
| `3` | Infrastructure: a file could not be read or written, or the sink failed. |

| Command | Arguments | Description |
|---|---|---|
| `run CONFIG` | `--data-root PATH`, `--output PATH` | Load, solve, verify, publish, and write the manifest. Prints the run id, the manifest path, and one line per portfolio. |
| `validate-config CONFIG` | | Validate and resolve a config without loading data; lists every resolved step with `[external]` and `[ctx]`/`[chain]` markers. |
| `verify` | `--manifest PATH --portfolio ID` | Reload the persisted spec, solution, and chain state and recompute every shipped constraint and term in numpy. Never imports cvxpy. |
| `diff-manifests LEFT RIGHT` | | Print the first stage at which two runs diverge, overall and per portfolio. |
| `schema` | | Print the JSON Schema for run configs; `> configs/run-config.schema.json` regenerates the checked-in file. |

## Output directory

`<OUTPUT_DIR>/<run_id>/`

| Path | Content |
|---|---|
| `manifest.json` | The run manifest (below). |
| `orders/orders.parquet` | Written by the `orders_to_parquet` sink: every solved portfolio's orders, sorted by `(portfolio_id, security_id)`. Absent when no portfolio solved. |
| `problem_specs/<portfolio_id>.npz` | The `ProblemSpec` as arrays plus JSON metadata; no pickle. |
| `solutions/<portfolio_id>.npz` | The solver's `w`, `buy`, `sell`, objective, status, and versions. |
| `chain/<portfolio_id>.npz` | The cumulative shares from earlier portfolios the solve depended on. |

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
| `as_of` | `datetime64[ns, UTC]` | |

Rounding: nearest whole share (half-even), then down to a lot multiple, then a sell is clamped to the
shares held; orders below the style's `min_trade_notional` are dropped.

## Manifest

`manifest.json` validates as `portfolio_optimizer.engine.manifest.RunManifest`. `manifest_sha256` is
the hash of the rest of the document; `load_manifest` refuses a document whose content does not match it.

| Field | Description |
|---|---|
| `run_id`, `run_name`, `created_at_utc`, `as_of` | Identity. `created_at_utc` comes from the injected clock. |
| `git_sha`, `git_dirty` | The code revision, or `unknown` outside a repository. |
| `execution_mode` | |
| `versions` | `python`, `cvxpy`, `numpy`, `pandas`, `solver`, `solver_version`, and `packages`: the installed version of every distribution that supplied a step named outside the template modules (`{"my-firm-quant": "1.4.2"}`; a module no distribution provides is listed under its own name as `unknown`). |
| `config` | `path`, `sha256` of the canonical resolved config, and `resolved` (the full config). |
| `settings` | Non-secret settings the run used. |
| `terms`, `constraints` | Qualified name and params of every configured step, in order; `verify` uses these. |
| `datasets[]` | `name`, `loader_qualname`, `loader_source_sha256`, `params_sha256`, `rows`, `columns`, `content_sha256`, `load_time_s` (wall-clock seconds the loader took). |
| `assembly[]` | Per assembly step, in order: `qualname`, `source_sha256`, `params_sha256`, `rows_in` and `rows_out` (rows per dataset before and after), `columns_added` (per dataset, the columns the step introduced). |
| `portfolios[]` | See below. A `sink` failure appears as an extra record with `portfolio_id: "*"`. |
| `artifacts[]` | `path`, `sha256`, `size_bytes` of every file written. |
| `exit_code` | The code the run returned. |

### `portfolios[]`

| Field | Description |
|---|---|
| `portfolio_id`, `status` | `solved` or `failed`. |
| `rules[]` | `qualname`, `source_sha256`, `params_sha256`, `rows_in`, `rows_out` per rule; the row counts cover `holdings`, `universe`, `targets`, and every extra dataset in the bundle by name. |
| `problem_spec_sha256` | Hash of every array and scalar the solver saw. |
| `chain_inputs_sha256` | Hash of the chain state. |
| `solve` | `solver`, `solver_version`, `cvxpy_version`, `status`, `iterations`, `objective_value`, `solve_time_s`. |
| `check` | Tolerances, `max_violation`, `violated`, `objective_gap`, `objective_passed`, `unverified`, `passed`. |
| `drift` | `max_weight_error`, `tolerance`, `dropped_orders`, `passed` — the effect of rounding to shares. |
| `orders` | `count`, `sha256` (content hash excluding `run_id`), `gross_notional`. |
| `failure_stage`, `error` | For failed portfolios: `slice`, `build`, `solve`, `worker`, `skipped`, or `sink`, and the error. |

### Content hashes

Frame hashes are independent of row order (given the schema key), column order, and index; Decimals are
normalized; timestamps are compared as UTC instants; every column's dtype is part of the hash. Spec
hashes cover each array's name, shape, dtype, and bytes (with `-0.0` normalized) plus the scalar
metadata. Source hashes are of the function's source text.

### `diff-manifests` stages

Checked in order: `config`, `code` (git sha), `versions` (libraries, solver, and step packages), `datasets` (per dataset), `assembly` (the steps, their source and params hashes, and their `rows_out` and `columns_added`), then per portfolio
`status`, `rules`, `spec`, `solve` (objective value), `orders`. Only the first divergence per portfolio
is reported.
