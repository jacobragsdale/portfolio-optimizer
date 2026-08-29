# How to add security analytics columns to holdings and the universe

Most assembly work is attaching per-security analytics — scores, betas, liquidity buckets, vendor
flags, fair values — to the two tables the optimizer consumes: `holdings` (what each portfolio owns)
and `universe` (what it may buy). This guide loads an analytics dataset, attaches its columns to both
tables, computes a derived column, attaches a per-portfolio flag, and checks that the two tables stack
into one optimizer frame cleanly.

## Prerequisites

- The environment is installed (`uv sync --locked`) and the example runs.
- You know the analytics source: its columns, which column identifies a security, and whether it has
  one row per security (global) or one row per portfolio and security (per-portfolio).
- Two facts about the tables, from [the bundle reference](reference-portfolio-data.md): both `holdings`
  and `universe` accept any columns beyond their schemas, and a column present on both must have the
  same dtype on both, because `PortfolioData.optimizer_frame()` stacks the two tables and refuses a
  mismatch. Decide each column's dtype once, at the loader, and let every step carry it unchanged.

## 1. Load the analytics dataset, typed

Declare it under `datasets` with any name that is not engine-known. Type every column at the loader:
the key as `string`, money and rates as `decimal_columns`, statistical values as `Float64`, labels as
`string`, flags as `boolean`.

```json
"datasets": {
  "...": "the required datasets",
  "analytics": {
    "loader": {
      "name": "csv",
      "params": {
        "path": "analytics.csv",
        "dtypes": {"security_id": "string", "score": "Float64", "liquidity_bucket": "string"},
        "decimal_columns": ["fair_value"]
      }
    }
  }
}
```

From a database or an API, [write a loader](how-to-add-a-loader-or-sink.md) that returns the frame
already typed. Whatever dtypes leave the loader are the dtypes that land on `holdings` and `universe`.

## 2. Attach the columns with `join` steps

```json
"assembly": [
  {"name": "join", "params": {"into": "universe", "source": "analytics", "on": ["security_id"],
                              "cardinality": "one_to_one", "require_all_matched": true}},
  {"name": "join", "params": {"into": "holdings", "source": "analytics", "on": ["security_id"],
                              "cardinality": "many_to_one"}},
  {"name": "drop", "params": {"datasets": ["analytics"]}}
]
```

Read each join as a claim the engine checks:

- `universe` gets `one_to_one` with `require_all_matched`: every buyable name has exactly one analytics
  row, or the run is rejected naming the unmatched securities.
- `holdings` gets `many_to_one` without `require_all_matched`: one analytics row serves every portfolio
  that holds the name, and a held name outside coverage keeps its row with nulls in the new columns.
  Add `"require_all_matched": true` here too if a held name without analytics should stop the run.
- Both joins bring the same source columns, so the dtypes agree on both tables by construction. Use
  `"columns": ["score"]` to bring a subset and `"rename": {"score": "vendor_score"}` to rename it, the
  same way on both joins.
- `drop` discards the source once it has been used; otherwise it is carried into every portfolio's
  bundle as `data.extras["analytics"]`, which costs memory in every worker process for nothing.

Check the config resolves, then run and read the manifest:

```bash
uv run --env-file .env portfolio-optimizer validate-config configs/my_run.json
uv run --env-file .env portfolio-optimizer run configs/my_run.json
```

Each assembly step's record in `manifest.json` lists `columns_added` per dataset and row counts before
and after; the first join should show `{"universe": ["score", "liquidity_bucket", "fair_value"]}`.

## 3. Compute a derived column in a custom step

When the analytic is computed rather than loaded, write an assembly step. It sees every dataset by
name and returns a new `Frames`; the engine records its source hash and what it added. This one
standardizes a column against the buy universe's distribution and writes the result to both tables,
so a held name outside the universe gets a z-score on the same scale:

```python
# src/portfolio_optimizer/assembly.py — or my_firm.assembly:zscore_against_universe
class ZScoreParams(Params):
    column: str = Field(min_length=1)
    output: str = Field(min_length=1)


def zscore_against_universe(frames: Frames, params: ZScoreParams) -> Frames:
    """Standardize ``column`` on holdings and universe with the universe's mean and standard deviation."""
    reference = frames["universe"][params.column].astype("Float64")
    mean, std = float(reference.mean()), float(reference.std())
    if not std > 0.0:
        msg = f"{params.column} has no dispersion in the universe (std={std}); a z-score is undefined"
        raise ValueError(msg)

    def standardized(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(**{params.output: ((frame[params.column].astype("Float64") - mean) / std).astype("Float64")})

    return frames.with_frame("universe", standardized(frames["universe"])).with_frame("holdings", standardized(frames["holdings"]))
```

```json
{"name": "zscore_against_universe", "params": {"column": "score", "output": "score_z"}}
```

Three habits keep custom steps safe:

- Cast the output explicitly (`.astype("Float64")`) on every table you write it to; a computed column
  otherwise takes whatever dtype pandas infers, and it may differ between the tables.
- Raise `ValueError` with a specific message when a precondition fails. The engine reports it as
  `assembly[2] my_firm.assembly:zscore_against_universe: <message>` and rejects the run before anything
  is solved.
- Read datasets with `frames["name"]`; a missing name raises with the list of what exists.

Place the step after the joins that supply its inputs; steps run in list order.

## 4. Attach per-portfolio analytics with an extra dataset and a rule

A dataset with one row per portfolio and security — a mandate's exclusion list, per-portfolio lot
data — cannot be joined into `universe`, which has no portfolio dimension. Leave it as an extra
dataset with a `portfolio_id` column and do not `drop` it: the engine carries it into each portfolio's
bundle as `data.extras["<name>"]`, already reduced to that portfolio's rows. A rule then attaches it:

```python
# src/portfolio_optimizer/rules.py
def apply_mandate_exclusions(data: PortfolioData) -> PortfolioData:
    """Flag every universe name the mandate excludes; the flag is ``boolean`` so it stacks with holdings."""
    excluded = data.extras["mandate_exclusions"][["security_id"]].assign(mandate_excluded=True)
    universe = data.universe.merge(excluded, on="security_id", how="left", validate="one_to_one")
    universe = universe.assign(mandate_excluded=universe["mandate_excluded"].astype("boolean").fillna(False))
    return data.with_changes(universe=universe)
```

```json
"datasets": {"mandate_exclusions": {"loader": {"name": "csv", "params": {"path": "exclusions.csv", "dtypes": {"portfolio_id": "string", "security_id": "string"}}}}},
"rules": ["apply_mandate_exclusions"]
```

`with_changes` re-validates the bundle, so a rule that produced a conflicting dtype fails at that
portfolio with the column named, not later in the optimizer.

## 5. Check the optimizer frame

The proof that the two tables are compatible is the frame itself. From a rule, a test, or a notebook:

```python
frame = data.optimizer_frame()
frame[["source", "security_id", "quantity", "price", "score", "score_z", "mandate_excluded"]]
```

Rows are the holdings followed by the universe, tagged in `source`; a name that is both held and
buyable appears twice. Columns one side lacks are null on that side, promoted to the nullable dtype
where needed (`bool` to `boolean`, `int64` to `Int64`, `float64` to `Float64`). The failure signal is
raised on construction of every bundle, so you will see it at stage `slice` or at the rule that caused it:

```text
portfolio 'P1': holdings and universe disagree on column 'score': holdings has dtype 'Float64', universe has 'float64'
```

Fix it where the column was created — the loader's `dtypes`, or the `.astype` in the step — not by
casting in the consumer.

## 6. Feed the shipped optimizer, if you use it

The shipped build (`engine/build.py`) is aligned to the universe: it exports every numeric universe
column the schema does not declare into `spec.columns`, where a term reads it with
`spec.column("score_z")`, and every boolean column into `spec.flags`, where a term reads it as a real
boolean mask with `spec.flag("mandate_excluded")` (see [how to add a term](how-to-add-a-term.md)).
Holdings' analytics columns are not exported, because the build has no row for a name that is not in
the universe. A custom build
that consumes `optimizer_frame()` directly sees both tables' columns and is not subject to this.

## Verify

- `validate-config` lists each assembly step with its qualified name.
- The manifest's `assembly[]` records name the columns each step added; `diff-manifests` reports
  `assembly:` when a step, its parameters, or its effect on the datasets changed between two runs.
- A rule audit's `rows_in`/`rows_out` includes every extra dataset the bundle carried, so an extra
  that was meant to be dropped shows up there by name.
