# How to QA a deployment

A build in a QA environment has to answer one question: does this model meet every rule, for every
portfolio in the batch, and what was *not* tested? Every run writes the answer — the manifest, the
orders, the rows a check found in breach — and the CLI reads it. This guide runs a config and reads
the verdict.

## 1. Run with a known id

```bash
uv run portfolio-optimizer run CONFIG --as-of 2026-09-03T00:00:00Z --run-id qa-2026-09-03
```

The run prints one line per portfolio naming the constraints that bind, then one line per configured
check with its status. Exit codes: `0` every portfolio solved and every check passed; `1` a portfolio
or a check failed; `2` the inputs or the config were refused; `3` infrastructure. With `--run-id` the
directory is known before the run finishes, so a pipeline can read `out/qa-2026-09-03/manifest.json`
without parsing the output.

## 2. Read the verdict

The manifest is the whole verdict; [the reference](reference-manifest.md) lists every field. What to
read, in order:

- **Outcomes**: each portfolio's status and, when it failed, the stage and the traceback.
- **Not proven**: a check with `not_exercised` — the book never reached the rule — and a constraint
  kind no portfolio carried. A build is not proven against a rule the test book never exercises; this
  is what to fix in the book, not in the engine.
- **Margins**: `check.residuals` on every portfolio, each signed. A limit passed with `1e-7` of room is
  binding, one passed with `0.05` is not; `check.active` names the ones that bound.
- **Checks**: `checks[]` — `passed`, `failed` with the rows that broke it in `checks/<label>.csv`, or
  `not_exercised`.

`verify` recomputes one portfolio's constraints and terms from the persisted spec without cvxpy:

```bash
uv run portfolio-optimizer verify --manifest out/qa-2026-09-03/manifest.json --portfolio P1
```

## 3. Compare with the previous build

```bash
uv run portfolio-optimizer diff-manifests out/previous/manifest.json out/qa-2026-09-03/manifest.json
```

It prints the first stage at which two runs part — config, code, versions, datasets, assembly, then
each portfolio's status, rules, spec, solve, and orders — and every check whose outcome changed. Two
runs of one config over one snapshot on two builds should say `No differences`, or name exactly the
code hash; anything else is what changed, located.
