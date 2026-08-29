# How to add a rule

A rule is business logic that runs between loading and the optimizer: it takes a portfolio's validated
data bundle and returns a new one. This guide adds a rule with parameters, wires it into a run config,
and tests it.

## Prerequisites

- The environment is installed (`uv sync --locked`) and you can run the example.
- You know which part of the bundle the rule changes: `holdings`, `universe`, `targets`, `extras`, or
  the `style` constraints. A rule that attaches a column to `holdings` or `universe` must give it the
  same dtype on both tables when both carry it; see [the bundle reference](reference-portfolio-data.md).

## 1. Write the function in `src/portfolio_optimizer/rules.py` — or in your package

A rule shared across desks belongs in a package installed in the environment (`uv add my-firm-quant`)
and is named `my_firm.rules:exclude_sector` in the config; everything below applies unchanged. A loose
module next to the config is not on the console script's import path, so install it or run with
`PYTHONPATH`.

```python
class ExcludeSectorParams(Params):
    sector: str = Field(min_length=1)


def exclude_sector(data: PortfolioData, params: ExcludeSectorParams) -> PortfolioData:
    """Freeze every name in ``sector`` at its current weight so no new money flows into it."""
    in_sector = data.universe["sector"] == params.sector
    universe = data.universe.assign(restricted=(data.universe["restricted"] | in_sector).astype("bool"))
    return data.with_changes(universe=universe)
```

The signature is the whole contract:

- `data: PortfolioData` — required, exactly this name and type.
- `params: <Params subclass>` — optional; the engine validates the JSON `params` object against it
  before any data loads. Money and weights arrive as `Decimal`, so write `Decimal` fields for them.
- `ctx: SolveContext` — optional; declare it only if the rule needs earlier portfolios' results
  (see step 4).
- Return `PortfolioData`, built with `data.with_changes(...)`. Construction re-validates every frame
  and the cross-frame invariants, so a rule cannot hand the optimizer an inconsistent bundle.

Keep rules pure: no I/O, no clock, no randomness. The pipeline records the function's source hash and
row counts before and after, so a rule that reads the world would make the manifest lie.

## 2. Name it in the run config

```json
"rules": [
  {"name": "exclude_sector", "params": {"sector": "TOBACCO"}},
  "add_zero_alpha"
]
```

Rules run in list order. A bare string is a rule without params. A qualified name such as
`"my_firm.rules:exclude_sector"` resolves a function outside this module — useful when a firm keeps
its rules in a separate package. Under a process executor that package must be importable by the
worker processes, exactly as any Python import.

## 3. Check it resolves

```bash
uv run --env-file .env portfolio-optimizer validate-config configs/my_run.json
```

A typo in the name, a parameter the model does not declare, a wrong annotation, or an unexpected
argument is reported here, with the function's qualified name and the reason.

## 4. If the rule needs earlier portfolios' results

Add `ctx: SolveContext`. `ctx.results` holds every `PortfolioResult` solved so far in this run, in
solve order; `avoid_cross_portfolio_wash_sales` in `rules.py` is a worked example. Two consequences,
both enforced at config load:

- The run's `execution.mode` must be `sequential` (rules run during build, and only the sequential
  mode builds with a live context).
- `execution.on_error` must be `fail_fast`; skipping a failed portfolio would silently change what
  later rules see.

## 5. Test it

Add a table-driven test to `tests/engine/test_pipeline.py` using the frame builders from
`tests/conftest.py`: one case at the boundary of the rule, one just past it, the empty input, and one
normal case. Assert on the returned bundle, not on how it was computed. If the rule is idempotent, say
so with a Hypothesis property as `test_restrict_low_liquidity_is_idempotent` does.

```bash
uv run pytest tests/engine/test_pipeline.py
```

## 6. Verify the whole pipeline still passes

```bash
uv run pre-commit run --all-files
uv run pytest
```

The convention test in `tests/test_sinks_settings_extensions.py` resolves every public function in
`rules.py`; a function that violates the contract fails there with the same message the resolver
would give at runtime.
