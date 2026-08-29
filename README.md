# portfolio-optimizer

A template repository for a JSON-driven, auditable portfolio-optimization engine built on pandas and cvxpy.
Clone it (or click **Use this template**), keep the engine, and write your own loaders, rules, objective
terms, constraints, and sinks as ordinary Python functions.

A run loads a list of portfolios, then holdings, the buy universe, portfolio details, style constraints, and
any extra datasets; combines them; applies business-logic rules; builds one convex problem per portfolio;
solves it; independently re-verifies the solution; rounds it to whole-share orders; publishes the orders;
and writes a manifest that lets anyone reproduce and audit the run.

## Quick start

```bash
uv sync --locked
cp .env.example .env
uv run --env-file .env portfolio-optimizer validate-config configs/example_run.json
uv run --env-file .env portfolio-optimizer run configs/example_run.json
```

The example runs two portfolios over three securities with a hand-checkable optimum; see
[the tutorial](docs/tutorial-first-run.md) for what to expect at each step.

## The one convention

A step is an ordinary function in a designated module; the JSON run config names it.

```python
# src/portfolio_optimizer/rules.py
class CapSingleNameParams(Params):
    max_weight: Decimal = Field(gt=0, le=1)


def cap_single_name(data: PortfolioData, params: CapSingleNameParams) -> PortfolioData: ...
```

```json
"rules": [{"name": "cap_single_name", "params": {"max_weight": "0.05"}}]
```

No decorators, no registries. Before any data loads, the engine imports every named function, checks its
signature, validates its `params` against the function's own model, and records its source hash in the
run manifest. Loaders, objective terms, constraints, and sinks follow the same rule.

Run configs are validated three ways: live in your editor through `"$schema": "./run-config.schema.json"`,
by `portfolio-optimizer validate-config` (which also imports and checks every step), and by any JSON Schema
validator against [`configs/run-config.schema.json`](configs/run-config.schema.json).

## Layout

| Path | Role |
|---|---|
| `src/portfolio_optimizer/{loaders,rules,terms,sinks}.py` | **Yours to edit.** Each ships worked, tested examples. |
| `src/portfolio_optimizer/engine/` | Loading and assembly, the rule pipeline, build, solve, cvxpy-free verification, orders, scheduling, manifest. Rarely edited. |
| `src/portfolio_optimizer/domain/` | Frame schemas, the per-portfolio data bundle, the pure-data problem spec and results. |
| `src/portfolio_optimizer/config/` | The run-config models and the step resolver. |
| `src/portfolio_optimizer/cvx/adapter.py` | The only module that imports cvxpy. |
| `configs/example_run.json`, `configs/run-config.schema.json`, `examples/data/` | The shipped example and the generated JSON Schema. |
| `docs/` | Tutorial, how-to guides, reference, and explanation. |

## Documentation

- [Tutorial: your first run](docs/tutorial-first-run.md)
- [How to add a rule](docs/how-to-add-a-rule.md)
- [How to add a loader or a sink](docs/how-to-add-a-loader-or-sink.md)
- [How to add an objective term or a constraint](docs/how-to-add-a-term.md)
- [Reference: the run config](docs/reference-run-config.md)
- [Reference: outputs, the manifest, and the CLI](docs/reference-manifest.md)
- [Explanation: how the engine is built and why](docs/explanation-architecture.md)

## Development

```bash
uv sync --locked
uv run pre-commit run --all-files   # ruff format, ruff check --fix, ty, uv lock, prettier for JSON
uv run pytest                       # unit, property, and smoke tests; warnings are errors
```

Python 3.12 or newer. Dependencies are locked in `uv.lock`; solver versions are recorded in every manifest.
