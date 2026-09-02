# Working notes for agents

- **This is a first draft with no users.** Never compromise a design change to keep backwards
  compatibility — no deprecation shims, no legacy config forms, no dual code paths. Pick the right
  design and change every caller, config, doc, and test to match in the same commit.
- Verify with `uv run pytest` and `uv run pre-commit run --all-files` (ruff format/check, ty, uv lock,
  prettier for JSON). Regenerate the published schema after config-model changes:
  `uv run portfolio-optimizer schema > configs/run-config.schema.json`.
- The README's annotated `jsonc` block must parse equal to `configs/example_inflow.json`; a test enforces
  it, so update both together. `configs/example_outflow.json` is the same wiring for the outflow.
