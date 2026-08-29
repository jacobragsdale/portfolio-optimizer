# How to add a loader or a sink

Loaders and sinks are the engine's only I/O. This guide adds a loader that reads a dataset from a
database and a sink that submits orders to a trading system, keeping each testable without the real
dependency.

## Prerequisites

- You know the dataset's shape. Engine-known datasets (`portfolios`, `holdings`, `universe`, `details`,
  `targets`, `covariance`) must satisfy the schemas in `src/portfolio_optimizer/domain/schemas.py` after
  assembly; any other dataset only needs the columns your joins and rules use.
- The `constraints` dataset is a dict per portfolio, not a frame.

## Add a loader

### 1. Write the function in `src/portfolio_optimizer/loaders.py`

```python
class SqlParams(Params):
    query: str = Field(min_length=1)
    dsn_env: str = "PORTFOLIO_OPTIMIZER_DSN"


def holdings_from_sql(request: LoadRequest, params: SqlParams) -> pd.DataFrame:
    """Read holdings for the requested portfolios as of ``request.as_of``."""
    frame = my_gateway.query(params.query, portfolio_ids=request.portfolio_ids, as_of=request.as_of)
    return coerce_frame(frame, DATASET_SCHEMAS[request.dataset])
```

- `request: LoadRequest` carries `dataset` (the config key being loaded), `portfolio_ids` in solve
  order, `as_of`, `data_root`, and `run_id`.
- Return `pd.DataFrame` with every dtype declared. `coerce_frame` casts to the dataset's schema and turns
  money written as strings, ints, or floats into `Decimal` — do this at the read boundary, not later.
- The `constraints` loader returns `dict[str, dict[str, object]]` keyed by portfolio id; money inside may
  be strings (`"0.05"`), which the engine validates into `Decimal`.

Pass the database client in as a parameter of your gateway object rather than reaching for a global,
so a tier-4 contract test can call the real query and validate its shape with the production schema.

### 2. Name it in the config

```json
"datasets": {
  "holdings": {"loader": {"name": "holdings_from_sql", "params": {"query": "EXEC dbo.holdings_asof ?"}}},
  ...
}
```

Datasets that are not engine-known are combined through `assembly.joins`; each join declares its
cardinality (`one_to_one`, `one_to_many`, `many_to_one`) and can require every row to match. A join
never silently overwrites a column the target frame already has.

### 3. Check and test

`validate-config` confirms the signature. For the loader itself, write one contract test marked
`integration` that runs the real query and validates the result with the same schema production
uses — shape only, never values.

## Add a sink

### 1. Write the function in `src/portfolio_optimizer/sinks.py`

```python
class GatewaySinkParams(Params):
    account: str = Field(min_length=1)


def orders_to_gateway(orders: pd.DataFrame, io: IoContext, params: GatewaySinkParams) -> tuple[Artifact, ...]:
    """Submit the run's orders and return the gateway's acknowledgement as an artifact."""
    gateway: TradingGateway = build_gateway(params.account)
    return gateway.submit(orders, io.run_id)
```

- The engine calls the sink **once per run**, after every portfolio has been processed, with the orders
  of every solved portfolio concatenated and sorted by `(portfolio_id, security_id)`. It is not called
  when nothing solved.
- Return a tuple of `Artifact(path, sha256, size_bytes)` describing what was written or acknowledged;
  the manifest records them. For a network destination, write the acknowledgement to a file under
  `io.output_dir / io.run_id` and return that.
- `TradingGateway` is a `Protocol`; implement it in your own module and keep the network client behind
  it so the sink can be exercised against a fake in tests.
- A sink that raises is recorded in the manifest as a `sink` failure and the run exits with code 3.

### 2. Name it in the config

```json
"sink": {"name": "orders_to_gateway", "params": {"account": "SMA-TAX-01"}}
```

### 3. Test

Write the sink's test against a fake `TradingGateway`; assert on the artifacts returned and on what the
fake received. The shipped `orders_to_parquet` and `orders_to_csv` write atomically (temp file plus
rename) — do the same for any file destination so a crash never leaves a partial file behind.
