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

### 1. Write the function in `src/portfolio_optimizer/loaders.py` — or in your package

A loader shared across desks belongs in a package installed in the environment and is named
`my_firm.loaders:holdings_from_sql` in the config; the manifest records the package's version.

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
  order, `as_of`, `data_root`, `run_id`, and `rate_limiter` (see below).
- Return `pd.DataFrame` with every dtype declared. `coerce_frame` casts to the dataset's schema and turns
  money written as strings, ints, or floats into `Decimal` — do this at the read boundary, not later.
- The `constraints` loader returns `dict[str, dict[str, object]]` keyed by portfolio id; money inside may
  be strings (`"0.05"`), which the engine validates into `Decimal`.

Pass the database client in as a parameter of your gateway object rather than reaching for a global,
so a tier-4 contract test can call the real query and validate its shape with the production schema.

### Async loaders, fan-out, and rate limits

Every dataset loader runs concurrently once the portfolio list is known: an `async def` loader runs on
the engine's event loop, a plain `def` loader in a worker thread. Use `async def` for a source with an
async client; a blocking driver is fine as a plain function and still overlaps with the other loaders.

A source that answers one portfolio per call needs two things a large run cannot do without: fan-out
and a rate limit. Every input can be bounded on its own, because sources scale differently:

```json
"portfolios": {"loader": "portfolios_from_api", "rate_limit": {"max_in_flight": 1}},
"rate_limits": {"vendor_api": {"requests_per_second": 20, "burst": 40, "max_in_flight": 8}},
"datasets": {
  "holdings": {"loader": "holdings_from_api", "rate_limit": "vendor_api"},
  "universe": {"loader": "universe_from_api", "rate_limit": "vendor_api"},
  "details": {"loader": "details_from_sql", "rate_limit": {"max_in_flight": 32}}
}
```

`holdings` and `universe` name the same pool, so together they never exceed 20 requests per second or
8 in flight against the vendor. `details` has an inline bound of its own — the database takes 32
concurrent queries happily — and the portfolio list is held to one call at a time. The loader receives
whichever bound its input carries as `request.rate_limiter`:

```python
async def holdings_from_api(request: LoadRequest, params: ApiParams) -> pd.DataFrame:
    client = build_client(params)

    async def one(portfolio_id: PortfolioId) -> pd.DataFrame:
        return await client.holdings(portfolio_id, as_of=request.as_of)

    parts = await fan_out(request.portfolio_ids, one, limiter=request.rate_limiter)
    return coerce_frame(pd.concat(parts, ignore_index=True), DATASET_SCHEMAS[request.dataset])
```

`fan_out` starts every call at once and lets the limiter decide when each one runs; results come back
in portfolio order, and one failure cancels the rest and surfaces as an `ExceptionGroup`. From a plain
loader, wrap each call in `with request.rate_limiter.sync:` instead — it draws from the same pool. The
shipped `csv_per_portfolio` is this pattern with files in place of a client; copy its shape.

The manifest records how long each dataset took (`load_time_s`), and the run log reports each pool's
request count and total time spent waiting, so a slow run can be traced to the dataset and the limit
that paced it.

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

### 1. Write the function in `src/portfolio_optimizer/sinks.py` — or in your package

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
