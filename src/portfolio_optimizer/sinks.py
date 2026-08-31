"""Order sinks — yours to edit.

A sink is an ordinary function ``(orders: pd.DataFrame, io: IoContext, params: P) -> tuple[Artifact, ...]``
that publishes the run's orders somewhere — a file, a queue, a trading system — and returns what
it wrote so the manifest can record it. The engine calls the sink once per run with every
successful portfolio's orders, and only when at least one portfolio solved. A sink that submits to a
trading system keeps the network client behind its own seam, so the sink itself stays testable.
"""

from pathlib import Path

import pandas as pd
from pydantic import Field

from portfolio_optimizer.domain.data import IoContext
from portfolio_optimizer.domain.results import Artifact
from portfolio_optimizer.domain.types import Params
from portfolio_optimizer.engine.files import write_atomically
from portfolio_optimizer.engine.hashing import file_sha256


class FileSinkParams(Params):
    """Where under the run's output directory to write."""

    subdir: str = Field(default="orders", min_length=1)


def orders_to_parquet(orders: pd.DataFrame, io: IoContext, params: FileSinkParams) -> tuple[Artifact, ...]:
    """Write the orders as Parquet (dtypes preserved, Decimals as Arrow decimals), atomically."""
    target = io.output_dir / io.run_id / params.subdir / "orders.parquet"
    return (_artifact(write_atomically(target, lambda path: orders.to_parquet(path, index=False))),)


def orders_to_csv(orders: pd.DataFrame, io: IoContext, params: FileSinkParams) -> tuple[Artifact, ...]:
    """Write the orders as CSV for humans; the Parquet sink is the one to feed downstream systems."""
    target = io.output_dir / io.run_id / params.subdir / "orders.csv"
    return (_artifact(write_atomically(target, lambda path: orders.to_csv(path, index=False))),)


def _artifact(target: Path) -> Artifact:
    """What the manifest records about a file a sink wrote."""
    return Artifact(path=str(target), sha256=file_sha256(target), size_bytes=target.stat().st_size)
