"""Order sinks — yours to edit.

A sink is an ordinary function ``(orders: pd.DataFrame, io: IoContext, params: P) -> tuple[Artifact, ...]``
that publishes the run's orders somewhere — a file, a queue, a trading system — and returns what
it wrote so the manifest can record it. The engine calls the sink once per run with every
successful portfolio's orders, and only when at least one portfolio solved.

To send orders to a trading system, implement :class:`TradingGateway` in your own module and
write a sink that calls it; keep the network client behind that seam so the sink stays testable.
"""

from pathlib import Path
from typing import Protocol

import pandas as pd
from pydantic import Field

from portfolio_optimizer.domain.data import IoContext
from portfolio_optimizer.domain.results import Artifact
from portfolio_optimizer.domain.types import Params
from portfolio_optimizer.engine.hashing import file_sha256


class TradingGateway(Protocol):
    """What a trading-system client must offer for a sink to submit orders through it."""

    def submit(self, orders: pd.DataFrame, run_id: str) -> tuple[Artifact, ...]:
        """Submit ``orders`` and return a record (acknowledgement id, hash) of what was accepted."""
        ...


class FileSinkParams(Params):
    """Where under the run's output directory to write."""

    subdir: str = Field(default="orders", min_length=1)


def orders_to_parquet(orders: pd.DataFrame, io: IoContext, params: FileSinkParams) -> tuple[Artifact, ...]:
    """Write the orders as Parquet (dtypes preserved, Decimals as Arrow decimals), atomically."""
    target = io.output_dir / io.run_id / params.subdir / "orders.parquet"
    return (_write_atomically(target, lambda path: orders.to_parquet(path, index=False)),)


def orders_to_csv(orders: pd.DataFrame, io: IoContext, params: FileSinkParams) -> tuple[Artifact, ...]:
    """Write the orders as CSV for humans; the Parquet sink is the one to feed downstream systems."""
    target = io.output_dir / io.run_id / params.subdir / "orders.csv"
    return (_write_atomically(target, lambda path: orders.to_csv(path, index=False)),)


def _write_atomically(target: Path, write: "Writer") -> Artifact:
    """Write to a sibling temp file and rename, so a crash mid-write leaves no partial output."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.tmp")
    try:
        write(temp)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return Artifact(path=str(target), sha256=file_sha256(target), size_bytes=target.stat().st_size)


class Writer(Protocol):
    """Callable that writes a frame to ``path``."""

    def __call__(self, path: Path) -> None: ...
