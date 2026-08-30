"""Tier 2: the shipped sinks publish atomically and report the artifact they wrote."""

from decimal import Decimal
from pathlib import Path

import pandas as pd

from portfolio_optimizer.domain.frames import validate_frame
from portfolio_optimizer.domain.schemas import ORDERS
from portfolio_optimizer.engine.hashing import file_sha256
from portfolio_optimizer.sinks import FileSinkParams, orders_to_csv, orders_to_parquet
from tests.conftest import Frames
from tests.engine.support import io_context


def test_parquet_sink_writes_atomically_and_reports_the_artifact(tmp_path: Path, frames: Frames) -> None:
    orders = frames.orders({"security_id": "A"}, {"security_id": "B", "quantity": 7, "notional": Decimal(700)})
    (artifact,) = orders_to_parquet(orders, io_context(tmp_path), FileSinkParams())
    path = Path(artifact.path)
    assert path == tmp_path / "run-test" / "orders" / "orders.parquet"
    assert artifact.sha256 == file_sha256(path)
    assert artifact.size_bytes == path.stat().st_size
    assert sorted(p.name for p in path.parent.iterdir()) == ["orders.parquet"]
    validate_frame(pd.read_parquet(path), ORDERS)


def test_csv_sink_writes_a_readable_file(tmp_path: Path, frames: Frames) -> None:
    (artifact,) = orders_to_csv(frames.orders(), io_context(tmp_path), FileSinkParams(subdir="human"))
    assert Path(artifact.path).read_text().startswith("portfolio_id,security_id,side,quantity")
