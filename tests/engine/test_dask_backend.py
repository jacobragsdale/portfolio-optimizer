"""Tier 4: the run owns a real (local) Dask cluster and produces the same answer as the process pool."""

from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from portfolio_optimizer.engine.environment import GitInfo
from portfolio_optimizer.engine.runner import EXIT_OK, run
from portfolio_optimizer.settings import ExecutionSettings
from tests.conftest import io_context, resolved_example_real

pytestmark = pytest.mark.integration

GIT = GitInfo(sha="0123456789abcdef", dirty=False)
NO_CHAIN_CONSTRAINTS = ["trade_balance", "long_only", "max_weight", "cash_bounds", "turnover_cap", "sector_bounds"]


@pytest.mark.parametrize("mode", ["parallel_build_sequential_solve", "parallel"])
def test_a_local_dask_cluster_matches_the_sequential_run(tmp_path: Path, mode: str) -> None:
    constraints = NO_CHAIN_CONSTRAINTS if mode == "parallel" else None
    overrides: dict[str, object] = {"constraints": constraints} if constraints is not None else {}
    sequential = run(
        resolved_example_real(execution={"mode": "sequential"}, sink="orders_to_parquet", **overrides),
        io_context(tmp_path / "seq", run_id="seq"),
        execution=ExecutionSettings(executor="process", max_workers=1),
        git=GIT,
        config_path="c.json",
        settings={},
    )
    dask = run(
        resolved_example_real(execution={"mode": mode}, sink="orders_to_parquet", **overrides),
        io_context(tmp_path / "dask", run_id="dask"),
        execution=ExecutionSettings(executor="dask", max_workers=2, cluster="local", min_workers=1, cluster_timeout_s=180.0),
        git=GIT,
        config_path="c.json",
        settings={},
    )
    assert dask.exit_code == EXIT_OK, [str(o) for o in dask.outcomes]
    for left, right in zip(sequential.solved, dask.solved, strict=True):
        assert left.spec.content_hash() == right.spec.content_hash()
        assert_frame_equal(left.orders.drop(columns=["run_id"]), right.orders.drop(columns=["run_id"]))
    cluster = dask.manifest.cluster
    assert cluster is not None
    assert (cluster.executor, cluster.kind, cluster.min_workers, cluster.max_workers) == ("dask", "local", 1, 2)
    assert cluster.scheduler_address is not None and cluster.scheduler_address.startswith("tcp://")
    assert cluster.workers_ready is not None and cluster.workers_ready >= 1
    (worker,) = dask.manifest.versions.workers
    assert worker.portfolios == 2
    assert worker.environment == sequential.manifest.versions.workers[0].environment
