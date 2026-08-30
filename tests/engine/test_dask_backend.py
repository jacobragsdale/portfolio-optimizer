"""Tier 2: the run provisions, scales, and tears down its own local cluster, and produces the hand-checked answer on it."""

from pathlib import Path

from portfolio_optimizer.engine.runner import EXIT_OK, RunContext, run
from portfolio_optimizer.settings import ExecutionSettings
from tests.conftest import resolved_example_real
from tests.engine.support import EXAMPLE_ORDERS_P1, GIT, io_context


def test_a_run_owned_local_cluster_solves_the_example(tmp_path: Path) -> None:
    context = RunContext(
        io=io_context(tmp_path / "dask", run_id="dask"), execution=ExecutionSettings(cluster="local", min_workers=1, max_workers=2, cluster_timeout_s=180.0), git=GIT, config_path="c.json", settings={}
    )
    report = run(resolved_example_real(sink="orders_to_parquet"), context)
    assert report.exit_code == EXIT_OK, [str(o) for o in report.outcomes]
    p1, p2 = report.solved
    assert p1.orders[["security_id", "side", "quantity"]].to_dict("records") == EXAMPLE_ORDERS_P1
    assert len(p2.orders) == 0 and p2.chain_state.predecessors == ("P1",)
    cluster = report.manifest.cluster
    assert cluster is not None
    assert (cluster.kind, cluster.min_workers, cluster.max_workers) == ("local", 1, 2)
    assert cluster.scheduler_address is not None and cluster.scheduler_address.startswith("tcp://")
    assert cluster.workers_ready is not None and cluster.workers_ready >= 1
    (worker,) = report.manifest.versions.workers
    assert worker.portfolios == 2
    assert report.manifest.schedule is not None and report.manifest.schedule.edges == 1
