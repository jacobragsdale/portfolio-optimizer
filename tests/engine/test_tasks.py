"""Tier 2: the task functions a worker runs — a build from the shared data reports its environment, and a step package the worker cannot import is a worker failure."""

from pathlib import Path

from portfolio_optimizer.domain.results import PortfolioFailure
from portfolio_optimizer.domain.types import PortfolioId
from portfolio_optimizer.engine.backends import SharedRunData
from portfolio_optimizer.engine.environment import environment_for
from portfolio_optimizer.engine.load import assemble, load_datasets
from portfolio_optimizer.engine.tasks import BuildResult, build_task
from tests.conftest import EXAMPLE_DATA, example_config_real, resolved_example_real


def _shared() -> SharedRunData:
    resolved = resolved_example_real(sink="orders_to_parquet")  # every step from the template modules: a spawned worker cannot import tests.steps
    assembled = assemble(load_datasets(resolved, data_root=EXAMPLE_DATA, run_id="run-x"), resolved, run_id="run-x")
    return SharedRunData(assembled=assembled, config=resolved.config, config_sha256="example", run_id="run-x")


def test_build_task_slices_rules_and_builds_from_the_shared_data_and_reports_its_environment() -> None:
    shared = _shared()
    output = build_task(shared, PortfolioId("P1"))
    assert isinstance(output.outcome, BuildResult)
    assert output.outcome.spec.security_ids == ("A", "B", "C")
    assert output.outcome.solve_order == 0
    assert output.environment == environment_for(shared.config, cwd=Path.cwd(), image_digest=None)
    assert output.host
    missing = build_task(shared, PortfolioId("P9"))
    assert isinstance(missing.outcome, PortfolioFailure) and missing.outcome.stage == "slice"


def test_a_step_package_the_worker_cannot_import_is_a_worker_failure() -> None:
    shared = _shared()
    unresolvable = SharedRunData(assembled=shared.assembled, config=example_config_real(sink="no_such_package.sinks:publish"), config_sha256="other", run_id="run-y")
    output = build_task(unresolvable, PortfolioId("P1"))
    assert isinstance(output.outcome, PortfolioFailure) and (output.outcome.stage, output.outcome.error_type) == ("worker", "ConfigResolutionError")
    assert output.environment.packages == (("no_such_package", "unknown"),)
