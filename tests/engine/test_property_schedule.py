"""Property: the overlap schedule gives exactly what the line gives — orders, chain hashes, statuses — over random buy universes and tying priorities.

Each seed builds a small book under ``tmp_path``: portfolios holding random subsets of five securities,
a per-portfolio ``buy_list`` dataset that a rule turns into the buyable set (the shape of a real
buy-universe filter), and ``solve_order`` values that tie. The run goes through the in-process backend
twice, with ``dependencies`` ``overlap`` and ``all``.
"""

import json
import random
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from portfolio_optimizer.domain.results import PortfolioResult
from portfolio_optimizer.engine.backends import Backend
from portfolio_optimizer.engine.runner import EXIT_OK, RunReport, run
from portfolio_optimizer.settings import ExecutionSettings
from tests.conftest import EXAMPLE_CONFIG, EXAMPLE_DATA, execution_on, io_context, resolved_example_real
from tests.engine.test_backends import GIT, LazyBackend

SECURITIES = ("A", "B", "C", "D", "E")
PRICES = {"A": 100, "B": 50, "C": 10, "D": 20, "E": 40}


def synthetic_book(root: Path, rng: random.Random, portfolios: int = 4) -> None:
    """Write a random book: holdings, per-portfolio buy lists, tying priorities; universe, prices, targets, and styles shared."""
    root.mkdir()
    portfolio_ids = [f"P{index}" for index in range(1, portfolios + 1)]
    holdings = ["portfolio_id,security_id,quantity,avg_cost,acquired_on"]
    details = ["portfolio_id,name,state,st_tax_rate,lt_tax_rate,cash,nav,benchmark_id"]
    buy_list = ["portfolio_id,security_id"]
    for portfolio_id in portfolio_ids:
        nav = 0
        for security in sorted(rng.sample(SECURITIES, k=rng.randint(1, 3))):
            quantity = rng.randint(1, 4) * 1000
            holdings.append(f"{portfolio_id},{security},{quantity},{PRICES[security]},2024-01-15T00:00:00Z")
            nav += quantity * PRICES[security]
        details.append(f"{portfolio_id},{portfolio_id} book,NY,0.40,0.20,0,{nav},B1")
        buy_list.extend(f"{portfolio_id},{security}" for security in sorted(rng.sample(SECURITIES, k=rng.randint(1, 3))))
    style = json.loads((EXAMPLE_DATA / "constraints.json").read_text())["P1"]
    (root / "portfolios.csv").write_text("\n".join(["portfolio_id,solve_order", *(f"{portfolio_id},{rng.randint(0, 1)}" for portfolio_id in portfolio_ids)]) + "\n")
    (root / "holdings.csv").write_text("\n".join(holdings) + "\n")
    (root / "details.csv").write_text("\n".join(details) + "\n")
    (root / "buy_list.csv").write_text("\n".join(buy_list) + "\n")
    (root / "universe.csv").write_text("\n".join(["security_id,sector,adv_shares,lot_size,restricted", *(f"{security},TECH,20000,1,false" for security in SECURITIES)]) + "\n")
    (root / "prices.csv").write_text("\n".join(["security_id,price", *(f"{security},{PRICES[security]}" for security in SECURITIES)]) + "\n")
    (root / "targets.csv").write_text("\n".join(["benchmark_id,security_id,weight", *(f"B1,{security},0.2" for security in SECURITIES)]) + "\n")
    (root / "constraints.json").write_text(json.dumps(dict.fromkeys(portfolio_ids, style)))


def execute(tmp_path: Path, root: Path, dependencies: str, run_id: str) -> RunReport:
    example = json.loads(EXAMPLE_CONFIG.read_text())
    datasets = {**example["datasets"], "buy_list": {"loader": {"name": "csv", "params": {"path": "buy_list.csv", "dtypes": {"portfolio_id": "string", "security_id": "string"}}}}}
    rules = [*example["rules"], "tests.conftest:buy_only_listed"]
    resolved = resolved_example_real(execution={"dependencies": dependencies}, sink="orders_to_parquet", datasets=datasets, rules=rules)

    def factory(execution: ExecutionSettings, *, run_id: str) -> Backend:
        del execution, run_id
        return LazyBackend()

    return run(resolved, io_context(tmp_path / run_id, data_root=root, run_id=run_id), execution=execution_on("tcp://fake:8786"), git=GIT, config_path="c.json", settings={}, backend_factory=factory)


@pytest.mark.parametrize("seed", range(6))
def test_the_overlap_schedule_reproduces_the_line(tmp_path: Path, seed: int) -> None:
    root = tmp_path / "book"
    synthetic_book(root, random.Random(seed))
    overlap = execute(tmp_path, root, "overlap", "overlap")
    line = execute(tmp_path, root, "all", "line")
    assert overlap.exit_code == line.exit_code == EXIT_OK, [str(outcome) for outcome in (*overlap.outcomes, *line.outcomes)]
    assert [outcome.portfolio_id for outcome in overlap.outcomes] == [outcome.portfolio_id for outcome in line.outcomes]
    for left, right in zip(overlap.outcomes, line.outcomes, strict=True):
        assert isinstance(left, PortfolioResult) and isinstance(right, PortfolioResult)
        assert_frame_equal(left.orders.drop(columns=["run_id"]), right.orders.drop(columns=["run_id"]))
        assert left.chain_state.content_hash() == right.chain_state.content_hash()
        assert set(left.chain_state.predecessors) <= set(right.chain_state.predecessors)
        assert set(left.orders.loc[left.orders["side"] == "BUY", "security_id"]) <= {left.spec.security_ids[index] for index in range(left.spec.n) if left.spec.buyable[index]}
    assert overlap.manifest.schedule is not None and line.manifest.schedule is not None
    assert overlap.manifest.schedule.edges <= line.manifest.schedule.edges
    assert [record.solve_order for record in overlap.manifest.portfolios] == [record.solve_order for record in line.manifest.portfolios]
