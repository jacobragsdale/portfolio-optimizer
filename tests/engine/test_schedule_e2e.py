"""Tier 2: the overlap schedule gives exactly what the line gives — orders, chain hashes, statuses — over seeded random buy universes and tying priorities.

Each seed builds a small book under ``tmp_path``: portfolios holding random subsets of five securities,
a per-portfolio ``buy_list`` dataset that a rule turns into the buyable set (the shape of a real
buy-universe filter), and ``solve_order`` values that tie. The run goes through the in-process backend
twice, with ``dependencies`` ``overlap`` and ``all``.
"""

import random
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from portfolio_optimizer.domain.results import PortfolioResult
from portfolio_optimizer.engine.runner import EXIT_OK, RunReport
from tests.conftest import SHIPPED_CONSTRAINTS, example_body
from tests.engine.fakes import LazyBackend, factory_for
from tests.engine.support import execute

SECURITIES = ("A", "B", "C", "D", "E")
PRICES = {"A": 100, "B": 50, "C": 10, "D": 20, "E": 40}
ALPHAS = {"A": "0.01", "B": "0.02", "C": "0.03", "D": "0.04", "E": "0.05"}
"""Distinct, so every portfolio wants to move toward its best buyable name and the schedule has contention to resolve."""


def synthetic_book(root: Path, rng: random.Random, portfolios: int = 4) -> None:
    """Write a random book: holdings and details for every account, per-portfolio buy lists, tying priorities; one shared universe."""
    root.mkdir()
    portfolio_ids = [f"P{index}" for index in range(1, portfolios + 1)]
    holdings = ["portfolio_id,security_id,quantity,avg_cost,acquired_on"]
    details = ["portfolio_id,name,state,st_tax_rate,lt_tax_rate,cash,nav,max_weight,max_turnover,max_adv_participation,min_trade_notional,cash_lb,cash_ub"]
    style = "1,2,0.25,0,0,0"  # the example's limits, which bind nothing here: the schedule is what this test is about
    buy_list = ["portfolio_id,security_id"]
    for portfolio_id in portfolio_ids:
        nav = 0
        for security in sorted(rng.sample(SECURITIES, k=rng.randint(1, 3))):
            quantity = rng.randint(1, 4) * 1000
            holdings.append(f"{portfolio_id},{security},{quantity},{PRICES[security]},2024-01-15T00:00:00Z")
            nav += quantity * PRICES[security]
        details.append(f"{portfolio_id},{portfolio_id} book,NY,0.40,0.20,0,{nav},{style}")
        buy_list.extend(f"{portfolio_id},{security}" for security in sorted(rng.sample(SECURITIES, k=rng.randint(1, 3))))
    (root / "holdings.csv").write_text("\n".join(holdings) + "\n")
    (root / "details.csv").write_text("\n".join(details) + "\n")
    (root / "portfolios.csv").write_text("\n".join(["portfolio_id,solve_order", *(f"{portfolio_id},{rng.randint(0, 1)}" for portfolio_id in portfolio_ids)]) + "\n")
    (root / "buy_list.csv").write_text("\n".join(buy_list) + "\n")
    (root / "buy_universe_parameters.csv").write_text("name,value\nmin_adv_shares,1000\n")
    (root / "global_parameters.csv").write_text("name,value\nrisk_aversion,2.5\n")
    universe_rows = (f"{security},{PRICES[security]},TECH,20000,1,false,{ALPHAS[security]}" for security in SECURITIES)
    (root / "universe.csv").write_text("\n".join(["security_id,price,sector,adv_shares,lot_size,restricted,alpha", *universe_rows]) + "\n")
    constraints = ["portfolio_id,name,label,params"]
    constraints.extend(f"{portfolio_id},{name}," for portfolio_id in portfolio_ids for name in SHIPPED_CONSTRAINTS)
    (root / "constraints.csv").write_text("\n".join(constraints) + "\n")


def run_book(tmp_path: Path, root: Path, dependencies: str, run_id: str) -> RunReport:
    """The example config over ``root`` with the ``buy_list`` dataset loaded and the rule that reads it appended."""
    example = example_body()
    example_datasets, example_rules = example["datasets"], example["rules"]
    assert isinstance(example_datasets, dict) and isinstance(example_rules, list)
    datasets = {**example_datasets, "buy_list": {"loader": "tests.steps:load_buy_list"}}
    rules = [*example_rules, "tests.steps:buy_only_listed"]
    return execute(tmp_path, backend_factory=factory_for(LazyBackend()), data_root=root, run_id=run_id, dependencies=dependencies, datasets=datasets, rules=rules)


@pytest.mark.parametrize("seed", range(6))
def test_the_overlap_schedule_reproduces_the_line(tmp_path: Path, seed: int) -> None:
    root = tmp_path / "book"
    synthetic_book(root, random.Random(seed))
    overlap = run_book(tmp_path, root, "overlap", "overlap")
    line = run_book(tmp_path, root, "all", "line")
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
