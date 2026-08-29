"""Tier 1/2: the shipped rules (boundary, empty, normal) and the pipeline's provenance and guards."""

import json
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

from portfolio_optimizer.config.models import StepSpec
from portfolio_optimizer.config.resolve import resolve_step
from portfolio_optimizer.domain.results import ChainState, ConstraintReport, DriftReport, PortfolioResult, Solution, SolveContext, SolveStatus
from portfolio_optimizer.engine.pipeline import RuleError, apply_rules
from portfolio_optimizer.rules import CapSingleNameParams, LiquidityParams, add_zero_alpha, avoid_cross_portfolio_wash_sales, cap_single_name, restrict_low_liquidity
from tests.conftest import Factories, Frames


def step(name: str, **params: object) -> object:
    return resolve_step(StepSpec.model_validate_json(json.dumps({"name": name, "params": params})), "rule")


# --- cap_single_name ---


@pytest.mark.parametrize(("style_limit", "param", "expected"), [("0.10", "0.05", "0.05"), ("0.10", "0.10", "0.10"), ("0.10", "0.20", "0.10")])
def test_cap_single_name_only_tightens(make: Factories, style_limit: str, param: str, expected: str) -> None:
    data = make.portfolio_data(style=make.style(max_weight=Decimal(style_limit)))
    result = cap_single_name(data, CapSingleNameParams(max_weight=Decimal(param)))
    assert result.style.max_weight == Decimal(expected)
    assert data.style.max_weight == Decimal(style_limit)


# --- add_zero_alpha ---


def test_add_zero_alpha_adds_a_float_column_only_when_missing(make: Factories, frames: Frames) -> None:
    added = add_zero_alpha(make.portfolio_data())
    assert str(added.universe["alpha"].dtype) == "Float64"
    assert added.universe["alpha"].tolist() == [0.0, 0.0, 0.0]
    universe = frames.three_security_universe().assign(alpha=np.array([0.1, 0.2, 0.3]))
    universe["alpha"] = universe["alpha"].astype("Float64")
    kept = add_zero_alpha(make.portfolio_data(universe=universe))
    assert kept.universe["alpha"].tolist() == [0.1, 0.2, 0.3]


def test_add_zero_alpha_on_an_empty_universe(make: Factories, frames: Frames) -> None:
    data = make.portfolio_data(holdings=frames.holdings().iloc[0:0], universe=frames.universe().iloc[0:0], targets=frames.targets().iloc[0:0])
    assert add_zero_alpha(data).universe["alpha"].tolist() == []


# --- restrict_low_liquidity ---


@pytest.mark.parametrize(("threshold", "restricted"), [(100_000, [False, False, False]), (100_001, [False, False, True]), (1_000_001, [True, True, True])])
def test_restrict_low_liquidity_freezes_names_strictly_below_the_threshold(make: Factories, threshold: int, restricted: list[bool]) -> None:
    result = restrict_low_liquidity(make.portfolio_data(), LiquidityParams(min_adv_shares=threshold))
    assert result.universe["restricted"].tolist() == restricted
    assert str(result.universe["restricted"].dtype) == "bool"


def test_restrict_low_liquidity_keeps_names_already_restricted(make: Factories, frames: Frames) -> None:
    universe = frames.three_security_universe()
    universe.loc[0, "restricted"] = True
    result = restrict_low_liquidity(make.portfolio_data(universe=universe), LiquidityParams(min_adv_shares=0))
    assert result.universe["restricted"].tolist() == [True, False, False]


@given(threshold=st.integers(min_value=0, max_value=2_000_000))
@settings(deadline=None, max_examples=25)
def test_restrict_low_liquidity_is_idempotent(threshold: int) -> None:
    from tests.conftest import make_portfolio_data  # hypothesis tests cannot take fixtures

    params = LiquidityParams(min_adv_shares=threshold)
    once = restrict_low_liquidity(make_portfolio_data(), params)
    twice = restrict_low_liquidity(once, params)
    assert_frame_equal(once.universe, twice.universe)


# --- avoid_cross_portfolio_wash_sales ---


def prior_context(make: Factories, frames: Frames, *sold: str) -> SolveContext:
    spec = make.spec()
    solution = Solution(
        w=spec.w0, buy=np.zeros(3), sell=np.zeros(3), objective=0.0, status=SolveStatus.OPTIMAL, solver="X", solver_version="0", cvxpy_version="0", solve_time_s=0.0, iterations=None, spec_hash="h"
    )
    report = ConstraintReport(checks=(), objective_terms=(), recomputed_objective=0.0, solver_objective=0.0, objective_gap=0.0, objective_passed=True, unverified=())
    rows = [{"portfolio_id": "P0", "security_id": security, "side": "SELL", "quantity": 1, "reference_price": Decimal(1), "notional": Decimal(1)} for security in sold]
    orders = frames.orders(*rows) if rows else frames.orders().iloc[0:0]
    return SolveContext().with_result(PortfolioResult("P0", spec, solution, report, orders, (), ChainState.empty(spec.security_ids), DriftReport(0.0, 0.0, 0)))


def test_wash_sale_rule_caps_names_sold_earlier_at_their_current_weight(make: Factories, frames: Frames) -> None:
    result = avoid_cross_portfolio_wash_sales(make.portfolio_data(), prior_context(make, frames, "A", "C"))
    caps = dict(zip(result.universe["security_id"], result.universe["max_weight"], strict=True))
    assert caps == {"A": Decimal("0.5"), "B": None, "C": Decimal(0)}


def test_wash_sale_rule_keeps_a_tighter_existing_cap_and_is_a_no_op_without_prior_sales(make: Factories, frames: Frames) -> None:
    base = frames.three_security_universe()
    universe = base.assign(max_weight=pd.Series([Decimal("0.1"), None, None], index=base.index, dtype="object"))
    result = avoid_cross_portfolio_wash_sales(make.portfolio_data(universe=universe), prior_context(make, frames, "A"))
    assert result.universe["max_weight"].tolist() == [Decimal("0.1"), None, None]
    untouched = avoid_cross_portfolio_wash_sales(make.portfolio_data(), prior_context(make, frames))
    assert "max_weight" not in untouched.universe.columns


# --- apply_rules ---


def test_apply_rules_records_provenance_and_row_counts(make: Factories) -> None:
    rules = (step("cap_single_name", max_weight="0.5"), step("add_zero_alpha"))
    result, audits = apply_rules(make.portfolio_data(), rules, ctx=None)  # ty: ignore[invalid-argument-type]  # resolve_step returns ResolvedStep
    assert result.applied_rules == ("portfolio_optimizer.rules:cap_single_name", "portfolio_optimizer.rules:add_zero_alpha")
    assert [audit.qualname for audit in audits] == list(result.applied_rules)
    assert audits[0].rows_in == audits[0].rows_out == {"holdings": 2, "universe": 3, "targets": 3}
    assert len(audits[0].source_sha256) == 64


def test_apply_rules_passes_context_only_to_chain_aware_rules(make: Factories, frames: Frames) -> None:
    result, _ = apply_rules(make.portfolio_data(), (step("avoid_cross_portfolio_wash_sales"),), ctx=prior_context(make, frames, "A"))  # ty: ignore[invalid-argument-type]  # see above
    assert result.universe["max_weight"].tolist()[0] == Decimal("0.5")


def test_apply_rules_rejects_a_rule_that_returns_the_wrong_type(make: Factories) -> None:
    with pytest.raises(RuleError, match="returned DataFrame, expected PortfolioData"):
        apply_rules(make.portfolio_data(), (step("tests.conftest:lying_rule"),), ctx=None)  # ty: ignore[invalid-argument-type]  # see above


def test_apply_rules_with_no_rules_is_identity(make: Factories) -> None:
    data = make.portfolio_data()
    result, audits = apply_rules(data, (), ctx=None)
    assert result is data
    assert audits == ()
