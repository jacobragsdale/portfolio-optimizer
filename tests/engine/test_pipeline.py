"""Tier 1: the rule pipeline's provenance and guards; rules never see other portfolios."""

import pytest

from portfolio_optimizer.config.resolve import resolve_step
from portfolio_optimizer.config.steps import ResolvedStep
from portfolio_optimizer.engine.pipeline import RuleError, apply_rules
from tests.conftest import Factories, step_spec


def rule(name: str, **params: object) -> ResolvedStep:
    return resolve_step(step_spec(name, **params), "rule")


def test_apply_rules_records_provenance_and_row_counts(make: Factories) -> None:
    rules = (rule("cap_single_name", max_weight="0.5"), rule("add_zero_alpha"))
    result, audits = apply_rules(make.portfolio_data(), rules)
    assert result.applied_rules == ("portfolio_optimizer.rules:cap_single_name", "portfolio_optimizer.rules:add_zero_alpha")
    assert [audit.qualname for audit in audits] == list(result.applied_rules)
    assert audits[0].rows_in == audits[0].rows_out == {"holdings": 2, "universe": 3, "targets": 3}
    assert len(audits[0].source_sha256) == 64


def test_apply_rules_rejects_a_rule_that_returns_the_wrong_type(make: Factories) -> None:
    with pytest.raises(RuleError, match="returned DataFrame, expected PortfolioData"):
        apply_rules(make.portfolio_data(), (rule("tests.steps:lying_rule"),))


def test_apply_rules_with_no_rules_is_identity(make: Factories) -> None:
    data = make.portfolio_data()
    result, audits = apply_rules(data, ())
    assert result is data
    assert audits == ()
