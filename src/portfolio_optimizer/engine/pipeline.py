"""Apply the configured rules to a portfolio's bundle, recording what each one did."""

from collections.abc import Sequence

from portfolio_optimizer.config.resolve import ResolvedStep
from portfolio_optimizer.domain.data import PortfolioData
from portfolio_optimizer.domain.results import RuleAuditRecord, SolveContext


class RuleError(ValueError):
    """A rule returned something other than a ``PortfolioData``."""


def apply_rules(data: PortfolioData, rules: Sequence[ResolvedStep], ctx: SolveContext | None) -> tuple[PortfolioData, tuple[RuleAuditRecord, ...]]:
    """Run ``rules`` in order. Each step gets the previous step's output; chain-aware rules get ``ctx``."""
    audits: list[RuleAuditRecord] = []
    current = data
    for step in rules:
        rows_in = _row_counts(current)
        result = step.invoke(data=current, context=ctx if step.needs_context else None)
        if not isinstance(result, PortfolioData):
            msg = f"rule {step.qualname!r} returned {type(result).__name__}, expected PortfolioData"
            raise RuleError(msg)
        current = result.with_rule_applied(step.qualname)
        audits.append(RuleAuditRecord(qualname=step.qualname, source_sha256=step.source_sha256, params_sha256=step.params_sha256, rows_in=rows_in, rows_out=_row_counts(current)))
    return current, tuple(audits)


def _row_counts(data: PortfolioData) -> dict[str, int]:
    return {"holdings": len(data.holdings), "universe": len(data.universe), "targets": len(data.targets)}
