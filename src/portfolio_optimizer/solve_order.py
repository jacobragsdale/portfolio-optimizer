"""Solve-order steps — yours to edit.

A solve-order step is ``(data: PortfolioData[, params]) -> Decimal``: it looks at one portfolio's
ruled bundle and returns its **solve-order key**. Lower keys solve first; equal keys break on
``portfolio_id``. The key must be finite. Name one step in the run config's ``solve_order`` field and
it replaces the portfolios frame's ``solve_order`` column.

Solve order is a priority, not a sequence: a portfolio waits only for higher-priority portfolios
that can buy a security it can buy too, so the key decides who gets first pick of a shared budget.
"""

from decimal import Decimal

from portfolio_optimizer.domain.data import PortfolioData


def furthest_from_target_first(data: PortfolioData) -> Decimal:
    """Minus the portfolio's active share, so the portfolio furthest from its target solves first.

    Active share is half the sum of absolute weight differences against the benchmark, in exact
    ``Decimal`` from holdings, prices, and NAV. Two portfolios at the same distance tie.
    """
    prices = {str(security): Decimal(str(price)) if not isinstance(price, Decimal) else price for security, price in zip(data.universe["security_id"], data.universe["price"], strict=True)}
    nav = data.details.nav
    current = {str(security): Decimal(int(quantity)) * prices[str(security)] / nav for security, quantity in zip(data.holdings["security_id"], data.holdings["quantity"], strict=True)}
    target = {str(security): weight if isinstance(weight, Decimal) else Decimal(str(weight)) for security, weight in zip(data.targets["security_id"], data.targets["weight"], strict=True)}
    distance = sum((abs(current.get(security, Decimal(0)) - target.get(security, Decimal(0))) for security in current.keys() | target.keys()), Decimal(0))
    return -(distance / 2)
