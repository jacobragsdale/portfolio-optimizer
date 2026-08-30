"""Solve-order steps — yours to edit.

A solve-order step is ``(data: PortfolioData[, params]) -> Decimal``: it looks at one portfolio's
ruled bundle and returns its **solve-order key**. Lower keys solve first; equal keys break on
``portfolio_id``. The key must be finite. Name one step in the run config's ``solve_order`` field and
it replaces the portfolios frame's ``solve_order`` column.

Solve order is a priority, not a sequence: a portfolio waits only for higher-priority portfolios
that can trade a security it can trade too, on the side the run couples through, so the key decides
who gets first pick of a shared budget.
"""

from decimal import Decimal

from portfolio_optimizer.domain.data import PortfolioData


def most_uninvested_first(data: PortfolioData) -> Decimal:
    """Minus the fraction of NAV the portfolio has yet to invest, so the account with the most to put to work solves first.

    The uninvested fraction is one minus the market value of the holdings over NAV, in exact
    ``Decimal`` from holdings, prices, and NAV — the same quantity the cash bounds constrain. It is the
    key that matters when the scarce thing is liquidity: whoever has the most to buy gets first pick of
    each name's budget. Two portfolios equally invested tie.
    """
    prices = {str(security): Decimal(str(price)) if not isinstance(price, Decimal) else price for security, price in zip(data.universe["security_id"], data.universe["price"], strict=True)}
    nav = data.details.nav
    invested = sum((Decimal(int(quantity)) * prices[str(security)] / nav for security, quantity in zip(data.holdings["security_id"], data.holdings["quantity"], strict=True)), Decimal(0))
    return invested - 1
