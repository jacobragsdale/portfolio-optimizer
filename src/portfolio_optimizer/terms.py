"""Objective terms and constraints — yours to edit.

An objective term is ``(x: DecisionVars, spec: ProblemSpec, params: P) -> ObjectiveTerm``; a
constraint is the same signature returning ``ConstraintSet``. Add ``chain: ChainState`` to read
what earlier portfolios in the run have already ordered. Everything is expressed through the
typed atoms in :mod:`portfolio_optimizer.cvx.adapter`, so the post-solve verifier can recompute
each shipped term and constraint without cvxpy.
"""
