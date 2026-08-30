"""What a solve step receives and what it must return: the engine's interpreter seam.

The engine builds the problem as data (a :class:`~portfolio_optimizer.domain.results.ProblemSpec`),
folds the chain, and then hands *one* step — configured as ``solve`` — everything it needs to decide
the weights: the spec, the chain, the side profile, the resolved terms and constraints, and the
cvxpy options block. The shipped step (``solvers.cvxpy``) builds and solves a cvxpy problem; a firm's
own library or a pure numpy function fits the same contract. Whatever the step does, the side profile
turns its ``w`` into the trade and the verifier decides whether the answer is acceptable — the
guarantees are the verifier's, not the step's.

A solve step returns weights and nothing else: it writes no files, reads no clock, and sees no other
portfolio. It may raise; the engine records that as the portfolio's failure at stage ``solve``.
"""

from dataclasses import dataclass

from portfolio_optimizer.config.models import SolverConfig
from portfolio_optimizer.config.steps import ResolvedConstraint, ResolvedStep
from portfolio_optimizer.domain.results import F64, ChainState, ProblemSpec, SolveStatus
from portfolio_optimizer.domain.sides import SideProfile


class SolveSetupError(ValueError):
    """A term, constraint, or solve step did not produce what its contract promises."""


@dataclass(frozen=True, slots=True)
class SolveRequest:
    """Everything a solve step may use, and all it will get."""

    spec: ProblemSpec
    chain: ChainState
    profile: SideProfile
    terms: tuple[ResolvedStep, ...]
    constraints: tuple[ResolvedConstraint, ...]
    solver: SolverConfig
    """The ``solver`` block of the run config: cvxpy's solver and its options. A step that is not cvxpy may ignore it."""


@dataclass(frozen=True, slots=True, eq=False)
class SolveResult:
    """What a solve step returns.

    ``w`` is the weights, aligned to the spec, and is all a pure function needs to fill in. ``objective``
    is the value the step minimized, when it minimized one; without it the verifier skips the
    objective comparison and evaluates the configured terms as a report line instead. ``solver`` and
    ``solver_version`` name what produced the answer; left ``None``, the engine records the step's
    qualified name and its package version.
    """

    w: F64 | None
    status: SolveStatus = SolveStatus.OPTIMAL
    objective: float | None = None
    iterations: int | None = None
    solve_time_s: float = 0.0
    solver: str | None = None
    solver_version: str | None = None
    cvxpy_version: str = "n/a"
    detail: str = ""
