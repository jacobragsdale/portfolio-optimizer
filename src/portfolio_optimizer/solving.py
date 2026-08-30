"""What a solve step receives and what it must return: the engine's interpreter seam.

The engine builds the problem as data (a :class:`~portfolio_optimizer.domain.results.ProblemSpec`),
folds the chain, and then hands *one* step — configured as ``solve`` — everything it needs to decide
the weights: the spec, the chain, the side profile, the resolved terms and constraints, the run's
extra datasets, and the cvxpy options block. The shipped step (``solvers.cvxpy``) builds and solves a cvxpy problem; a firm's
own library or a pure numpy function fits the same contract. Whatever the step does, the side profile
turns its ``w`` into the trade and the verifier decides whether the answer is acceptable — the
guarantees are the verifier's, not the step's.

A solve step returns weights and nothing else: it writes no files, reads no clock, and sees no other
portfolio. It may raise; the engine records that as the portfolio's failure at stage ``solve``.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from portfolio_optimizer.config.models import SolverConfig
from portfolio_optimizer.config.steps import ResolvedStep
from portfolio_optimizer.domain.results import F64, ChainState, ProblemSpec, SolveStatus, StepRef
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
    constraints: pd.DataFrame
    """This portfolio's constraint rows as loaded and as the rules left them, in the desk's own shape.

    The engine reads only the rows' *declarations*: a row whose ``kind`` column names a typed model
    (``domain/constraints.py``) tells the schedule whether it reads the chain and what it couples
    through, and nothing more. What a row *does* is the step's business: the shipped ``cvxpy`` step
    renders typed rows from their models and reads the ``name``/``label``/``params`` function
    convention for the rest; a step with its own syntax reads its own, and one that needs no
    constraints ignores the frame, which is empty when the run declares no such dataset.
    """

    solver: SolverConfig
    """The ``solver`` block of the run config: cvxpy's solver and its options. A step that is not cvxpy may ignore it."""

    extras: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    """Every extra dataset the run carried, as the rules left it: each one reduced to this portfolio's rows where it has a ``portfolio_id`` column, passed whole where it does not.

    An extra is any dataset the engine does not know, and it knows nothing about these either — they
    reach the step exactly as they were loaded. This is where runtime parameters live: a
    ``global_parameters`` frame of run-wide settings, a per-security score a desk's own library reads,
    anything a step needs that is not a per-security column of the universe. Each one is
    content-hashed and recorded in the manifest like every other input, so a run driven by them is
    still a pure function of a snapshot.
    """


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
    detail: str = ""
    constraints: tuple[StepRef, ...] = field(default_factory=tuple)
    """What the step applied, for the verifier and the manifest.

    The step is what decides what the constraint rows mean, so it says what it made of them: a
    qualified name, JSON-safe params, and a label each — for a typed row, the model's kind and fields. The verifier re-checks every ref it has a
    numpy twin for and reports the rest as ``unverified``; the manifest records them under the
    portfolio. A step that interprets nothing leaves this empty and its constraints go unverified,
    which is the honest answer rather than a silent pass.
    """
