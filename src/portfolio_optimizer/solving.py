"""What a solve step receives and what it must return: the engine's interpreter seam.

The engine builds the problem as data (a :class:`~portfolio_optimizer.domain.results.ProblemSpec`),
folds the chain, and then hands *one* step — configured as ``solve`` — everything it needs to decide
the weights: the spec, the chain, the side profile, the typed terms, this portfolio's constraint
rows, and the run's extra datasets. The shipped step (``solvers.cvxpy``) builds and solves a cvxpy
problem from the terms' and constraints' own renderers; a firm's own library or a pure numpy
function fits the same contract. Whatever the step does, the side profile turns its ``w`` into the
trade and the verifier decides whether the answer is acceptable — the guarantees are the verifier's,
not the step's.

A solve step returns weights and nothing else: it writes no files, reads no clock, and sees no other
portfolio. It may raise; the engine records that as the portfolio's failure at stage ``solve``.
"""

import importlib.metadata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from portfolio_optimizer.config.models import RunConfig
from portfolio_optimizer.domain.objective import TypedTerm
from portfolio_optimizer.domain.results import F64, ChainState, ConstraintRecord, ProblemSpec, SolveStatus
from portfolio_optimizer.domain.sides import SideProfile

SHIPPED_CVXPY_SOLVE = "portfolio_optimizer.solvers:cvxpy"
"""The one solve step whose chain access is exactly the configured terms and constraints; any other step may read ``request.chain`` however it likes, so its runs couple conservatively."""

DEFAULT_SOLVER = "CLARABEL"


class SolveSetupError(ValueError):
    """A term, constraint, or solve step did not produce what its contract promises."""


@dataclass(frozen=True, slots=True)
class SolverSpec:
    """What the engine knows about one cvxpy solver: the distribution that versions it and how it spells a time limit.

    A solver cvxpy can see but this table does not name is refused when the config resolves: without
    its distribution the environment fingerprint would record ``unknown`` for its version on every
    process, and two different builds of it would compare equal.
    """

    name: str
    distribution: str
    time_limit_option: str | None


SOLVERS: Mapping[str, SolverSpec] = {
    spec.name: spec
    for spec in (
        SolverSpec("CLARABEL", "clarabel", "time_limit"),
        SolverSpec("OSQP", "osqp", "time_limit"),
        SolverSpec("SCS", "scs", "time_limit_secs"),
        SolverSpec("HIGHS", "highspy", "time_limit"),
        SolverSpec("PIQP", "piqp", None),
    )
}
"""Every solver the shipped cvxpy step may name. cvxpy installs the first four; ``PIQP`` is the ``piqp`` extra. Adding one is a row here and an extra in ``pyproject.toml``."""


def solver_failures(name: str, time_limit_s: float | None, installed: Sequence[str]) -> list[str]:
    """Why solver ``name`` cannot run against ``installed``, if it cannot; empty when it can.

    Unknown to :data:`SOLVERS`, not installed, or asked for a time limit it has no option for. The
    resolver runs this in every process that will solve, so a run fails before any data loads.
    """
    spec = SOLVERS.get(name)
    if spec is None:
        return [f"solver {name!r} is not one the adapter knows; known: {sorted(SOLVERS)}"]
    failures: list[str] = []
    if name not in installed:
        failures.append(f"solver {name!r} is not installed in this environment; installed: {sorted(set(installed) & set(SOLVERS))}")
    if time_limit_s is not None and spec.time_limit_option is None:
        failures.append(f"solver {name!r} has no time-limit option; remove time_limit_s")
    return failures


def solver_version(solver: str) -> str:
    """Installed version of the distribution behind ``solver``, or ``"unknown"``."""
    spec = SOLVERS.get(solver)
    if spec is None:
        return "unknown"
    try:
        return importlib.metadata.version(spec.distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def configured_solver(config: RunConfig) -> str:
    """What the config says will solve: the cvxpy solver the shipped step's params name, or the solve step itself."""
    if config.solve.name in ("cvxpy", SHIPPED_CVXPY_SOLVE):
        return str(config.solve.params.get("solver", DEFAULT_SOLVER))
    return config.solve.name


@dataclass(frozen=True, slots=True)
class SolveRequest:
    """Everything a solve step may use, and all it will get."""

    spec: ProblemSpec
    chain: ChainState
    profile: SideProfile
    terms: tuple[TypedTerm, ...]
    """The configured objective terms as models; the shipped step renders each through its ``to_cvxpy``, a firm's library reads their fields."""

    constraints: pd.DataFrame
    """This portfolio's constraint rows as loaded and as the rules left them.

    The engine reads only the rows' *declarations*: a row whose ``kind`` column names a typed model
    (``domain/constraints.py``) tells the schedule whether it reads the chain and what it couples
    through, and nothing more. What a row *does* is the step's business: the shipped ``cvxpy`` step
    renders typed rows from their models; a step with its own syntax reads its own, and one that
    needs no constraints ignores the frame, which is empty when the run declares no such dataset.
    """

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
    constraints: tuple[ConstraintRecord, ...] = field(default_factory=tuple)
    """What the step applied, as constraint records (a model's ``record()``), for the verifier and the manifest.

    The step is what decides what the constraint rows mean, so it says what it made of them. The
    verifier re-checks every record through its model's own residual; a step that interprets nothing
    leaves this empty and its constraints go unchecked, which is the honest answer rather than a
    silent pass.
    """

    duals: Mapping[str, float] = field(default_factory=dict)
    """Per constraint name, the largest dual value the solver reported — its shadow price; empty for a step that has none."""
