"""The environment a task ran in, as a fingerprint the run compares and the manifest records.

Interpreter, numerical libraries, solver, the packages that supplied external steps, the code revision,
and the container image: on a local cluster the workers are spawned from the run's own interpreter and
the fingerprints agree by construction. On a cluster the workers are pods, and a worker running a stale
image or an older step package is exactly the case the comparison exists to catch — its portfolio fails
at stage ``worker`` instead of producing an answer the manifest could not explain.
"""

import functools
import importlib.metadata
import platform
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from portfolio_optimizer.config.models import DatasetConfig, RunConfig
from portfolio_optimizer.domain.types import StrictModel
from portfolio_optimizer.solving import configured_solver, solver_version

IMAGE_DIGEST_VARIABLE = "PORTFOLIO_OPTIMIZER_IMAGE_DIGEST"
"""The variable an image or platform sets to the digest of the image a process runs in; part of the fingerprint when present.

The run and every worker read it from their own environment: a client that does not run the worker
image does not carry its digest, and the fingerprint comparison is what says so.
"""


@dataclass(frozen=True, slots=True)
class GitInfo:
    """The code revision a process executes."""

    sha: str
    dirty: bool


def read_git_info(repo: Path) -> GitInfo:
    """Ask git for the current revision; ``unknown`` when not in a repository."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True, timeout=10).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return GitInfo(sha="unknown", dirty=False)
    return GitInfo(sha=sha, dirty=bool(status.strip()))


class WorkerEnvironment(StrictModel):
    """What determines a task's numerical result, independent of which host ran it.

    ``packages`` holds the installed version of every distribution behind a step named outside the
    template modules, as sorted ``(distribution, version)`` pairs. Two processes with equal fingerprints
    run the same code on the same libraries.
    """

    python: str
    cvxpy: str
    numpy: str
    pandas: str
    solver: str
    solver_version: str
    packages: tuple[tuple[str, str], ...]
    git_sha: str
    image_digest: str | None

    def differences(self, other: "WorkerEnvironment") -> list[str]:
        """Field-by-field description of how ``other`` differs from this fingerprint."""
        return [f"{name}: {getattr(self, name)!r} here, {getattr(other, name)!r} there" for name in type(self).model_fields if getattr(self, name) != getattr(other, name)]


def environment_for(config: RunConfig, *, cwd: Path, image_digest: str | None) -> WorkerEnvironment:
    """The fingerprint of this process for ``config``: cached per process, since none of it changes while it runs."""
    return _environment(configured_solver(config), external_modules(config), str(cwd), image_digest)


def external_modules(config: RunConfig) -> tuple[str, ...]:
    """The modules of every qualified step name in ``config``, sorted; what ``packages`` is computed over.

    Terms and constraints are kinds, not steps: a kind a package publishes is covered when that package
    also supplies a step named here, which is the ordinary arrangement; one from a package nothing else
    names is not fingerprinted, and a worker missing it fails at stage ``build`` or ``solve`` rather than
    at the environment check.
    """
    specs = [*(dataset.loader for dataset in config.datasets.values() if isinstance(dataset, DatasetConfig)), *config.assembly, *config.rules, config.build, config.solve, config.sink]
    if config.solve_order is not None:
        specs.append(config.solve_order)
    return tuple(sorted({spec.name.partition(":")[0] for spec in specs if spec.is_qualified}))


@functools.cache
def _environment(solver: str, external: tuple[str, ...], cwd: str, image_digest: str | None) -> WorkerEnvironment:
    return WorkerEnvironment(
        python=platform.python_version(),
        cvxpy=distribution_version("cvxpy"),
        numpy=distribution_version("numpy"),
        pandas=distribution_version("pandas"),
        solver=solver,
        solver_version=solver_version(solver),
        packages=tuple(sorted(package_versions(external).items())),
        git_sha=read_git_info(Path(cwd)).sha,
        image_digest=image_digest,
    )


def host_name() -> str:
    """The name of the machine this process runs on, for the manifest's record of who did the work."""
    return platform.node()


def package_versions(module_names: Iterable[str]) -> dict[str, str]:
    """The installed version of the distribution behind each module, keyed by distribution name.

    A module's top-level package is looked up among the installed distributions; an editable install
    that the metadata does not index is found by name when the distribution is named after the package.
    A module no distribution claims is recorded under its own top-level name as ``unknown``.
    """
    claimed = importlib.metadata.packages_distributions()
    found: dict[str, str] = {}
    for top_level in sorted({name.partition(".")[0] for name in module_names}):
        distributions = claimed.get(top_level)
        if distributions is None:
            try:
                distributions = [str(importlib.metadata.distribution(top_level).metadata["Name"])]
            except importlib.metadata.PackageNotFoundError:
                found[top_level] = "unknown"
                continue
        for distribution in distributions:
            found[distribution] = distribution_version(distribution)
    return found


def distribution_version(package: str) -> str:
    """The installed version of ``package``, or ``unknown``."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
