"""Tier 1/2: the worker environment fingerprint names what differs, package versions are found per distribution, and git info degrades to unknown."""

import importlib.metadata
from pathlib import Path

import pandas as pd

from portfolio_optimizer.engine.environment import WorkerEnvironment, environment_for, external_modules, package_versions, read_git_info
from tests.conftest import example_config


def test_environment_fingerprint_names_what_differs_and_which_packages_it_covers() -> None:
    config = example_config()
    here = environment_for(config, cwd=Path.cwd(), image_digest=None)
    there = environment_for(config, cwd=Path.cwd(), image_digest="sha256:abc")
    assert here.differences(here) == []
    assert here.differences(there) == ["image_digest: None here, 'sha256:abc' there"]
    assert external_modules(config) == ("tests.steps",)
    assert dict(here.packages) == {"tests": "unknown"}
    assert isinstance(here, WorkerEnvironment) and hash(here) == hash(environment_for(config, cwd=Path.cwd(), image_digest=None))


def test_package_versions_name_the_distribution_behind_each_external_module() -> None:
    found = package_versions(["pandas.core.frame", "pandas", "portfolio_optimizer.rules", "fake_steps"])
    assert found["pandas"] == pd.__version__  # an indexed distribution, once, whatever the submodule
    assert found["portfolio-optimizer"] == importlib.metadata.version("portfolio-optimizer")  # an editable install, found by name
    assert found["fake_steps"] == "unknown"  # a module no distribution provides
    assert package_versions([]) == {}


def test_git_info_outside_a_repository_is_unknown(tmp_path: Path) -> None:
    info = read_git_info(tmp_path)
    assert info.sha == "unknown"
    assert not info.dirty
