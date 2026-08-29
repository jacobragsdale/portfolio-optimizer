"""Process configuration from the environment; loaded once at startup, never defaulted.

Two kinds of setting live here. Where data is read from, where runs are written, and how loudly to
log are the run's surroundings. Which cluster the run provisions for itself and how big it is are the
run's *execution mechanics* — deliberately not part of the run config, so a laptop run and a cluster
run of one config hash identically and differ only in the manifest's ``settings`` block.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, override

from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, EnvSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict

ENV_PREFIX = "PORTFOLIO_OPTIMIZER_"
KUBERNETES_MARKER = "KUBERNETES_SERVICE_HOST"
"""Set in every pod by Kubernetes and on no laptop; what ``cluster: auto`` resolves on."""

type ClusterKind = Literal["local", "kubernetes", "address"]

CLUSTER_PATTERN = r"^(local|kubernetes|auto|tcp://\S+|tls://\S+)$"


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    """Which cluster the run provisions and how big it is; the runner's view of the settings.

    ``cluster`` is the resolved kind (``local``, ``kubernetes``) or a scheduler address — never
    ``auto``. ``min_workers`` is what is provisioned before the load stage and ``max_workers`` what the
    run scales to after assembly.
    """

    cluster: str
    min_workers: int
    max_workers: int
    cluster_timeout_s: float
    worker_image: str | None = None
    image_digest: str | None = None

    @property
    def cluster_kind(self) -> ClusterKind:
        """``local``, ``kubernetes``, or ``address`` for a scheduler URL."""
        if self.cluster == "local":
            return "local"
        if self.cluster == "kubernetes":
            return "kubernetes"
        return "address"


class Settings(BaseSettings):
    """Where data is read from, where runs are written, how loudly to log, and which cluster the run provisions."""

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, strict=True, extra="forbid", frozen=True, validate_default=True, revalidate_instances="always", allow_inf_nan=False)

    output_dir: Path
    data_root: Path
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    cluster: str = Field(pattern=CLUSTER_PATTERN)
    min_workers: int = Field(ge=1)
    max_workers: int = Field(ge=1)
    cluster_timeout_s: float = Field(gt=0)
    worker_image: str | None = Field(default=None, min_length=1)
    image_digest: str | None = Field(default=None, min_length=1)

    @classmethod
    @override
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Only explicit values count: no dotenv or secrets-directory lookups behind the caller's back."""
        del settings_cls, dotenv_settings, file_secret_settings
        return (init_settings, env_settings)

    @model_validator(mode="after")
    def _cluster_settings_agree(self) -> Self:
        if self.cluster == "kubernetes" and self.worker_image is None:
            msg = f"{_variable('cluster')}=kubernetes requires {_variable('worker_image')}: the image worker pods run, normally this run's own"
            raise ValueError(msg)
        if self.min_workers > self.max_workers:
            msg = f"{_variable('min_workers')} ({self.min_workers}) exceeds {_variable('max_workers')} ({self.max_workers})"
            raise ValueError(msg)
        return self

    def execution(self) -> ExecutionSettings:
        """The execution mechanics as the runner consumes them."""
        return ExecutionSettings(
            cluster=self.cluster, min_workers=self.min_workers, max_workers=self.max_workers, cluster_timeout_s=self.cluster_timeout_s, worker_image=self.worker_image, image_digest=self.image_digest
        )

    def shown(self) -> dict[str, str]:
        """Every setting as text, for the manifest; a setting that does not apply is omitted."""
        return {name: str(value) for name, value in self.model_dump().items() if value is not None}


def _variable(field: str) -> str:
    return f"{ENV_PREFIX}{field.upper()}"


class SettingsError(ValueError):
    """Required configuration is missing or invalid; startup must stop."""


class _MappingEnvSource(EnvSettingsSource):
    """The standard environment source, reading from an explicit mapping instead of ``os.environ``."""

    def __init__(self, settings_cls: type[BaseSettings], env: Mapping[str, str]) -> None:
        self._mapping = {key.lower(): value for key, value in env.items()}
        super().__init__(settings_cls)

    @override
    def _load_env_vars(self) -> Mapping[str, str | None]:
        return self._mapping


def load_settings(env: Mapping[str, str]) -> Settings:
    """Build settings from an explicit environment mapping (a seam for tests; production passes ``os.environ``).

    Every ``PORTFOLIO_OPTIMIZER_*`` variable must correspond to a field, and every field must be
    present: a typo in a variable name is an error, not a silently ignored value. ``cluster: auto``
    is resolved here — ``kubernetes`` inside a pod, ``local`` anywhere else — so the settings the run
    sees, and the manifest records, name what actually happened.
    """
    known = {_variable(name) for name in Settings.model_fields}
    unknown = sorted(key for key in env if key.upper().startswith(ENV_PREFIX) and key.upper() not in known)
    if unknown:
        msg = f"invalid settings: unknown variable(s) {unknown}; expected {sorted(known)}"
        raise SettingsError(msg)
    try:
        return Settings.model_validate(_MappingEnvSource(Settings, _resolve_auto_cluster(env))())
    except ValidationError as error:
        details = "; ".join(_detail(item["loc"], str(item["msg"])) for item in error.errors())
        msg = f"invalid settings: {details}"
        raise SettingsError(msg) from error


def _resolve_auto_cluster(env: Mapping[str, str]) -> dict[str, str]:
    resolved = dict(env)
    for key, value in env.items():
        if key.upper() == _variable("cluster") and value == "auto":
            resolved[key] = "kubernetes" if KUBERNETES_MARKER in env else "local"
    return resolved


def _detail(loc: tuple[int | str, ...], message: str) -> str:
    if not loc:
        return message.removeprefix("Value error, ")
    return f"{ENV_PREFIX}{'.'.join(str(part) for part in loc).upper()}: {message}"
