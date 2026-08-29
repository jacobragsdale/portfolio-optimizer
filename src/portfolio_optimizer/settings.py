"""Process configuration from the environment; loaded once at startup, never defaulted."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, override

from pydantic import ValidationError
from pydantic_settings import BaseSettings, EnvSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict

ENV_PREFIX = "PORTFOLIO_OPTIMIZER_"


class Settings(BaseSettings):
    """Where data is read from, where runs are written, and how loudly to log."""

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, strict=True, extra="forbid", frozen=True, validate_default=True, revalidate_instances="always", allow_inf_nan=False)

    output_dir: Path
    data_root: Path
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]

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
    present: a typo in a variable name is an error, not a silently ignored value.
    """
    known = {f"{ENV_PREFIX}{name}".upper() for name in Settings.model_fields}
    unknown = sorted(key for key in env if key.upper().startswith(ENV_PREFIX) and key.upper() not in known)
    if unknown:
        msg = f"invalid settings: unknown variable(s) {unknown}; expected {sorted(known)}"
        raise SettingsError(msg)
    try:
        return Settings.model_validate(_MappingEnvSource(Settings, env)())
    except ValidationError as error:
        details = "; ".join(f"{ENV_PREFIX}{'.'.join(str(part) for part in item['loc']).upper()}: {item['msg']}" for item in error.errors())
        msg = f"invalid settings: {details}"
        raise SettingsError(msg) from error
