"""ServiceSettings — env loading + validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from bioagent_service import ServiceSettings


class _NoEnvSettings(ServiceSettings):
    # Don't bleed env vars from the developer machine into tests.
    model_config = SettingsConfigDict(env_file=None, env_prefix="DOES_NOT_EXIST_", extra="ignore")


def test_defaults() -> None:
    s = _NoEnvSettings()
    assert s.port == 9000
    assert s.max_concurrent_jobs == 2
    assert s.disk_limit_mb == 8000


def test_env_prefix_override(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MySettings(ServiceSettings):
        model_config = SettingsConfigDict(env_file=None, env_prefix="MY_", extra="ignore")

    monkeypatch.setenv("MY_PORT", "1234")
    monkeypatch.setenv("MY_MAX_CONCURRENT_JOBS", "4")
    s = _MySettings()
    assert s.port == 1234
    assert s.max_concurrent_jobs == 4


def test_invalid_port_rejected() -> None:
    with pytest.raises(ValidationError):
        _NoEnvSettings(port=70000)


def test_invalid_disk_limit_rejected() -> None:
    with pytest.raises(ValidationError):
        _NoEnvSettings(disk_limit_mb=10)  # below ge=100
