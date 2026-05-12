"""Shared fixtures for the framework test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from bioagent_service import ServiceSettings


class _TempSettings(ServiceSettings):
    """Settings backed by a tmp_path, never reads env or .env files."""

    # Disable env/.env loading so unrelated env vars on the developer's machine
    # can't poison the test (e.g., a stale JOBS_BASE_DIR=/var/...).
    model_config = SettingsConfigDict(
        env_file=None,
        env_prefix="BIOAGENT_TEST_",
        extra="ignore",
    )


@pytest.fixture
def settings(tmp_path: Path) -> ServiceSettings:
    return _TempSettings(jobs_base_dir=tmp_path / "jobs", max_concurrent_jobs=2)
