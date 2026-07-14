"""Env-driven config for reinvent-server (REINVENT_ prefix).

NB: `prior_base` intentionally maps to env var REINVENT_PRIOR_BASE — the SAME
var upstream REINVENT4's prior_registry reads. Setting it once configures both
this service and the subprocess.
"""
from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class ReinventSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="REINVENT_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/reinvent_jobs"))
    root: Path = Field(default=Path("/opt/reinvent-server"))
    upstream_dir: Path = Field(default=Path("/opt/reinvent-server/upstream/REINVENT4"))
    python: Path = Field(default=Path("/opt/reinvent-server/.venv/bin/python"))
    reinvent_bin: Path = Field(default=Path("/opt/reinvent-server/.venv/bin/reinvent"))
    prior_base: Path = Field(default=Path("/data/models/reinvent"))
    device: str = "cuda:0"
    max_concurrent_jobs: int = 2
    task_endpoints_enabled: bool = True

    # Framework convention field; same value as prior_base.
    weights_dir: Path = Field(default=Path("/data/models/reinvent"))
