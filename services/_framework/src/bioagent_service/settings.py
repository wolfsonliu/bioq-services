"""Runtime configuration for services, backed by pydantic-settings.

Every service subclasses `ServiceSettings`, sets its `env_prefix`, and adds
service-specific fields. The framework reads only the base fields; adapters
read everything via `self.settings`. No `os.getenv` calls anywhere.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    """Base settings shared by all bioagent services.

    Subclasses must set their own `env_prefix` in `model_config`, e.g.:

        class Genie3Settings(ServiceSettings):
            model_config = SettingsConfigDict(env_prefix="GENIE3_", extra="ignore")
            genie3_root: Path = Path("/opt/genie3")
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Where each job's working directory lives (created on demand).
    jobs_base_dir: Path = Field(
        default=Path("/data/jobs"),
        description="Root directory under which per-job dirs are created.",
    )
    # When total bytes under jobs_base_dir exceed this, completed jobs are evicted.
    disk_limit_mb: int = Field(
        default=8000,
        ge=100,
        description="Soft cap on jobs_base_dir size; older completed/failed jobs are cleaned past this.",
    )
    # HTTP listening port; default matches Alibaba Cloud FC's CAPort.
    port: int = Field(default=9000, ge=1, le=65535)
    # uvicorn keep-alive; FC requires >= 15 minutes for long jobs.
    keep_alive_sec: int = Field(default=900, ge=60)
    # Bound on concurrent subprocesses. >1 only makes sense if the algorithm can share GPU.
    max_concurrent_jobs: int = Field(default=1, ge=1)
    # How many trailing bytes of the log to embed in JobInfo.error_tail on failure.
    error_tail_chars: int = Field(default=4000, ge=200, le=64000)
