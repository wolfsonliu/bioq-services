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
    # Bound on concurrent subprocesses. Exceeding this limit returns HTTP 503.
    max_concurrent_jobs: int = Field(default=2, ge=1)
    # How many trailing bytes of the log to embed in JobInfo.error_tail on failure.
    error_tail_chars: int = Field(default=4000, ge=200, le=64000)
    # FC self-keepalive: while jobs are active, ping /healthz at this interval
    # to prevent FC from reclaiming the instance. 0 = disabled.
    keepalive_interval_s: int = Field(default=60, ge=0)
    # External URL to ping for keepalive. When set, the keepalive thread sends
    # requests through FC's gateway (which counts as activity) instead of
    # localhost (which FC ignores). Typically the function's own fcapp.run URL
    # with /healthz appended.  Env: <PREFIX>_KEEPALIVE_URL
    keepalive_url: str | None = Field(default=None)
    # FC session affinity header name. When set, POST responses that contain a
    # job_id will include this header so FC can bind follow-up requests to the
    # same instance.  Env: <PREFIX>_SESSION_HEADER_NAME
    # Naming rules: no "x-fc-" prefix, letter-start, 5-40 chars, [a-zA-Z0-9_-].
    session_header_name: str | None = Field(default=None)
    # Task endpoint (FC async task mode) — controls /api/tasks/<name> registration.
    # When False, `register_task_endpoint` is a no-op (useful for services that have
    # not yet declared task endpoints, or for legacy deployments).
    task_endpoints_enabled: bool = Field(default=True)
    # HTTP header from which the task endpoint reads a client-supplied job_id.
    # Empty/missing → server generates a UUID job_id.  Naming: avoid 'X-Fc-' prefix
    # (FC strips those); we ALSO read 'X-Fc-Async-Task-Id' as a fallback so a single
    # FCDispatcher.submit can populate both.
    task_job_id_header: str = Field(default="X-Bioagent-Job-Id")
