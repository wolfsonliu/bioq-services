"""Env-driven config for dockq-server.

All values go through pydantic-settings (no `os.getenv`). Env vars use the
`DOCKQ_` prefix (e.g. `DOCKQ_ROOT`, `DOCKQ_JOBS_BASE_DIR`).
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class DockQSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCKQ_", env_file=".env", extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/dockq_jobs"))

    # DockQ repo root. `DockQ` binary lives in the venv's PATH after `pip install -e .`,
    # so `subprocess_cwd()` doesn't strictly need this — kept so admin can confirm
    # the editable install picked the expected source tree.
    root: Path = Field(default=Path("/opt/dockq"))

    # Absolute path to the `DockQ` CLI entrypoint. Defaults to `DockQ` (resolved
    # via PATH inside the container). Overridable via `DOCKQ_BINARY` env var.
    binary: str = Field(default="DockQ")

    # Per-call upper bound on the number of model PDBs accepted by /api/score_batch.
    # Each invocation is independent (CPU only, seconds-scale) so the limit is
    # mostly to keep job_dirs bounded.
    max_batch_size: int = Field(default=200, ge=1, le=10000)

    # CPU parallelism passed to DockQ's `--n_cpu`. FC instances typically have
    # 8 vCPUs; default matches the instance size.
    default_n_cpu: int = Field(default=8, ge=1, le=64)

    # DockQ is CPU-only, so concurrent jobs don't fight over a GPU. Default 1
    # because DockQ already saturates all cores on a single job. Override via
    # `DOCKQ_MAX_CONCURRENT_JOBS`; capped at 8 to keep memory bounded.
    max_concurrent_jobs: int = Field(default=2, ge=1, le=8)
