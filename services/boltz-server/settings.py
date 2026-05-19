"""Env-driven config for boltz-server.

All values go through pydantic-settings (no `os.getenv`). Env vars use the
`BOLTZ_` prefix (e.g. `BOLTZ_ROOT`, `BOLTZ_JOBS_BASE_DIR`).

Field names that would start with `boltz_` are deliberately shortened so the
env var becomes `BOLTZ_<NAME>` rather than `BOLTZ_BOLTZ_<NAME>` (pydantic-settings
prepends the prefix to the field name verbatim).
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class BoltzSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="BOLTZ_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/boltz_jobs"))

    # Boltz repo / install root. Used as subprocess cwd so any relative paths
    # inside boltz (mostly absent — argv is fully absolute) resolve predictably.
    root: Path = Field(default=Path("/opt/boltz"))

    # Absolute path to the `boltz` CLI entrypoint (click group). Defaults to the
    # venv-installed binary; settable to `boltz` if PATH resolution is desired.
    binary: str = Field(default="/opt/boltz/.venv/bin/boltz")

    # Pre-staged weights + `mols/` CCD directory (copied into the Docker image
    # from `opensource/boltz/weights/`). Passed to `boltz predict --cache` so the
    # CLI doesn't try to download at runtime.
    cache_dir: Path = Field(default=Path("/opt/boltz/weights"))

    # Single-GPU FC instances run jobs serially. Higher values would require
    # multi-GPU scheduling that boltz doesn't currently support out of the box.
    max_concurrent_jobs: int = Field(default=1, ge=1, le=8)

    oss_region: str = Field(default="cn-hangzhou")
