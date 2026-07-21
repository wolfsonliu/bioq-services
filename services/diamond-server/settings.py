"""Env-driven config for diamond-server.

All values go through pydantic-settings (no `os.getenv`). Env vars use the
`DIAMOND_` prefix (e.g. `DIAMOND_DB_DIR`, `DIAMOND_JOBS_BASE_DIR`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from bioq_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class DiamondSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="DIAMOND_", env_file=".env", extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/diamond_jobs"))

    # Absolute path to the DIAMOND binary (vendored prebuilt release, see
    # Dockerfile). Overridable via DIAMOND_BINARY (tests point it at /bin/true).
    binary: str = Field(default="/usr/local/bin/diamond")

    # Root dir holding pre-built reference `.dmnd` databases, mounted from NAS
    # (weights-externalization convention). Used by /api/msa and as the base for
    # blastp/blastx `db_uri` references. Default follows `/data/models/<svc>/`.
    db_dir: Path = Field(default=Path("/data/models/diamond"))

    # Default reference DB for /api/msa, relative to db_dir (e.g. "uniref50.dmnd").
    # Empty → /api/msa requires a caller-supplied `db_uri`.
    msa_db: Optional[str] = Field(default=None)

    # CPU threads handed to each diamond invocation (`-p`). DIAMOND scales well
    # across cores; default matches a typical 8 vCPU FC instance.
    threads: int = Field(default=8, ge=1, le=128)

    # Sensitivity applied when a request omits it. Empty → DIAMOND default (fast).
    default_sensitivity: Optional[str] = Field(default=None)

    # DIAMOND is CPU-only; keep concurrency low since one job already saturates
    # all cores. Override via DIAMOND_MAX_CONCURRENT_JOBS.
    max_concurrent_jobs: int = Field(default=2, ge=1, le=8)
