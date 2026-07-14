"""Env-driven config for esmfold2-server.

All values go through pydantic-settings (no `os.getenv`). Env vars use the
`ESMFOLD2_` prefix (e.g. `ESMFOLD2_ROOT`, `ESMFOLD2_JOBS_BASE_DIR`).
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class ESMFold2Settings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="ESMFOLD2_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/esmfold2_jobs"))

    root: Path = Field(default=Path("/opt/esmfold2"))

    python: str = Field(default="/opt/esmfold2/.venv/bin/python")

    inference_script: str = Field(default="/opt/esmfold2/inference.py")

    model_dir: Path = Field(default=Path("/data/models/esmfold2"))

    esmc_dir: Path = Field(default=Path("/data/models/esmc/6b"))

    ccd_path: Path = Field(default=Path("/data/models/esmfold2/ccd.pkl"))

    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)
