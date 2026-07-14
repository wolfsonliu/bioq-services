"""Env-driven config for alphafold-server.

All values go through pydantic-settings (no `os.getenv`). Env vars use the
`ALPHAFOLD_` prefix (e.g. `ALPHAFOLD_ROOT`, `ALPHAFOLD_DATA_DIR`).
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class AlphaFoldSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="ALPHAFOLD_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/alphafold_jobs"))

    root: Path = Field(default=Path("/opt/alphafold"))

    python: str = Field(default="/opt/conda/bin/python")

    data_dir: Path = Field(default=Path("/data/models/alphafold"))

    n_cpu: int = Field(default=8, ge=1, le=64)

    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)
