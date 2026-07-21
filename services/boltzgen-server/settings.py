"""Env-driven config for boltzgen-server.

All values go through pydantic-settings (no `os.getenv`). Env vars use the
`BOLTZGEN_` prefix (e.g. `BOLTZGEN_ROOT`, `BOLTZGEN_JOBS_BASE_DIR`).
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class BoltzGenSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="BOLTZGEN_", env_file=".env", extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/boltzgen_jobs"))

    root: Path = Field(
        default=Path("/opt/boltzgen"),
        description="Service root inside the container.",
    )

    python: str = Field(
        default="/opt/conda/envs/boltzgen/bin/python",
        description="Python interpreter inside the conda env.",
    )

    cli: str = Field(
        default="/opt/conda/envs/boltzgen/bin/boltzgen",
        description="BoltzGen CLI entry point.",
    )

    weights_dir: Path = Field(
        default=Path("/data/models/boltzgen/weights"),
        description=(
            "Directory containing model checkpoint files.  Externalized to "
            "NAS — must be mounted at /data/models/boltzgen/weights/ (FC) or "
            "bound via `apptainer --bind` (SIF).  See "
            "engineering/decisions/2026-06-26-service-weights-externalization.md."
        ),
    )

    moldir: Path = Field(
        default=Path("/data/models/boltzgen/moldir"),
        description=(
            "CCD molecule directory (unpacked mols.zip).  Externalized to NAS."
        ),
    )
