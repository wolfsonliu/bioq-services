"""Env-driven config for promera-server.

All values go through pydantic-settings (no ``os.getenv``).  Env vars use the
``PROMERA_`` prefix (e.g. ``PROMERA_ROOT``, ``PROMERA_JOBS_BASE_DIR``).
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class PromeraSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="PROMERA_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/promera_jobs"))

    root: Path = Field(default=Path("/opt/promera"))

    python: str = Field(default="/opt/promera/.venv/bin/python")

    # Promera + tinyprot + LigandMPNN weights — externalized to NAS at
    # /data/models/promera/{promera,tinyprot,ligandmpnn}/ (FC mount; SIF /
    # HPC bind via apptainer).  LigandMPNN model_params is reached via a
    # symlink baked into the image (/opt/promera/LigandMPNN/model_params →
    # /data/models/promera/ligandmpnn).
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    weights: str = Field(default="/data/models/promera/promera/promera_2606.ckpt")

    ligandmpnn_dir: str = Field(default="/opt/promera/LigandMPNN")

    templates_dir: str = Field(default="/opt/promera/promera_src/examples/templates")

    # tinyprot LMDB cache dir; tinyprot/init.py reads TINYPROT_CACHE at import.
    tinyprot_cache: Path = Field(default=Path("/data/models/promera/tinyprot"))

    max_concurrent_jobs: int = Field(default=1, ge=1, le=8)
