"""Env-driven config for immunebuilder-server.

All values via pydantic-settings; no `os.getenv` anywhere else in this package.
Env vars use the `IMMUNEBUILDER_` prefix (e.g. `IMMUNEBUILDER_VENV_BIN`).
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class ImmuneBuilderSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="IMMUNEBUILDER_", env_file=".env", extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/immunebuilder_jobs"))

    venv_bin: Path = Field(
        default=Path("/opt/conda/envs/immunebuilder/bin"),
        description="conda env bin/ where ABodyBuilder2/NanoBodyBuilder2/TCRBuilder2 live",
    )

    # Trained model weights — externalized to NAS at
    # /data/models/immunebuilder/trained_model/ (FC mount; SIF / HPC bind via
    # apptainer).  The Docker image contains a symlink at the package's
    # expected path (`/opt/immunebuilder/ImmuneBuilder/ImmuneBuilder/trained_model`)
    # pointing here, so the upstream code finds weights transparently.
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    weights_dir: Path = Field(
        default=Path("/data/models/immunebuilder/trained_model"),
        description="ABody/Nano/TCR weight files (16 .pt files, ~600 MB).",
    )
