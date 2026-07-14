"""Env-driven config for odesign-server.

All values go through pydantic-settings (no `os.getenv`). Env vars use the
`ODESIGN_` prefix (e.g. `ODESIGN_ROOT`, `ODESIGN_JOBS_BASE_DIR`).
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class ODesignSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="ODESIGN_", env_file=".env", extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/odesign_jobs"))

    root: Path = Field(
        default=Path("/opt/odesign/ODesign"),
        description="ODesign project root inside the container.",
    )

    python: str = Field(
        default="/opt/conda/envs/odesign/bin/python",
        description="Python interpreter.",
    )

    inference_script: str = Field(
        default="/opt/odesign/ODesign/scripts/inference.py",
        description="Path to the ODesign inference.py Hydra entry point.",
    )

    # Externalized to NAS at /data/models/odesign/{ckpt,data}/ since v0.0.5
    # (FC mount; SIF / HPC bind via apptainer).  ckpt/ holds HF checkpoints
    # + grnade.h5; data/ holds CCD components.cif + rdkit_mol.pkl.
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    ckpt_root_dir: Path = Field(
        default=Path("/data/models/odesign/ckpt"),
        description="Directory containing model checkpoint files.",
    )

    data_root_dir: Path = Field(
        default=Path("/data/models/odesign/data"),
        description="Directory containing CCD data (components.cif + rdkit_mol.pkl).",
    )
