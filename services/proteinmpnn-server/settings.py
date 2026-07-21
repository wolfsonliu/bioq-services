"""Env-driven config for proteinmpnn-server.

All values via pydantic-settings; no `os.getenv` anywhere else in this package.
Env vars use the `PROTEINMPNN_` prefix (e.g. `PROTEINMPNN_ROOT`).
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class ProteinMPNNSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="PROTEINMPNN_", env_file=".env", extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/proteinmpnn_jobs"))

    # ProteinMPNN repo root; subprocess cwd for `protein_mpnn_run.py` &
    # `helper_scripts/*.py`. All four weight folders live below this by default.
    root: Path = Field(default=Path("/opt/proteinmpnn"))

    # Parent of `vanilla_model_weights/`, `soluble_model_weights/`,
    # `ca_model_weights/`, `AbMPNN_model_weights/`. Kept separate from `root`
    # so future deployments can mount weights from NAS.
    weights_dir: Path = Field(default=Path("/opt/proteinmpnn"))
