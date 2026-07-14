"""Env-driven config for drughive-server.

All values via pydantic-settings (no `os.getenv`).  Env vars use the
`DRUGHIVE_` prefix.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict


class DrughiveSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="DRUGHIVE_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/drughive_jobs"))

    root: Path = Field(
        default=Path("/opt/drughive"),
        description="Service root (subprocess cwd — upstream reads "
        "`data/pains_filter/PAINS.sieve` via relative path).",
    )

    python: str = Field(
        default="/opt/conda/envs/drughive/bin/python",
        description="Python interpreter inside the conda env.",
    )

    generate_script: Path = Field(
        default=Path("/opt/drughive/generate_molecules.py"),
        description="Upstream ligand generation script — handles both "
        "de novo (/api/generate) and substructure modification "
        "(/api/generate_spatial) via YAML config routing.",
    )

    optimize_script: Path = Field(
        default=Path("/opt/drughive/generate_optimize.py"),
        description="Upstream multi-cycle QVina2 optimization script.",
    )

    weights_dir: Path = Field(
        default=Path("/data/models/drughive/checkpoints"),
        description="NAS mount for pretrained checkpoints "
        "(see engineering/decisions/2026-06-26-service-weights-externalization.md).",
    )

    checkpoint_filename: str = Field(
        default="drughive_model_ch9.ckpt",
        description="Single upstream-released checkpoint from Zenodo "
        "(10.5281/zenodo.12668687).",
    )

    model_id: str = Field(
        default="c9_pdbzinc",
        description="Model ID string upstream expects in YAML config.",
    )

    docking_cmd: str = Field(
        default="qvina2.1",
        description="QVina2 binary name.  Upstream default is `qvina2.1`; "
        "we vendor that exact binary from the QVina github repo "
        "(Apache-2.0, see scripts/vendor_qvina.sh) into the conda env "
        "bin/.  A `qvina2` symlink is also created for flexibility.  "
        "Passed through to upstream YAML config `docking_cmd:` field.",
    )

    # Single-GPU FC instances run jobs serially; /api/optimize is
    # especially GPU/CPU-hungry (QVina2 docking on N thousand mols).
    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)

    @property
    def checkpoint_path(self) -> Path:
        return self.weights_dir / self.checkpoint_filename

    @model_validator(mode="after")
    def _validate_paths(self) -> "DrughiveSettings":
        # Nothing to enforce at construction — files may not exist yet
        # in dev env; /healthz/detail reports missing weights at runtime.
        return self
