"""Env-driven config for diffusion-hopping-server.

All values via pydantic-settings (no `os.getenv`).  Env vars use the
`DIFFUSION_HOPPING_` prefix.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class DiffusionHoppingSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="DIFFUSION_HOPPING_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/diffusion_hopping_jobs"))

    root: Path = Field(
        default=Path("/opt/diffusion-hopping"),
        description="Service root (subprocess cwd — upstream's data transforms "
        "use relative paths off cwd for temp files).",
    )

    python: str = Field(
        default="/opt/conda/envs/diffusion_hopping/bin/python",
        description="Python interpreter inside the conda env.",
    )

    inference_script: str = Field(
        default="/opt/diffusion-hopping/server/inference.py",
        description="Service wrapper that imports diffusion_hopping.* and "
        "exposes a clean CLI with --checkpoint / --variant flags.  See "
        "services/diffusion-hopping-server/inference.py.",
    )

    # Pretrained checkpoints — externalized to NAS at
    # /data/models/diffusion-hopping/checkpoints/ (FC mount; SIF / HPC bind
    # via apptainer).  Although upstream git ships the 4 ckpts in repo, we
    # follow the project-wide convention and load them from NAS so weights
    # are versionable independently of the image.
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    weights_dir: Path = Field(
        default=Path("/data/models/diffusion-hopping/checkpoints"),
        description="Directory with the 4 .ckpt files "
        "(gvp_conditional / gvp_unconditional / egnn_conditional / egnn_unconditional).",
    )

    # Single-GPU FC instances run jobs serially.
    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)
