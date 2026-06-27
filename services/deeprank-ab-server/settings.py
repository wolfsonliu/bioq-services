"""Env-driven config for deeprank-ab-server.

All values go through pydantic-settings (no `os.getenv`). Env vars use the
`DEEPRANK_AB_` prefix (e.g. `DEEPRANK_AB_ROOT`, `DEEPRANK_AB_JOBS_BASE_DIR`).
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class DeepRankAbSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="DEEPRANK_AB_", env_file=".env", extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/deeprank_ab_jobs"))

    root: Path = Field(
        default=Path("/opt/deeprank-ab"),
        description="Service root inside the container.",
    )

    python: str = Field(
        default="/opt/conda/envs/deeprank-ab/bin/python",
        description="Python interpreter inside the conda env.",
    )

    inference_script: str = Field(
        default="/opt/deeprank-ab/server/run_inference.py",
        description="Wrapper that patches and runs upstream inference.py.",
    )

    # ESM-2 weights — externalized to NAS at /data/models/deeprank-ab/esm/
    # (FC mount; SIF / HPC bind via apptainer).  Per-service path (NOT shared
    # with other ESM-2 consumers; deeprank-ab has its own copy fetched via
    # scripts/fetch_esm_weights.sh).  The model + contact-regression files
    # are read by `inference.py` via WEIGHT_PATH / REG_WEIGHT_PATH env vars
    # (kept for back-compat) or via the `<weights_dir>/<file>` defaults.
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    weights_dir: Path = Field(
        default=Path("/data/models/deeprank-ab/esm"),
        description="ESM-2 weights root (esm2_t33_650M_UR50D.pt + contact-regression.pt).",
    )

    oss_region: str = Field(default="cn-hangzhou")
