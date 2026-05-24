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
        default="/opt/deeprank-ab/DeepRank-Ab/scripts/inference.py",
        description="Absolute path to DeepRank-Ab inference.py.",
    )

    oss_region: str = Field(default="cn-hangzhou")
