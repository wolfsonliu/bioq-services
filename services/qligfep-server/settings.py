"""Env-driven config for qligfep-server. All values via pydantic-settings.

Env vars use the ``QLIGFEP_`` prefix.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from bioq_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class QligfepSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="QLIGFEP_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/qligfep_jobs"))
    root: Path = Field(default=Path("/opt/qligfep-server"))
    upstream_dir: Path = Field(default=Path("/opt/qligfep-server/upstream/qligfep"))
    q_bin_dir: Path = Field(default=Path("/opt/Q6/bin"))
    python: Path = Field(default=Path("/opt/conda/envs/qligfep/bin/python"))
    default_cluster: Literal["LOCAL", "CSB", "SLURM"] = "LOCAL"
    max_concurrent_jobs: int = 4
    task_endpoints_enabled: bool = False

    # 保留框架字段惯例（不使用）
    weights_dir: Path = Field(default=Path("/data/models/qligfep"))
