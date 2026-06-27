"""Env-driven config for ppiflow-server.

All values via pydantic-settings; no `os.getenv` anywhere else in this package.
Env vars use the `PPIFLOW_` prefix (e.g. `PPIFLOW_ROOT`, `PPIFLOW_JOBS_BASE_DIR`).
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class PPIFlowSettings(ServiceSettings):
    """Configuration sourced entirely from environment variables.

    The defaults match the layout produced by the service's Dockerfile: PPIFlow's
    `tool/PPIFlow` directory under `/opt/ppiflow`, checkpoints baked into
    `/opt/ppiflow/checkpoint/`, and jobs persisted on the NAS at
    `/data/ppiflow_jobs/`.
    """

    model_config = SettingsConfigDict(
        env_prefix="PPIFLOW_", env_file=".env", extra="ignore",
    )

    # NAS-shared job root so multi-instance FC + sidecar persistence works.
    jobs_base_dir: Path = Field(default=Path("/data/ppiflow_jobs"))

    # `PPIFLOW_ROOT` — the `tool/PPIFlow/` directory inside the image. All
    # `sample_*.py` scripts are launched with this as cwd, matching the way
    # upstream's README documents the usage.
    root: Path = Field(default=Path("/opt/ppiflow"))

    # Checkpoints — externalized to NAS at /data/models/ppiflow/checkpoint/
    # (FC mount; SIF / HPC bind via apptainer).  4 .ckpt files (binder /
    # antibody / nanobody / monomer) ~1.1 GB total.
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    ckpt_dir: Path = Field(default=Path("/data/models/ppiflow/checkpoint"))

    # Default inference configs shipped with upstream PPIFlow. Endpoints can
    # override on a per-request basis if you need a custom YAML.
    config_dir: Path = Field(default=Path("/opt/ppiflow/configs"))

    # OSS region for downloading inputs via `oss://` URIs (matches the other
    # bioagent services' convention).
    oss_region: str = Field(default="cn-hangzhou")
