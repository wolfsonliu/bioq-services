"""Runtime settings for rfantibody-server.

All env-driven config is consolidated here; the rest of the codebase reads
`settings.foo` rather than calling `os.getenv`. Defaults match the Docker image
layout (`/opt/rfantibody/{,weights,scripts}`).
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class RFantibodySettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="RFANTIBODY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Override the framework default so existing FC deployments keep their
    # NAS layout (/data/rfantibody_jobs/<job_id>/...). Env: RFANTIBODY_JOBS_BASE_DIR
    jobs_base_dir: Path = Field(default=Path("/data/rfantibody_jobs"))

    # RFantibody source tree (cloned + installed at image build time).
    # Env var: RFANTIBODY_ROOT — short field name keeps the env var name clean
    # under env_prefix="RFANTIBODY_".
    root: Path = Field(default=Path("/opt/rfantibody"))

    # Pretrained weights for the three tools — externalized to NAS at
    # /data/models/rfantibody/weights/ (FC mount); SIF / HPC bind via apptainer.
    # Empty / missing files are tolerated at startup; per-tool calls log a
    # warning and fall back to whatever the script's default checkpoint
    # resolution produces.  Env: RFANTIBODY_WEIGHTS_DIR
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    weights_dir: Path = Field(default=Path("/data/models/rfantibody/weights"))

    # CLI entry points that wrap each of the three tools. Env: RFANTIBODY_SCRIPTS_DIR
    scripts_dir: Path = Field(default=Path("/opt/rfantibody/scripts"))

    # OSS download region — only consulted when an `oss://` URI is resolved.
    # Env: RFANTIBODY_OSS_REGION
    oss_region: str = Field(default="cn-hangzhou")
