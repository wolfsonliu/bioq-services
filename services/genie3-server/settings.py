"""Runtime settings for genie3-server.

env_prefix=GENIE3_; field names are deliberately short so the env vars stay
clean (`GENIE3_ROOT`, not `GENIE3_GENIE3_ROOT`).
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class Genie3Settings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="GENIE3_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Override the framework default so existing FC deployments keep their
    # NAS layout (/data/genie3_jobs/<id>/...). Env: GENIE3_JOBS_BASE_DIR
    jobs_base_dir: Path = Field(default=Path("/data/genie3_jobs"))

    # genie3 source tree (cloned + installed at image build time).
    # Env: GENIE3_ROOT — short field name keeps env var clean under env_prefix.
    root: Path = Field(default=Path("/opt/genie3"))

    # CLI entry point ("genie3" once `pip install -e .` has registered it).
    # Env: GENIE3_BIN
    bin: str = Field(default="genie3")

    # Pretrained checkpoints — externalized to NAS at
    # /data/models/genie3/pretrained/v1/ (FC mount; SIF / HPC bind via
    # apptainer).  The Docker image contains a symlink at
    # /opt/genie3/pretrained → /data/models/genie3/pretrained so the genie3
    # CLI (which looks up `<cwd>/pretrained/<version>/`) resolves them
    # transparently.  Env: GENIE3_PRETRAINED_DIR
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    pretrained_dir: Path = Field(
        default=Path("/data/models/genie3/pretrained/v1"),
        description="Genie3 pretrained v1 checkpoints + config.yaml (~512 MB).",
    )
