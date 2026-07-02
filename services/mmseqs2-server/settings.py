"""Env-driven config for mmseqs2-server.

All values go through pydantic-settings (no `os.getenv`). Env vars use the
`MMSEQS2_` prefix (e.g. `MMSEQS2_DB_DIR`, `MMSEQS2_JOBS_BASE_DIR`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class MMseqs2Settings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="MMSEQS2_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/mmseqs2_jobs"))

    # Absolute path to the GPU-enabled MMseqs2 binary baked into the Docker
    # image (see Dockerfile). Override at deploy time only if you bake a
    # different layout.
    mmseqs_binary: str = Field(default="/opt/mmseqs-gpu/bin/mmseqs")

    # Root directory under which pre-built ColabFold-style MMseqs2 databases
    # live (UniRef30 GPU subset + ColabFoldDB env). Mounted from NAS at runtime
    # so the image stays small; see scripts/prepare_databases.sh. Default
    # follows the `/data/models/<svc>/` weights externalization convention
    # (engineering/decisions/2026-06-26-service-weights-externalization.md).
    db_dir: Path = Field(default=Path("/data/models/mmseqs2"))

    # UniRef30 database name (relative to db_dir). Default follows the
    # ColabFold-canonical release naming (`uniref30_YYMM_db`) — this is
    # what `mmseqs databases UniRef30` produces and what upstream ColabFold
    # notebooks reference. Override via MMSEQS2_DEFAULT_DB if you use a
    # differently-named DB (e.g. a `_gpu` subset preformatted with
    # `mmseqs createindex --split 0 --index-subset 2`).
    default_db: str = Field(default="uniref30_2302_db")

    # Environmental DB (ColabFoldDB). `None` disables the env-mode branch so
    # the orchestrator runs UniRef30-only (mode = "all" / "nofilter"). Default
    # is the standard ColabFold envDB 202108 release.
    env_db: Optional[str] = Field(default="colabfold_envdb_202108_db")

    # Whether to run mmseqs in GPU mode. Set to False to force CPU fallback
    # (e.g. local dev without a CUDA card).
    gpu_enabled: bool = Field(default=True)

    # CPU threads handed to each mmseqs invocation (search / pairaln / etc.).
    # FC GPU instances usually have 4-8 vCPU — keep this conservative.
    threads: int = Field(default=4, ge=1, le=64)

    # FC session affinity: POST responses including a job_id will set this
    # header so FC binds follow-up GETs (status, download) to the same
    # instance. Naming rules: no `x-fc-` prefix, letter-start, 5-40 chars,
    # `[a-zA-Z0-9_-]`.
    session_header_name: Optional[str] = Field(default="bioagent-session-id")
