"""Env-driven config for seqkit-server.

All values go through pydantic-settings (no `os.getenv`). Env vars use the
`SEQKIT_` prefix (e.g. `SEQKIT_JOBS_BASE_DIR`, `SEQKIT_BIN`, `SEQKIT_THREADS`).

SeqKit is a static Go binary with NO model weights, so there is no
`weights_dir` (unlike GPU services). `/healthz/detail` probes the binary
instead.
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class SeqkitSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEQKIT_", env_file=".env", extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/seqkit_jobs"))

    # Path to the vendored seqkit binary (static, CPU-only). Tests point this
    # at /bin/true to build argv without really running seqkit.
    bin: Path = Field(default=Path("/opt/seqkit/bin/seqkit"))

    # `-j/--threads` passed to seqkit (it is internally multi-threaded).
    threads: int = Field(default=4, ge=1, le=128)

    # CPU-only tool; keep concurrency modest while seqkit uses its own threads.
    max_concurrent_jobs: int = Field(default=2, ge=1, le=16)


__all__ = ["SeqkitSettings"]
