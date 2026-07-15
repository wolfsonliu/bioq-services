"""Env-driven config for plip-server.

All values go through pydantic-settings (no `os.getenv`). Env vars use the
`PLIP_` prefix (e.g. `PLIP_JOBS_BASE_DIR`, `PLIP_UPSTREAM_DIR`, `PLIP_THREADS`).

PLIP is a rule-based tool with NO model weights, so there is no `weights_dir`
(unlike GPU services). `/healthz/detail` probes upstream-source + openbabel/pymol
importability instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class PlipSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="PLIP_", env_file=".env", extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/plip_jobs"))

    # Vendored PLIP source root (contains the flat `plip/` package). Used as the
    # PYTHONPATH entry so `python -m plip.plipcmd` can `import plip`, and as the
    # subprocess cwd base.
    upstream_dir: Path = Field(default=Path("/opt/plip/upstream"))

    # Python interpreter used to launch `plip.plipcmd`. The venv is created with
    # --system-site-packages so it sees the apt-installed openbabel/pymol/lxml.
    # Tests point this at /bin/true to build argv without really running PLIP.
    python: str = Field(default="/opt/plip/.venv/bin/python")

    # Default `--maxthreads` (binding-site visualization parallelism).
    threads: int = Field(default=4, ge=1, le=128)

    # PLIP is CPU-only and one job can spawn several render threads; keep
    # concurrency low. Override via PLIP_MAX_CONCURRENT_JOBS.
    max_concurrent_jobs: int = Field(default=2, ge=1, le=8)

    # Optional: force a specific PYTHONPATH prefix into the subprocess env. When
    # None, the adapter derives it from upstream_dir.
    pythonpath: Optional[str] = Field(default=None)
