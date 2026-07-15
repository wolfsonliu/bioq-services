"""Env-driven config for haddock3-server.

All values via pydantic-settings; env_prefix=`HADDOCK3_`. No `os.getenv`.

HADDOCK3 ships no NN weights.  Its one heavy external dependency is the CNS
(Crystallography & NMR System) engine, which is license-gated and therefore
NOT baked into the image — it is staged to NAS at `cns_exec` (mirroring the
project's weight-externalisation convention) and discovered by haddock3 via
the `CNS_EXEC` env var, injected by the adapter.  CNS-free endpoints
(restraints utilities) run without it.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class Haddock3Settings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="HADDOCK3_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/haddock3_jobs"))

    root: Path = Field(
        default=Path("/opt/haddock3"),
        description="Service root (parent of upstream/ + server/).",
    )

    python: str = Field(
        default="/opt/haddock3/.venv/bin/python",
        description="venv interpreter that has haddock3 + its console scripts installed.",
    )

    inference_script: str = Field(
        default="/opt/haddock3/server/inference.py",
        description="Wrapper we subprocess into; dispatches to the haddock3 console "
        "scripts and normalises stdout-only outputs into files.",
    )

    # The externally-staged, license-gated CNS executable. Treated like a weight:
    # NAS-mounted on FC, `--bind` on SIF. Injected into the subprocess env as
    # CNS_EXEC (see adapter.subprocess_env). Docking/scoring need it; restraints
    # utilities do not.
    cns_exec: Path = Field(default=Path("/data/models/haddock3/cns/cns"))

    # Kept for cross-service uniformity of the /healthz/detail probe (weights_dir
    # points at the parent of the CNS binary). HADDOCK3 has no NN weights.
    weights_dir: Path = Field(default=Path("/data/models/haddock3"))

    default_ncores: int = Field(
        default=8,
        ge=1,
        le=128,
        description="Default CPU cores for a haddock3 run (mode='local').",
    )

    # HADDOCK3 is CPU-only; several light jobs can share an instance. Kept modest
    # so a docking run's multiprocessing fan-out doesn't oversubscribe the box.
    max_concurrent_jobs: int = Field(default=2, ge=1, le=8)
