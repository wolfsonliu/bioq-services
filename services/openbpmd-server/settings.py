"""Env-driven config for openbpmd-server.

All values via pydantic-settings; env_prefix=`OPENBPMD_`.

OpenBPMD does NOT ship neural-network weights — it is a pure OpenMM
metadynamics workflow.  The `weights_dir` field is kept for cross-service
uniformity of the `/healthz/detail` probe, but its contents are unused.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class OpenBPMDSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENBPMD_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/openbpmd_jobs"))

    root: Path = Field(
        default=Path("/opt/openbpmd"),
        description="Service root (parent of upstream/ + server/).",
    )

    python: str = Field(
        default="/opt/conda/envs/openbpmd/bin/python",
        description="Python interpreter inside the conda env (openmm + mdanalysis + mdtraj + parmed).",
    )

    inference_script: str = Field(
        default="/opt/openbpmd/server/inference.py",
        description="Wrapper we subprocess into; injects the simtk->openmm shim then calls openbpmd.main().",
    )

    # OpenMM platform. Production = CUDA (upstream hardcodes it; our wrapper
    # makes it configurable). Offline smoke may use CPU.
    platform: str = Field(
        default="CUDA",
        description="OpenMM platform name (CUDA / CPU / OpenCL). Production must be CUDA.",
    )

    # Field kept for cross-service uniformity of /healthz/detail; OpenBPMD has
    # no NN weights. See design doc §6.7.
    weights_dir: Path = Field(default=Path("/data/models/openbpmd"))

    # One metadynamics workflow at a time per GPU instance.
    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)

    # Hard ceiling for a single subprocess. 48h covers 10 reps x 10 ns on a
    # slow GPU; FC's own 24h async ceiling is the real constraint there.
    subprocess_timeout_s: int = Field(default=48 * 3600, ge=60)
