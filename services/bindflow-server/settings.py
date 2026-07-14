"""Env-driven config for bindflow-server.

All values via pydantic-settings; env_prefix=`BINDFLOW_`.

BindFlow does NOT ship neural-network weights.  The `weights_dir` field is
kept for cross-service uniformity of the `/healthz/detail` probe, but its
contents are unused — force fields and MDP templates ship inside the pip
package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class BindFlowSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="BINDFLOW_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/bindflow_jobs"))

    root: Path = Field(
        default=Path("/opt/bindflow"),
        description="BindFlow install root (parent of upstream/ + server/).",
    )

    python: str = Field(
        default="/opt/conda/envs/bindflow/bin/python",
        description="Python interpreter inside the conda env (has bindflow + gromacs + snakemake on PATH).",
    )

    inference_script: Path = Field(
        default=Path("/opt/bindflow/server/inference.py"),
        description="Wrapper script we subprocess into; imports bindflow.runners.calculate.",
    )

    # GROMACS is installed via bioconda into the conda env — `gmx` is on PATH
    # by way of `PATH=/opt/conda/envs/bindflow/bin:...`.  This optional field
    # exists so the wrapper can source a custom GMXRC if a bind-mounted host
    # GROMACS is preferred over the bundled one.
    gmxrc: Optional[Path] = Field(default=None)

    # Field kept for cross-service uniformity of /healthz/detail; BindFlow has
    # no NN weights.  See design doc §6.5.
    weights_dir: Path = Field(default=Path("/data/models/bindflow"))

    # Long-running workflows: one at a time per instance.
    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)

    # BindFlow is not deployed to FC (workflows exceed 24h); task endpoints
    # remain disabled.  See design doc §3.3 + §11.1.
    task_endpoints_enabled: bool = Field(default=False)

    # Hard ceiling for a single subprocess.  Default 7 days covers the longest
    # FEP campaigns; sbatch time limit is the real constraint on HPC.
    subprocess_timeout_s: int = Field(default=7 * 24 * 3600, ge=60)
