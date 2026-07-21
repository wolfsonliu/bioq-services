"""Runtime settings for rfdiffusion2-server.

env_prefix=RFDIFFUSION2_; field names are short so the env vars stay clean
(`RFDIFFUSION2_ROOT`, not `RFDIFFUSION2_RFDIFFUSION2_ROOT`).
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class RFdiffusion2Settings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="RFDIFFUSION2_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Existing FC deployments keep job state on the NAS at
    # /data/rfdiffusion2_jobs/<id>/...  Env: RFDIFFUSION2_JOBS_BASE_DIR
    jobs_base_dir: Path = Field(default=Path("/data/rfdiffusion2_jobs"))

    # Service root (vendored at services/rfdiffusion2-server/upstream/, copied
    # into the image at build time). Env: RFDIFFUSION2_ROOT
    root: Path = Field(default=Path("/opt/rfdiffusion2-server"))

    # Diffusion model checkpoints (RFD_140.pt / RFD_173.pt). The default
    # config (aa.yaml) points at RFD_140.pt via `REPO_ROOT/rf_diffusion/model_weights/`.
    # Env: RFDIFFUSION2_MODELS_DIR
    models_dir: Path = Field(
        default=Path("/opt/rfdiffusion2-server/upstream/rf_diffusion/model_weights")
    )

    # `rf_diffusion/run_inference.py` — Hydra entry point. Default config is
    # `aa.yaml` (atomic motif scaffolding). All other modes are reachable via
    # `--config-name=<other>` + key overrides.
    # Env: RFDIFFUSION2_INFERENCE_SCRIPT
    inference_script: Path = Field(
        default=Path("/opt/rfdiffusion2-server/upstream/rf_diffusion/run_inference.py")
    )

    # Python interpreter inside the conda env created by the Dockerfile.
    # pyrosetta is conda-only (proprietary), openbabel is conda for ABI
    # stability; everything else is pip on top. Env: RFDIFFUSION2_PYTHON
    python: Path = Field(default=Path("/opt/conda/envs/rfd2/bin/python"))

    # The vendored upstream/ tree must be on PYTHONPATH for `rf_diffusion`/
    # `rf2aa` imports; the subprocess CWD is set there for Hydra config
    # resolution. Both happen via `subprocess_cwd()` + the runner's env merge.
    # Env: RFDIFFUSION2_PYTHONPATH
    pythonpath: Path = Field(default=Path("/opt/rfdiffusion2-server/upstream"))
