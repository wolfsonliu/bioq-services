"""Runtime settings for rfdiffusion-server.

env_prefix=RFDIFFUSION_; field names are short so the env vars stay clean
(`RFDIFFUSION_ROOT`, not `RFDIFFUSION_RFDIFFUSION_ROOT`).
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class RFdiffusionSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="RFDIFFUSION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Override the framework default so existing FC deployments keep their NAS
    # layout (/data/rfdiffusion_jobs/<id>/...). Env: RFDIFFUSION_JOBS_BASE_DIR
    jobs_base_dir: Path = Field(default=Path("/data/rfdiffusion_jobs"))

    # RFdiffusion source tree (cloned + installed at image build time).
    # Env: RFDIFFUSION_ROOT
    root: Path = Field(default=Path("/opt/rfdiffusion"))

    # Pretrained checkpoints (Base / Complex_base / ActiveSite / ...) baked into
    # the image at build time. `inference.model_directory_path` is set from this
    # so the auto-checkpoint-selector inside run_inference.py finds them.
    # Env: RFDIFFUSION_MODELS_DIR
    models_dir: Path = Field(default=Path("/opt/rfdiffusion/models"))

    # `scripts/run_inference.py` — Hydra-driven entry point that handles every
    # generation mode (unconditional / motif / binder / symmetry / ...).
    # Env: RFDIFFUSION_INFERENCE_SCRIPT
    inference_script: Path = Field(
        default=Path("/opt/rfdiffusion/scripts/run_inference.py")
    )

    # Python interpreter inside the venv created by the Dockerfile. Run_inference
    # is a #!/usr/bin/env python3 script but we invoke it via the venv interpreter
    # so PATH lookup never gets in the way. Env: RFDIFFUSION_PYTHON
    python: Path = Field(default=Path("/opt/rfdiffusion/.venv/bin/python"))

    # OSS download region — only consulted by `oss://` URI resolution.
    # Env: RFDIFFUSION_OSS_REGION
    oss_region: str = Field(default="cn-hangzhou")
