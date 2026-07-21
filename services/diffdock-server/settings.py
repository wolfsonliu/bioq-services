"""Env-driven config for diffdock-server.

All values via pydantic-settings (no `os.getenv`).  Env vars use the
`DIFFDOCK_` prefix.  See engineering/decisions/2026-07-06-diffdock-server-design.md.
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class DiffdockSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="DIFFDOCK_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/diffdock_jobs"))

    root: Path = Field(
        default=Path("/opt/diffdock"),
        description="Service root (subprocess cwd).  Upstream utils/so3.py "
        "reads `.so3_*.npy` and utils/torus.py reads `.p.npy` + `.score.npy` "
        "via CWD-relative paths — these LUT files are pre-computed at Docker "
        "build time and live under this root.",
    )

    python: str = Field(
        default="/opt/conda/envs/diffdock/bin/python",
        description="Python interpreter inside the conda env.",
    )

    inference_script: Path = Field(
        default=Path("/opt/diffdock/server/run_inference.py"),
        description="Wrapper script that invokes upstream inference.main() "
        "and post-processes rank*.sdf → confidence_scores.json.  Named "
        "``run_inference.py`` to avoid clashing with upstream inference.py.",
    )

    config_yaml: Path = Field(
        default=Path("/opt/diffdock/server/default_inference_args.yaml"),
        description="Bundled upstream default_inference_args.yaml with "
        "DiffDock-L v1.1 paper-tuned temperature parameters.",
    )

    weights_dir: Path = Field(
        default=Path("/data/models/diffdock"),
        description="NAS mount root for score / confidence checkpoints + "
        "ESM cache (see engineering/decisions/2026-06-26-service-weights-"
        "externalization.md).",
    )

    score_model_subdir: str = Field(
        default="score_model",
        description="Score model directory (relative to weights_dir).  "
        "Contains best_ema_inference_epoch_model.pt + model_parameters.yml.  "
        "diffdock_models.zip extracts score_model/ + confidence_model/ at its "
        "top level (the `workdir/v1.1` nesting only appears when upstream "
        "extracts at repo root); fetch_weights.sh unzips into weights_dir "
        "directly so no workdir/v1.1 prefix on NAS.",
    )

    confidence_model_subdir: str = Field(
        default="confidence_model",
        description="Confidence model directory (relative to weights_dir).  "
        "Contains best_model_epoch75.pt + model_parameters.yml.",
    )

    esm_cache_subdir: str = Field(
        default="esm_cache",
        description="TORCH_HOME points here.  Contains hub/checkpoints/"
        "esm2_t33_650M_UR50D.pt (~2.5 GB) and, if protein_sequence input "
        "is used, esmfold_3B_v1.pt (~5 GB).",
    )

    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)

    @property
    def score_model_dir(self) -> Path:
        return self.weights_dir / self.score_model_subdir

    @property
    def confidence_model_dir(self) -> Path:
        return self.weights_dir / self.confidence_model_subdir

    @property
    def esm_cache_dir(self) -> Path:
        return self.weights_dir / self.esm_cache_subdir
