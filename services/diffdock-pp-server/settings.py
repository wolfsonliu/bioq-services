"""Env-driven config for diffdock-pp-server.

All values via pydantic-settings (no `os.getenv`). Env vars use the
`DIFFDOCK_PP_` prefix.
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class DiffDockPPSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="DIFFDOCK_PP_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/diffdock_pp_jobs"))

    root: Path = Field(
        default=Path("/opt/diffdock-pp"),
        description="Service root (subprocess cwd). Contains upstream `src/`, "
        "`config/`, and our `server/` — all three referenced by relative paths "
        "in upstream `main_inf.py` / `args.py`.",
    )

    python: str = Field(
        default="/opt/conda/envs/diffdock_pp/bin/python",
        description="Python interpreter inside the conda env.",
    )

    inference_script: str = Field(
        default="/opt/diffdock-pp/server/inference.py",
        description="Our wrapper that constructs a DB5-style temp layout from "
        "two PDB files, invokes upstream `main_inf.main()` in-process, and "
        "post-processes the resulting pickle into `dock_pose_<rank>.pdb`.",
    )

    config_yaml: Path = Field(
        default=Path("/opt/diffdock-pp/server/single_pair_inference.yaml"),
        description="Bundled DiffDock-PP inference config with paper-optimized "
        "temperature sampling parameters. Copied from upstream "
        "`config/single_pair_inference.yaml`.",
    )

    # Externalized weights (see engineering/decisions/2026-06-26-service-weights-externalization.md).
    # Layout under this dir (see fetch_weights.sh):
    #   large_model_dips/fold_0/model_best_*.pth
    #   large_model_dips/args.yaml
    #   confidence_model_dips/fold_0/model_best_*.pth
    #   confidence_model_dips/args.yaml
    #   esm_cache/hub/checkpoints/esm2_t33_650M_UR50D.pt
    #   esm_cache/hub/facebookresearch_esm_main/  (source needed by torch.hub)
    weights_dir: Path = Field(
        default=Path("/data/models/diffdock-pp"),
        description="Root of externalized weights. Score + confidence model "
        "checkpoints and the ESM-2 torch.hub cache all live under here.",
    )

    # Single-GPU FC instances run jobs serially.
    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)
