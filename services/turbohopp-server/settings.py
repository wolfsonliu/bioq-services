"""Env-driven config for turbohopp-server.

All values via pydantic-settings (no `os.getenv`).  Env vars use the
`TURBOHOPP_` prefix.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class TurboHoppSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="TURBOHOPP_",
        env_file=".env",
        extra="ignore",
    )

    # Follows project convention: /data/<svc>_jobs/<id>/... on the NAS mount.
    jobs_base_dir: Path = Field(default=Path("/data/turbohopp_jobs"))

    # Service root (subprocess cwd) — upstream's data transforms use relative
    # paths off cwd for temp files (residue-mode preprocessing).
    root: Path = Field(default=Path("/opt/turbohopp"))

    # Python interpreter inside the conda env baked into the image.
    python: str = Field(
        default="/opt/conda/bin/python",
        description="Python interpreter inside the conda env.",
    )

    # Custom single-input wrapper we ship alongside upstream — upstream's
    # own evaluate_consistency.py is dataset-only (pdbbind_filtered /
    # crossdocked) and cannot accept a single (protein.pdb, ref_ligand.sdf).
    # See services/turbohopp-server/inference.py.
    inference_script: str = Field(
        default="/opt/turbohopp-server/server/inference.py",
    )

    # Consistency-model checkpoints externalized to NAS.
    # Upstream does NOT publish a public checkpoint — users must supply
    # one (train via upstream train_consistency.py, or from authors).
    # /healthz/detail probes for *.ckpt under this dir; weights_loaded=false
    # when empty so agents can detect the "alive-but-no-model" state.
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    weights_dir: Path = Field(
        default=Path("/data/models/turbohopp/checkpoints/v1"),
        description=(
            "Directory containing TurboHopp consistency-model .ckpt file(s). "
            "Populated by the deployer (rsync to NAS)."
        ),
    )

    # Single-GPU FC instances run jobs serially.
    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)

    oss_region: str = Field(default="cn-hangzhou")
