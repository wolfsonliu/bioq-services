"""Env-driven config for pocketxmol-server.

All values via pydantic-settings (no ``os.getenv``).  Env vars use the
``POCKETXMOL_`` prefix.

See engineering/decisions/2026-07-06-pocketxmol-server-design.md §Configuration.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class PocketXMolSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="POCKETXMOL_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/pocketxmol_jobs"))

    root: Path = Field(
        default=Path("/opt/pocketxmol"),
        description="Service root (subprocess cwd).  Upstream "
        "scripts/sample_use.py does ``sys.path.append('.')`` at import "
        "time and expects to find ``models/`` / ``utils/`` / ``process/`` "
        "under CWD — so cwd must be the vendored root.",
    )

    python: str = Field(
        default="/opt/conda/envs/pocketxmol/bin/python",
        description="Python interpreter inside the conda env.",
    )

    sample_script: Path = Field(
        default=Path("/opt/pocketxmol/scripts/sample_use.py"),
        description="Upstream user-friendly sampling script — used by "
        "dock / sbdd / linking / optimize / pepdesign endpoints.",
    )

    confidence_script: Path = Field(
        default=Path("/opt/pocketxmol/scripts/believe_use_pdb.py"),
        description="Upstream tuned-ranker script — used by /api/confidence.",
    )

    weights_dir: Path = Field(
        default=Path("/data/models/pocketxmol"),
        description="NAS root for PocketXMol weights.  Contains "
        "pxm/checkpoints/, tuned_ranker/checkpoints/, flex_cfd/checkpoints/, "
        "and per-checkpoint train_config/ folders.  "
        "See engineering/decisions/2026-06-26-service-weights-externalization.md.",
    )

    pxm_checkpoint: Path = Field(
        default=Path("/data/models/pocketxmol/pxm/checkpoints/pocketxmol.ckpt"),
        description="Main foundation-model ckpt used by all generation endpoints.",
    )

    tuned_cfd_ckpt: Path = Field(
        default=Path("/data/models/pocketxmol/tuned_ranker/checkpoints/tuned_ranker.ckpt"),
        description="Confidence tuned-ranker ckpt (default variant for /api/confidence).",
    )

    flex_cfd_ckpt: Path = Field(
        default=Path("/data/models/pocketxmol/flex_cfd/checkpoints/flex_cfd.ckpt"),
        description="Flexible-noise confidence ckpt (alternative variant).",
    )

    ccd_dir: Path = Field(
        default=Path("/data/models/pocketxmol/ccd"),
        description="CCD dictionary staged to NAS by fetch_weights.sh.  "
        "Only used by (future) sdf2pdb_robust endpoint; v0.0.1 does not "
        "consult this — probe left off /healthz/detail.",
    )

    # Confidence endpoint YAML lives inside the vendored source; we point
    # to the relative path so subprocess cwd + this string resolves.
    confidence_yaml_dir: Path = Field(
        default=Path("/opt/pocketxmol/configs/sample/confidence"),
        description="Directory containing tuned_cfd.yml / flex_cfd.yml.",
    )

    # PocketXMol is single-GPU heavy; overlap two runs on one card starves
    # both.  Framework will 503 on overflow.
    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)

    oss_region: str = Field(default="cn-hangzhou")
