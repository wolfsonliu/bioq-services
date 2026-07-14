"""Env-driven config for iggm-server.

All values via pydantic-settings (no `os.getenv`).  Env vars use the
`IGGM_` prefix.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict

# The five pretrained checkpoints IgGM loads via
# IgGM/model/pretrain.py:load_model_hub(<name>).  Upstream downloads them from
# Zenodo (record 16909543) into ./checkpoints/<name>.pth at runtime; the
# service pre-stages them on NAS and symlinks <root>/checkpoints -> weights_dir
# so no runtime download happens.  esm_ppi_650m_ab + igso3_buffer are needed by
# every task; the trunk depends on run_task (see TASK_TRUNK).
CHECKPOINT_NAMES = (
    "esm_ppi_650m_ab",
    "antibody_design_trunk",
    "antibody_inverse_design_trunk",
    "antibody_fr_design_trunk",
    "igso3_buffer",
)

# run_task -> the design-trunk checkpoint it needs.  design and
# affinity_maturation share antibody_design_trunk (see pretrain.py).
TASK_TRUNK = {
    "design": "antibody_design_trunk",
    "affinity_maturation": "antibody_design_trunk",
    "inverse_design": "antibody_inverse_design_trunk",
    "fr_design": "antibody_fr_design_trunk",
}

# Always-needed checkpoints, regardless of run_task.
COMMON_CHECKPOINTS = ("esm_ppi_650m_ab", "igso3_buffer")


class IgGMSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="IGGM_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/iggm_jobs"))

    root: Path = Field(
        default=Path("/opt/iggm"),
        description="Service root (subprocess cwd). PYTHONPATH includes this "
        "so `import IgGM` / `import design` resolve the vendored upstream, and "
        "the ./checkpoints symlink to weights_dir lives here.",
    )

    python: str = Field(
        default="/opt/conda/envs/iggm/bin/python",
        description="Python interpreter inside the conda env.",
    )

    design_script: str = Field(
        default="/opt/iggm/server/run_design.py",
        description="Thin wrapper around upstream design.py: injects seed, "
        "pre-validates the FASTA (chain count / antigen-last / X presence) and "
        "calls design.predict for design / inverse_design / fr_design / "
        "affinity_maturation.",
    )

    epitope_script: str = Field(
        default="/opt/iggm/server/epitope.py",
        description="Thin wrapper reusing IgGM.protein.cal_ppi to dump the "
        "interface epitope residue list to epitope.json (upstream only prints).",
    )

    # Checkpoints externalized to NAS.  Expected layout:
    #   <weights_dir>/<name>.pth   for name in CHECKPOINT_NAMES
    # A symlink /opt/iggm/checkpoints -> weights_dir (created in the Dockerfile)
    # makes upstream's hardcoded os.getcwd()/checkpoints path resolve here.
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    weights_dir: Path = Field(
        default=Path("/data/models/iggm"),
        description="Root of NAS-mounted IgGM checkpoints (.pth files).",
    )

    # Single-GPU FC instances run jobs serially.
    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)

    # ---- Checkpoint probe helpers ----

    def checkpoint_path(self, name: str) -> Path:
        return self.weights_dir / f"{name}.pth"

    def checkpoints_status(self) -> dict[str, bool]:
        """Presence of each of the five .pth files on NAS."""
        return {n: self.checkpoint_path(n).is_file() for n in CHECKPOINT_NAMES}

    def required_checkpoints(self, run_task: str) -> list[str]:
        """Checkpoints a given run_task needs (common + its trunk)."""
        trunk = TASK_TRUNK.get(run_task)
        names = list(COMMON_CHECKPOINTS)
        if trunk:
            names.append(trunk)
        return names

    def missing_checkpoints(self, run_task: str) -> list[str]:
        return [
            n for n in self.required_checkpoints(run_task)
            if not self.checkpoint_path(n).is_file()
        ]
