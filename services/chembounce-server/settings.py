"""Env-driven config for chembounce-server.

All values via pydantic-settings; env_prefix=`CHEMBOUNCE_`.
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import ServiceSettings
from pydantic import Field, computed_field
from pydantic_settings import SettingsConfigDict


class ChemBounceSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHEMBOUNCE_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/chembounce_jobs"))

    root: Path = Field(
        default=Path("/opt/chembounce/upstream"),
        description="Upstream source root (subprocess cwd).  chembounce.py "
        "uses `sys.path.append(__file__'s dir)` so cwd must be the upstream root.",
    )

    python: str = Field(
        default="/opt/conda/envs/chembounce/bin/python",
        description="Python interpreter inside the conda env.",
    )

    entrypoint: str = Field(
        default="/opt/chembounce/upstream/chembounce.py",
        description="Upstream CLI entrypoint we subprocess into.",
    )

    # Scaffold + fingerprint database — externalized to NAS at
    # /data/models/chembounce/data/ (FC mount; SIF / HPC bind via apptainer).
    # Field name `weights_dir` follows the project-wide convention so
    # /healthz/detail probes work uniformly, even though contents here are
    # scaffold SMILES + fingerprint .npz (not model weights).
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    weights_dir: Path = Field(
        default=Path("/data/models/chembounce/data"),
        description="Scaffold SMILES + Morgan-FP .npz database directory.",
    )

    # File names below match the upstream Zenodo release
    # (https://zenodo.org/records/16741967/files/data.zip).  Upstream's
    # own `install.sh` extracts these exact filenames; our fetch_data.sh
    # + NAS layout keep them as-is (mixed casing / `mw250` suffix
    # ordering / `_processed_` middle) rather than renaming.
    @computed_field  # type: ignore[misc]
    @property
    def scaffold_db_250mw(self) -> Path:
        return self.weights_dir / "Scaffolds_processed_mw250.txt"

    @computed_field  # type: ignore[misc]
    @property
    def scaffold_db_full(self) -> Path:
        return self.weights_dir / "Scaffolds_processed.txt"

    @computed_field  # type: ignore[misc]
    @property
    def fingerprint_250mw(self) -> Path:
        return self.weights_dir / "scaffold_fingerprints_mw250.npz"

    @computed_field  # type: ignore[misc]
    @property
    def fingerprint_full(self) -> Path:
        return self.weights_dir / "scaffold_fingerprints.npz"

    # CPU-bound; FC CPU function defaults to single concurrent job per
    # instance.  Set via env if you want more.
    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)
