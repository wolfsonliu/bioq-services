"""Env-driven config + model registry for megalodon-server.

All values via pydantic-settings (no `os.getenv`). Env vars use the
`MEGALODON_` prefix.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from .models import MODEL_REGISTRY

# Core statistics files that every variant needs on NAS: the 3 model-init
# priors + the size distribution + the novelty reference SMILES. Produced by
# the upstream data_processing scripts (save_statistics). drugs_fm needs one
# extra file — handled in _stats_present.
CORE_STATS = (
    "train_atom_types_h.npy",
    "train_bond_types_h.npy",
    "train_charges_prior_h.npy",
    "train_n_h.pickle",
    "train_smiles.pickle",
)


class ModelInfo(BaseModel):
    """One Megalodon variant + its NAS presence status."""

    name: str
    dataset: str  # "qm9" | "drugs"
    objective: str  # "diffusion" | "fm" | "quick"
    config_rel: str
    ckpt_path: Path
    stats_dir: Path
    ckpt_present: bool
    stats_present: bool

    @property
    def ready(self) -> bool:
        return self.ckpt_present and self.stats_present


class MegalodonSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEGALODON_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/megalodon_jobs"))

    root: Path = Field(
        default=Path("/opt/megalodon"),
        description="Upstream repo root (subprocess cwd). PYTHONPATH includes "
        "root + root/src so `import megalodon` and `import server` resolve.",
    )

    python: str = Field(
        default="/opt/conda/envs/megalodon/bin/python",
        description="Python interpreter inside the conda env.",
    )

    inference_script: str = Field(
        default="/opt/megalodon/server/inference.py",
        description="Service wrapper: runs its own sampling loop (so "
        "n_atoms_per_mol works) + reuses upstream metric components + dumps "
        "metrics.json / generation_stats.json.",
    )

    conf_dir: Path = Field(
        default=Path("/opt/megalodon/scripts/conf"),
        description="Vendored upstream config root; config_rel is relative to "
        "this. build_config rewrites the statistics paths per job.",
    )

    # Checkpoints + statistics externalized to NAS. Expected layout:
    #   <weights_dir>/ckpts/<dataset>/<ckpt_file>
    #   <weights_dir>/stats/<dataset>/<statistics files>
    weights_dir: Path = Field(
        default=Path("/data/models/megalodon"),
        description="Root of NAS-mounted checkpoints + statistics bundles.",
    )

    default_model: str = Field(
        default="drugs_diffusion",
        description="Fallback variant; matches GenerateRequest default.",
    )

    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)

    oss_region: str = Field(default="cn-hangzhou")

    # ---- Model registry ----

    def ckpt_path(self, dataset: str, ckpt_file: str) -> Path:
        return self.weights_dir / "ckpts" / dataset / ckpt_file

    def stats_dir(self, dataset: str) -> Path:
        return self.weights_dir / "stats" / dataset

    def _stats_present(self, dataset: str, objective: str) -> bool:
        sd = self.stats_dir(dataset)
        required = list(CORE_STATS)
        # drugs flow-matching config references train_charges_prior.npy (no _h).
        if dataset == "drugs" and objective == "fm":
            required.append("train_charges_prior.npy")
        return all((sd / f).is_file() for f in required)

    def _model_info(self, name: str) -> ModelInfo:
        spec = MODEL_REGISTRY[name]
        ckpt = self.ckpt_path(spec.dataset, spec.ckpt_file)
        return ModelInfo(
            name=name,
            dataset=spec.dataset,
            objective=spec.objective,
            config_rel=spec.config_rel,
            ckpt_path=ckpt,
            stats_dir=self.stats_dir(spec.dataset),
            ckpt_present=ckpt.is_file(),
            stats_present=self._stats_present(spec.dataset, spec.objective),
        )

    def list_models(self) -> list[ModelInfo]:
        return [self._model_info(name) for name in MODEL_REGISTRY]

    def get_model(self, name: str) -> ModelInfo | None:
        if name not in MODEL_REGISTRY:
            return None
        return self._model_info(name)
