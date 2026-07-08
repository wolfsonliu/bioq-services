"""Env-driven config for semlaflow-server.

All values via pydantic-settings (no `os.getenv`).  Env vars use the
`SEMLAFLOW_` prefix.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import ServiceSettings
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

# SemlaFlow ships two headline pretrained models.  The dataset kind is not a
# free string: semlaflow/scriptutil.py hardcodes QM9_COORDS_STD_DEV /
# GEOM_COORDS_STD_DEV + QM9_BUCKET_LIMITS / GEOM_DRUGS_BUCKET_LIMITS, so
# `dataset` MUST be one of these two.
DATASET_KINDS = ("qm9", "geom-drugs")

# The .smol splits expected under <model>/smol/.  train.smol is mandatory
# even for pure generation because init_metrics() builds the novelty
# reference set from every training SMILES.
SPLIT_NAMES = ("train", "val", "test")


class ModelInfo(BaseModel):
    """One pre-staged SemlaFlow model on NAS.

    A "model" bundles a checkpoint with a reference dataset: unconditional
    generation samples per-molecule atom counts from a `.smol` split and
    always computes novelty against `train.smol` — so both the ckpt and the
    dataset must be co-located on NAS.
    """

    name: str
    dataset: str  # one of DATASET_KINDS
    ckpt_path: Path
    data_dir: Path
    ckpt_present: bool
    splits_present: dict[str, bool]

    @property
    def ready(self) -> bool:
        """Usable for a default (dataset_split=test) generation call."""
        return (
            self.ckpt_present
            and self.splits_present.get("train", False)  # novelty reference
        )


class SemlaFlowSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEMLAFLOW_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/semlaflow_jobs"))

    root: Path = Field(
        default=Path("/opt/semlaflow"),
        description="Service root (subprocess cwd). PYTHONPATH includes this "
        "so `import semlaflow` resolves the vendored upstream package.",
    )

    python: str = Field(
        default="/opt/conda/envs/semlaflow/bin/python",
        description="Python interpreter inside the conda env.",
    )

    inference_script: str = Field(
        default="/opt/semlaflow/server/inference.py",
        description="Service wrapper that reuses semlaflow.predict functions "
        "and dumps the generative metrics table to metrics.json (upstream "
        "only prints it).",
    )

    # Checkpoints + reference datasets externalized to NAS.  Expected layout:
    #   <weights_dir>/<model_name>/model.ckpt
    #   <weights_dir>/<model_name>/smol/{train,val,test}.smol
    #   <weights_dir>/<model_name>/manifest.yaml   (optional: {dataset: ...})
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    weights_dir: Path = Field(
        default=Path("/data/models/semlaflow"),
        description="Root of NAS-mounted weights + reference datasets.",
    )

    default_model: str = Field(
        default="qm9",
        description="Fallback model if request omits one; matches the "
        "pydantic default in models.GenerateRequest.",
    )

    # Single-GPU FC instances run jobs serially.
    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)

    oss_region: str = Field(default="cn-hangzhou")

    # ---- Model registry ----

    def _infer_dataset(self, model_dir: Path) -> str | None:
        """Resolve dataset kind: manifest.yaml wins, else infer from name."""
        manifest = model_dir / "manifest.yaml"
        if manifest.is_file():
            try:
                import yaml

                data = yaml.safe_load(manifest.read_text()) or {}
                ds = str(data.get("dataset", "")).strip()
                if ds in DATASET_KINDS:
                    return ds
            except Exception:
                pass
        name = model_dir.name.lower()
        if "qm9" in name:
            return "qm9"
        if "geom" in name:
            return "geom-drugs"
        return None

    def _model_info(self, model_dir: Path) -> ModelInfo | None:
        dataset = self._infer_dataset(model_dir)
        if dataset is None:
            return None
        ckpt = model_dir / "model.ckpt"
        data_dir = model_dir / "smol"
        return ModelInfo(
            name=model_dir.name,
            dataset=dataset,
            ckpt_path=ckpt,
            data_dir=data_dir,
            ckpt_present=ckpt.is_file(),
            splits_present={
                s: (data_dir / f"{s}.smol").is_file() for s in SPLIT_NAMES
            },
        )

    def list_models(self) -> list[ModelInfo]:
        """Scan weights_dir for pre-staged models."""
        root = self.weights_dir
        if not root.is_dir():
            return []
        out: list[ModelInfo] = []
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            info = self._model_info(d)
            if info is not None:
                out.append(info)
        return out

    def get_model(self, name: str) -> ModelInfo | None:
        for m in self.list_models():
            if m.name == name:
                return m
        return None
