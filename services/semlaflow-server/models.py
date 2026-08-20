"""Per-endpoint pydantic request models for semlaflow-server."""

from __future__ import annotations

from typing import Literal, Optional

from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401  (re-exported)
from bioq_service import default_semantics
from pydantic import BaseModel, Field

# SemlaFlow ships two headline pretrained models.  `model_name` selects the
# checkpoint + reference dataset bundle on NAS (see settings.ModelInfo).
ModelName = Literal["qm9", "geom-drugs"]


class GenerateRequest(BaseModel):
    """Unconditional 3D small-molecule generation with SemlaFlow.

    No file inputs — pure parameterised sampling from the flow-matching
    prior.  Upstream reference: `semlaflow/predict.py` (reused in-process by
    our `server/inference.py`).
    """

    model_name: ModelName = Field(
        default="qm9",
        description="Which pre-staged model to use. `qm9` (small molecules, "
        "fast) or `geom-drugs` (drug-like, larger; novelty metric is slow — "
        "builds ~300k RDKit reference mols). Only models present on NAS are "
        "usable — check /api/models or /healthz/detail.",
    )

    n_molecules: int = Field(
        default=100, ge=1, le=10000,
        description="Number of molecules to generate. Atom counts are sampled "
        "(with replacement) from the reference dataset split.",
    )

    integration_steps: int = Field(
        default=100, ge=10, le=500,
        description="Number of ODE integration steps. 100 is the upstream "
        "default; fewer = faster, slightly lower quality.",
    )

    dataset_split: Literal["train", "val", "test"] = Field(
        default="test",
        description="Reference split to sample molecule sizes from. Default "
        "`test` (smallest file). NOTE novelty always additionally loads "
        "train.smol regardless.",
    )

    ode_sampling_strategy: Literal["log", "linear"] = Field(
        default="log",
        description="ODE step schedule. `log` is the upstream default.",
    )

    cat_sampling_noise_level: int = Field(
        default=1, ge=0, le=5,
        description="Categorical sampling noise level for discrete features "
        "(atom types / bonds).",
    )

    batch_cost: int = Field(
        default=8192, ge=256, le=65536,
        description="Bucket batching cost cap (advanced). Higher = larger "
        "batches / more GPU memory.",
    )

    bucket_cost_scale: Literal["linear", "constant"] = Field(
        default="linear",
        description="Cost scaling across size buckets (advanced).",
    )

    seed: Optional[int] = Field(
        default=None,
        description="Random seed (lightning seed_everything). null → upstream "
        "default 12345; recorded in JobInfo.input_params.",
        json_schema_extra=default_semantics("auto", "random seed selected by the tool at runtime"),
    )


__all__ = [
    "FailureKind",
    "GenerateRequest",
    "JobInfo",
    "JobStatus",
    "ModelName",
]
