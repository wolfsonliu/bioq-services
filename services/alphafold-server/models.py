"""Per-endpoint pydantic request models for alphafold-server.

Single endpoint: `/api/fold` — protein structure prediction using AlphaFold v2.3.2.
"""

from __future__ import annotations

from typing import Literal, Optional

from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401
from bioq_service import default_semantics
from pydantic import BaseModel, Field


class FoldRequest(BaseModel):
    """Request body for `/api/fold`."""

    model_preset: Literal[
        "monomer", "monomer_casp14", "monomer_ptm", "multimer"
    ] = Field(default="monomer_ptm")

    db_preset: Literal["reduced_dbs", "full_dbs"] = Field(default="reduced_dbs")

    max_template_date: str = Field(default="2022-01-01")

    num_multimer_predictions_per_model: int = Field(default=1, ge=1, le=20)

    models_to_relax: Literal["all", "best", "none"] = Field(default="best")

    use_precomputed_msas: bool = Field(default=False)

    random_seed: Optional[int] = Field(
        default=None,
        json_schema_extra=default_semantics("auto", "random seed selected by the tool at runtime"),
    )

    use_gpu_relax: bool = Field(default=True)


__all__ = [
    "FailureKind",
    "FoldRequest",
    "JobInfo",
    "JobStatus",
]
