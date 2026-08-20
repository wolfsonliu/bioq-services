"""Per-endpoint pydantic request models.

These describe the *parameter* payloads — the actual PDB / Quiver files come in
as `UploadFile` or are resolved from URIs (see `uris.py`). The framework's
`JobInfo` / `JobStatus` / `FailureKind` are imported directly from
`bioq_service` and re-exported here for backward compatibility with
existing clients.
"""

from __future__ import annotations

from typing import Optional

from bioq_service import FailureKind, JobInfo, JobStatus  # re-exports
from bioq_service import default_semantics
from pydantic import BaseModel, Field

__all__ = [
    "FailureKind",
    "JobInfo",
    "JobStatus",
    "ProteinMPNNRequest",
    "RF2Request",
    "RFdiffusionRequest",
]


class RFdiffusionRequest(BaseModel):
    """Params for `POST /api/rfdiffusion`. Files (`target`, `framework`) are uploaded separately."""

    num_designs: int = Field(default=10, ge=1, le=10000)
    design_loops: str = Field(default="H1:,H2:,H3:")
    hotspots: Optional[str] = Field(
        default=None,
        examples=["B146,B170,B177"],
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    diffuser_t: int = Field(default=50, ge=1, le=200)
    final_step: int = Field(default=1, ge=1)
    deterministic: bool = False
    no_trajectory: bool = True


class ProteinMPNNRequest(BaseModel):
    """Params for `POST /api/proteinmpnn`. Input Quiver comes in as upload or URI."""

    loops: str = Field(default="H1,H2,H3")
    seqs_per_struct: int = Field(default=4, ge=1, le=100)
    temperature: float = Field(default=0.2, ge=0.01, le=2.0)
    omit_aas: str = Field(default="CX")
    deterministic: bool = False


class RF2Request(BaseModel):
    """Params for `POST /api/rf2`. Input Quiver comes in as upload or URI."""

    num_recycles: int = Field(default=10, ge=1, le=50)
    hotspot_show_prop: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: Optional[int] = Field(
        default=None,
        json_schema_extra=default_semantics("auto", "random seed selected by the tool at runtime"),
    )
