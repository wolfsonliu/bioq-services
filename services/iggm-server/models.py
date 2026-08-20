"""Per-endpoint pydantic request models for iggm-server.

Re-export framework's JobInfo for compatibility with callers that import it
from the service package.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from bioq_service import JobInfo  # noqa: F401  (re-exported)
from bioq_service import default_semantics
from pydantic import BaseModel, Field

# design.py --run_task choices.  affinity_maturation is exposed via its own
# endpoint (extra required fasta_origin + different output cardinality), so the
# /api/design model only offers the three that share the same I/O shape.
DesignTask = Literal["design", "inverse_design", "fr_design"]


class _CommonDesignParams(BaseModel):
    """Sampling knobs shared by /api/design and /api/affinity-maturation."""

    epitope: Optional[List[int]] = Field(
        default=None,
        description="Antigen interface residue numbers (1-based). If omitted, "
        "IgGM's cal_ppi infers the epitope from the complex structure when "
        "possible. Get one from POST /api/epitope.",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    steps: int = Field(
        default=10, ge=1, le=200,
        description="Number of sampling steps (upstream default 10; 200 is the "
        "internal full schedule).",
    )
    num_samples: int = Field(
        default=1, ge=1, le=100,
        description="Number of designs to generate.",
    )
    chunk_size: int = Field(
        default=64, ge=8, le=512,
        description="Chunk size for long-chain inference (memory/speed tradeoff).",
    )
    temperature: float = Field(
        default=1.0, ge=0.1, le=2.0,
        description="Sampling temperature.",
    )
    max_antigen_size: int = Field(
        default=2000, ge=64, le=2000,
        description="Truncate the antigen chain to this many residues to avoid "
        "OOM. Upstream suggests 384 for large antigens.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed. If omitted, the upstream default (42) is used.",
        json_schema_extra=default_semantics("auto", "random seed selected by the tool at runtime"),
    )


class DesignRequest(_CommonDesignParams):
    """POST /api/design — design / inverse_design / fr_design.

    Inputs (multipart, not part of this model): fasta / fasta_uri and
    antigen (PDB) / antigen_uri.
    """

    run_task: DesignTask = Field(
        default="design",
        description="design = CDR sequence + full complex structure co-design; "
        "inverse_design = sequence design on a fixed backbone (FASTA only); "
        "fr_design = framework-region redesign (humanization).",
    )


class AffinityMaturationRequest(_CommonDesignParams):
    """POST /api/affinity-maturation — affinity maturation.

    Requires an extra fasta_origin (the original antibody sequence). Output
    cardinality scales as num_samples x masked-positions, so num_samples
    defaults lower than /api/design (see design doc risk 12.3).
    """

    num_samples: int = Field(
        default=10, ge=1, le=100,
        description="Number of maturation rounds. Total outputs scale as "
        "num_samples x masked positions x (chains-1); keep modest on FC and "
        "use the CLI/Slurm mode for large sweeps.",
    )


class EpitopeRequest(BaseModel):
    """POST /api/epitope — no sampling params, only file inputs (fasta + antigen)."""
