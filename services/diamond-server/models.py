"""Per-endpoint pydantic request models for diamond-server.

blastp / blastx / msa share the common search knobs in `_SearchCommon`
(sensitivity / evalue / max_target_seqs / threads). File and URI inputs
(query / subject / db / sequences) are parsed at the route layer via
`File(...)` / `Form(...)`, not carried on these models.

Enum-like fields (sensitivity / outfmt / algorithm) are validated with pydantic
field validators so bad values are rejected at request parsing (HTTP 422), not
deep inside the argv builder (which would surface as a 500).
"""

from __future__ import annotations

from typing import Optional

from bioq_service import FailureKind, JobInfo, JobStatus  # noqa: F401 (re-exports)
from bioq_service import default_semantics
from pydantic import BaseModel, Field, field_validator

__all__ = [
    "BlastpRequest",
    "BlastxRequest",
    "ClusterRequest",
    "FailureKind",
    "JobInfo",
    "JobStatus",
    "MakedbRequest",
    "MsaRequest",
]

# DIAMOND sensitivity ladder (fast is the implicit default when unset).
_SENSITIVITIES = (
    "fast", "mid-sensitive", "sensitive", "more-sensitive",
    "very-sensitive", "ultra-sensitive",
)

# Tabular / textual output formats we allow through (a subset of DIAMOND's set;
# DAA(100) and taxonomy(102) are out of scope for v0.0.1).
_OUTFMTS = ("6", "0", "5", "101", "103", "104")

_ALGORITHMS = ("cluster", "deepclust", "linclust")

_NAME_PATTERN = r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$"


class _SearchCommon(BaseModel):
    """Alignment knobs shared by blastp / blastx / msa."""

    sensitivity: Optional[str] = Field(
        default=None,
        description=(
            "DIAMOND sensitivity ladder: fast / mid-sensitive / sensitive / "
            "more-sensitive / very-sensitive / ultra-sensitive. None → server "
            "default (DIAMOND's fast mode)."
        ),
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    evalue: float = Field(
        default=0.001, gt=0,
        description="Maximum E-value to report an alignment (DIAMOND -e).",
    )
    max_target_seqs: int = Field(
        default=25, ge=1, le=100000,
        description="Max target sequences reported per query (DIAMOND -k).",
    )
    threads: Optional[int] = Field(
        default=None, ge=1, le=128,
        description="Override DIAMOND_THREADS for this call (DIAMOND -p).",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )

    @field_validator("sensitivity")
    @classmethod
    def _check_sensitivity(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _SENSITIVITIES:
            raise ValueError(f"invalid sensitivity {v!r}; allowed: {', '.join(_SENSITIVITIES)}")
        return v

    def resolved_sensitivity(self, default: Optional[str]) -> Optional[str]:
        """Request value wins; fall back to the server default when unset."""
        return self.sensitivity or default


class _SearchBase(_SearchCommon):
    name: str = Field(
        default="diamond", min_length=1, max_length=64, pattern=_NAME_PATTERN,
        description="Output basename stem (output/<name>.<ext>).",
    )
    outfmt: str = Field(
        default="6",
        description=(
            "DIAMOND output format: 6(tab, default) / 0(pairwise) / 5(xml) / "
            "101(sam) / 103(paf) / 104(json-flat)."
        ),
    )

    @field_validator("outfmt")
    @classmethod
    def _check_outfmt(cls, v: str) -> str:
        if v not in _OUTFMTS:
            raise ValueError(f"invalid outfmt {v!r}; allowed: {', '.join(_OUTFMTS)}")
        return v


class BlastpRequest(_SearchBase):
    """`POST /api/blastp` — protein query vs a protein DB."""


class BlastxRequest(_SearchBase):
    """`POST /api/blastx` — translated DNA query vs a protein DB."""


class MsaRequest(_SearchCommon):
    """`POST /api/msa` — DIAMOND→a3m homolog MSA for a protein query."""

    name: str = Field(
        default="query", min_length=1, max_length=64, pattern=_NAME_PATTERN,
        description="a3m basename stem (output/<name>.a3m).",
    )
    max_target_seqs: int = Field(
        default=2000, ge=1, le=100000,
        description="MSA depth — max homologs pulled per query (DIAMOND -k).",
    )


class ClusterRequest(_SearchCommon):
    """`POST /api/cluster` — small-scale protein sequence clustering."""

    name: str = Field(
        default="diamond", min_length=1, max_length=64, pattern=_NAME_PATTERN,
        description="Output basename stem (output/<name>.clusters.tsv).",
    )
    algorithm: str = Field(
        default="cluster",
        description="Clustering algorithm: cluster (cascaded) / deepclust / linclust.",
    )
    approx_id: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="Minimum approx. identity%% to cluster sequences (--approx-id).",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
    member_cover: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="Minimum member coverage%% (--member-cover).",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )

    @field_validator("algorithm")
    @classmethod
    def _check_algorithm(cls, v: str) -> str:
        if v not in _ALGORITHMS:
            raise ValueError(f"invalid algorithm {v!r}; allowed: {', '.join(_ALGORITHMS)}")
        return v


class MakedbRequest(BaseModel):
    """`python -m server makedb` (CLI-only) — build a `.dmnd` from a protein FASTA."""

    name: str = Field(
        default="ref", min_length=1, max_length=64, pattern=_NAME_PATTERN,
        description="Database basename stem (output/<name>.dmnd).",
    )
    threads: Optional[int] = Field(
        default=None, ge=1, le=128,
        description="Override DIAMOND_THREADS for this call (DIAMOND -p).",
        json_schema_extra=default_semantics("unset", "only used when explicitly provided"),
    )
