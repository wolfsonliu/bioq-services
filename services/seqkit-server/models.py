"""Per-endpoint pydantic request models for seqkit-server.

File inputs (`input_fasta` / `input_fasta_uri`) are parsed at the route layer
via `File(...)` / `Form(...)`, not on these models. Enum-like constraints are
validated here so bad values are rejected at request parsing (HTTP 422), not
deep inside the argv builder (which would surface as a 500).
"""

from __future__ import annotations

from typing import Literal

from bioq_service import FailureKind, JobInfo, JobStatus
from pydantic import BaseModel, Field

__all__ = [
    "FailureKind",
    "JobInfo",
    "JobStatus",
    "RevcompRequest",
    "StatsRequest",
]

# Sequence alphabets seqkit accepts for `-t/--seq-type`. `auto` lets seqkit
# guess from the first record (it then emits a WARN recommending an explicit
# type for complement — harmless, result unaffected).
_SEQ_TYPES = ("auto", "dna", "rna", "protein")


class StatsRequest(BaseModel):
    """`POST /api/stats` — summary statistics for one FASTA/FASTQ file."""

    all_stats: bool = Field(
        default=True,
        description=(
            "Emit the full `seqkit stats --all` column set (quartiles, N50, "
            "GC content, Q20/Q30, ...). False keeps only the core columns "
            "(num_seqs/sum_len/min/avg/max). Output is always TSV."
        ),
    )


class RevcompRequest(BaseModel):
    """`POST /api/revcomp` — reverse-complement every record in the input."""

    seq_type: Literal["auto", "dna", "rna", "protein"] = Field(
        default="auto",
        description=(
            "Sequence alphabet passed to `seqkit seq -t`. `auto` lets seqkit "
            "detect it; choose `dna`/`rna` to silence seqkit's complement WARN."
        ),
    )


# Keep the tuple importable for validators / docs even if unused above.
SEQ_TYPES = _SEQ_TYPES
