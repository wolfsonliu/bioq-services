"""Pure helpers to turn DIAMOND blastp tabular output into an a3m MSA.

DIAMOND has no native MSA output (see the design doc). The `/api/msa` endpoint
runs blastp with a custom `--outfmt 6` field set that includes the aligned query
(`qseq`) and subject (`sseq`) strings, then this module reconstructs a
query-anchored a3m: the query is column-defining (uppercase, full length), each
homolog is aligned onto the query columns with gaps as `-` and query-relative
insertions as lowercase.

Kept side-effect-free so it can be unit-tested without a DIAMOND binary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Column order requested from `diamond blastp --outfmt 6 ...`. Must match
# tools.py::_MSA_OUTFMT_FIELDS exactly.
BLAST_FIELDS = (
    "qseqid", "sseqid", "pident", "length",
    "qstart", "qend", "sstart", "send",
    "evalue", "bitscore", "qseq", "sseq",
)


@dataclass
class Hit:
    """One HSP row from DIAMOND tabular output (the fields we need)."""

    sseqid: str
    qstart: int
    qend: int
    qseq: str
    sseq: str
    bitscore: float


def parse_blast_tab(path: Path) -> list[Hit]:
    """Parse a `--outfmt 6` file whose columns are `BLAST_FIELDS`.

    Malformed / short lines are skipped. Returns hits in file order (DIAMOND
    already orders HSPs by descending bitscore within a query).
    """
    idx = {name: i for i, name in enumerate(BLAST_FIELDS)}
    hits: list[Hit] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < len(BLAST_FIELDS):
                continue
            try:
                hits.append(
                    Hit(
                        sseqid=cols[idx["sseqid"]],
                        qstart=int(cols[idx["qstart"]]),
                        qend=int(cols[idx["qend"]]),
                        qseq=cols[idx["qseq"]],
                        sseq=cols[idx["sseq"]],
                        bitscore=float(cols[idx["bitscore"]]),
                    )
                )
            except (ValueError, KeyError):
                continue
    return hits


def _align_hit_to_query(query_len: int, hit: Hit) -> str:
    """Render one hit as an a3m line aligned to the query columns.

    - Query columns before `qstart` / after `qend` → gaps (`-`).
    - Match columns (qseq residue) → subject residue uppercase, or `-` when the
      subject has a gap there.
    - Query-gap columns (insertion relative to query) → subject residue
      lowercase (does not consume a query column).

    Invariant: the number of non-lowercase characters equals `query_len`.
    """
    out: list[str] = ["-"] * (hit.qstart - 1)
    for qc, sc in zip(hit.qseq, hit.sseq):
        if qc == "-":
            if sc != "-":
                out.append(sc.lower())
        else:
            out.append(sc.upper() if sc != "-" else "-")
    out.append("-" * (query_len - hit.qend))
    return "".join(out)


def reconstruct_a3m(query_id: str, query_seq: str, hits: Iterable[Hit]) -> str:
    """Build a3m text: query as record 0, then one line per hit (best HSP each).

    Only the first (highest-bitscore) HSP per `sseqid` is kept. Duplicate
    subject ids get a numeric suffix so headers stay unique.
    """
    query_seq = query_seq.upper()
    query_len = len(query_seq)
    lines = [f">{query_id}", query_seq]

    seen: dict[str, int] = {}
    for hit in hits:
        if hit.sseqid in seen:
            continue
        seen[hit.sseqid] = 1
        header = hit.sseqid
        lines.append(f">{header}")
        lines.append(_align_hit_to_query(query_len, hit))

    return "\n".join(lines) + "\n"


def read_first_fasta(path: Path) -> tuple[str, str]:
    """Return (id, sequence) of the first FASTA record; sequence uppercased.

    Raises ValueError if the file has no record.
    """
    header: str | None = None
    seq_parts: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    break
                header = line[1:].strip().split()[0] if len(line) > 1 else "query"
            elif header is not None:
                seq_parts.append(line)
    if header is None or not seq_parts:
        raise ValueError(f"no FASTA record found in {path}")
    return header, "".join(seq_parts).upper()


__all__ = [
    "BLAST_FIELDS",
    "Hit",
    "parse_blast_tab",
    "read_first_fasta",
    "reconstruct_a3m",
]
