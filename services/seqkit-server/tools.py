"""Subprocess argv builders for seqkit-server.

SeqKit is a single static binary — these builders are pure argv assembly (no
env shims, no PYTHONPATH). Both write straight into `<job_dir>/output/` via
seqkit's global `-o/--out-file` flag.
"""

from __future__ import annotations

from pathlib import Path

from .models import RevcompRequest, StatsRequest
from .settings import SeqkitSettings


def stats_argv(
    req: StatsRequest,
    *,
    job_dir: Path,
    input_fasta: Path,
    settings: SeqkitSettings,
) -> list[str]:
    """Build `seqkit stats --tabular [-all] -o output/stats.tsv <input>`."""
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    argv: list[str] = [str(settings.bin), "stats", "--tabular"]
    if req.all_stats:
        argv.append("--all")
    argv += [
        "-j", str(settings.threads),
        "-o", str(out_dir / "stats.tsv"),
        str(input_fasta),
    ]
    return argv


def revcomp_argv(
    req: RevcompRequest,
    *,
    job_dir: Path,
    input_fasta: Path,
    settings: SeqkitSettings,
) -> list[str]:
    """Build `seqkit seq -r -p [-t <type>] -o output/revcomp.fasta <input>`."""
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    argv: list[str] = [str(settings.bin), "seq", "--reverse", "--complement"]
    if req.seq_type != "auto":
        argv += ["-t", req.seq_type]
    argv += [
        "-j", str(settings.threads),
        "-o", str(out_dir / "revcomp.fasta"),
        str(input_fasta),
    ]
    return argv


__all__ = ["revcomp_argv", "stats_argv"]
