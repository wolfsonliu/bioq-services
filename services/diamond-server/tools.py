"""Subprocess argv builders for diamond-server.

`makedb_argv` calls the DIAMOND binary directly (single command). The
blastp / blastx / cluster / msa builders invoke the in-package driver
(`python -m server.diamond_driver`), which handles the multi-step flows
(inline makedb before search/cluster, blastp→a3m reconstruction for msa).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from .models import BlastpRequest, BlastxRequest, ClusterRequest, MakedbRequest, MsaRequest
from .settings import DiamondSettings

# Output extension per DIAMOND output format.
_OUTFMT_EXT = {"6": "tsv", "104": "tsv", "0": "txt", "5": "xml", "101": "sam", "103": "paf"}

_DRIVER = ("-m", "server.diamond_driver")


def outfmt_ext(outfmt: str) -> str:
    return _OUTFMT_EXT.get(outfmt, "tsv")


def _search_flags(
    *,
    evalue: float,
    max_target_seqs: int,
    sensitivity: Optional[str],
    threads: int,
) -> list[str]:
    """Driver-facing flags (server.diamond_driver), NOT raw DIAMOND flags.

    The driver re-emits the DIAMOND short flags (-e / -k / --<sensitivity>).
    """
    flags = [
        "--evalue", str(evalue),
        "--max-target-seqs", str(max_target_seqs),
        "-p", str(threads),
    ]
    if sensitivity:
        flags += ["--sensitivity", sensitivity]
    return flags


def makedb_argv(
    req: MakedbRequest,
    *,
    job_dir: Path,
    sequences_path: Path,
    settings: DiamondSettings,
) -> list[str]:
    """Direct `diamond makedb --in <fasta> --db output/<name>` (→ <name>.dmnd)."""
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    threads = req.threads or settings.threads
    return [
        settings.binary, "makedb",
        "--in", str(sequences_path),
        "--db", str(out_dir / req.name),
        "-p", str(threads),
    ]


def _search_argv(
    command: str,
    req,
    *,
    job_dir: Path,
    query_path: Path,
    db_path: Optional[Path],
    subject_path: Optional[Path],
    settings: DiamondSettings,
) -> list[str]:
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    outfmt = req.outfmt
    out_path = out_dir / f"{req.name}.{outfmt_ext(outfmt)}"
    argv = [
        sys.executable, *_DRIVER, "search",
        "--command", command,
        "--query", str(query_path),
        "--output", str(out_path),
        "--diamond-bin", settings.binary,
        "--db-work", str(job_dir / "db"),
        "--outfmt", outfmt,
        *_search_flags(
            evalue=req.evalue,
            max_target_seqs=req.max_target_seqs,
            sensitivity=req.resolved_sensitivity(settings.default_sensitivity),
            threads=req.threads or settings.threads,
        ),
    ]
    if db_path is not None:
        argv += ["--db", str(db_path)]
    if subject_path is not None:
        argv += ["--subject", str(subject_path)]
    return argv


def blastp_argv(
    req: BlastpRequest,
    *,
    job_dir: Path,
    query_path: Path,
    db_path: Optional[Path] = None,
    subject_path: Optional[Path] = None,
    settings: DiamondSettings,
) -> list[str]:
    return _search_argv(
        "blastp", req, job_dir=job_dir, query_path=query_path,
        db_path=db_path, subject_path=subject_path, settings=settings,
    )


def blastx_argv(
    req: BlastxRequest,
    *,
    job_dir: Path,
    query_path: Path,
    db_path: Optional[Path] = None,
    subject_path: Optional[Path] = None,
    settings: DiamondSettings,
) -> list[str]:
    return _search_argv(
        "blastx", req, job_dir=job_dir, query_path=query_path,
        db_path=db_path, subject_path=subject_path, settings=settings,
    )


def cluster_argv(
    req: ClusterRequest,
    *,
    job_dir: Path,
    sequences_path: Path,
    settings: DiamondSettings,
) -> list[str]:
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable, *_DRIVER, "cluster",
        "--algorithm", req.algorithm,
        "--sequences", str(sequences_path),
        "--output", str(out_dir / f"{req.name}.clusters.tsv"),
        "--diamond-bin", settings.binary,
        "--db-work", str(job_dir / "db"),
        "-p", str(req.threads or settings.threads),
    ]
    sens = req.resolved_sensitivity(settings.default_sensitivity)
    if sens:
        argv += ["--sensitivity", sens]
    if req.approx_id is not None:
        argv += ["--approx-id", str(req.approx_id)]
    if req.member_cover is not None:
        argv += ["--member-cover", str(req.member_cover)]
    return argv


def msa_argv(
    req: MsaRequest,
    *,
    job_dir: Path,
    query_path: Path,
    db_path: Path,
    settings: DiamondSettings,
) -> list[str]:
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable, *_DRIVER, "msa",
        "--query", str(query_path),
        "--db", str(db_path),
        "--output", str(out_dir / f"{req.name}.a3m"),
        "--diamond-bin", settings.binary,
        *_search_flags(
            evalue=req.evalue,
            max_target_seqs=req.max_target_seqs,
            sensitivity=req.resolved_sensitivity(settings.default_sensitivity),
            threads=req.threads or settings.threads,
        ),
    ]
    return argv


__all__ = [
    "blastp_argv",
    "blastx_argv",
    "cluster_argv",
    "makedb_argv",
    "msa_argv",
    "outfmt_ext",
]
