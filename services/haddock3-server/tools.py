"""Argv builders — compose `inference.py <subcommand>` command lines.

Shared by app.py (HTTP) and __main__.py (CLI batch). Outputs land under
`<job_dir>/output/` per project convention.
"""

from __future__ import annotations

from pathlib import Path

from .models import (
    ActpassToAmbigRequest,
    DockRequest,
    ProteinProteinRequest,
    RestrainBodiesRequest,
    ScoreRequest,
)
from .settings import Haddock3Settings


def _base(settings: Haddock3Settings, subcmd: str) -> list[str]:
    return [settings.python, settings.inference_script, subcmd]


def dock_argv(
    _req: DockRequest | ProteinProteinRequest,
    *,
    config_path: Path,
    workdir: Path,
    job_dir: Path,
    settings: Haddock3Settings,
) -> list[str]:
    return _base(settings, "dock") + [
        "--config", str(config_path),
        "--workdir", str(workdir),
        "--output-dir", str(job_dir / "output"),
    ]


def score_argv(
    req: ScoreRequest, *, pdb: Path, job_dir: Path, settings: Haddock3Settings,
) -> list[str]:
    argv = _base(settings, "score") + [
        "--pdb", str(pdb),
        "--output-dir", str(job_dir / "output"),
    ]
    if req.full:
        argv.append("--full")
    for key, value in (req.params or {}).items():
        argv += ["-p", str(key), str(value)]
    return argv


def restrain_bodies_argv(
    req: RestrainBodiesRequest, *, pdb: Path, job_dir: Path, settings: Haddock3Settings,
) -> list[str]:
    argv = _base(settings, "restrain-bodies") + [
        "--pdb", str(pdb),
        "--output-dir", str(job_dir / "output"),
    ]
    if req.exclude:
        argv += ["--exclude", req.exclude]
    return argv


def actpass_to_ambig_argv(
    req: ActpassToAmbigRequest,
    *,
    actpass1: Path,
    actpass2: Path,
    job_dir: Path,
    settings: Haddock3Settings,
) -> list[str]:
    return _base(settings, "actpass-to-ambig") + [
        "--a1", str(actpass1),
        "--a2", str(actpass2),
        "--output-dir", str(job_dir / "output"),
        "--segid1", req.segid1,
        "--segid2", req.segid2,
    ]


__all__ = [
    "dock_argv",
    "score_argv",
    "restrain_bodies_argv",
    "actpass_to_ambig_argv",
]
