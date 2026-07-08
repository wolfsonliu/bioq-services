"""Argv builders — HTTP handlers and CLI batch entry share these.

All 5 modes route to `server.reinvent_cli`. Each builder writes the request
payload to work/params.json (side effect) and appends staged input-file flags.
`files` maps a reinvent_cli file flag (smiles_file / validation_smiles_file /
model_file / prior_file / agent_file / amino_acid_library) to an on-disk path.
"""
from __future__ import annotations

import json
from pathlib import Path

from .models import (
    EnumerationRequest,
    SamplingRequest,
    ScoringRequest,
    StagedLearningRequest,
    TransferLearningRequest,
)


def _emit(run_type: str, req, files: dict[str, Path | None],
          job_dir: Path, settings) -> list[str]:
    work = job_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    params_json = work / "params.json"
    params_json.write_text(json.dumps(req.model_dump(mode="json")))
    argv = [
        str(settings.python), "-m", "server.reinvent_cli",
        "--run-type", run_type,
        "--params-json", str(params_json),
        "--work-dir", str(work),
        "--output-dir", str(job_dir / "output"),
        "--device", (getattr(req, "device", None) or settings.device),
        "--prior-base", str(settings.prior_base),
        "--reinvent-bin", str(settings.reinvent_bin),
    ]
    for flag, path in files.items():
        if path is not None:
            argv += [f"--{flag.replace('_', '-')}", str(path)]
    return argv


def sampling_argv(req: SamplingRequest, files, job_dir, settings) -> list[str]:
    return _emit("sampling", req, files, job_dir, settings)


def scoring_argv(req: ScoringRequest, files, job_dir, settings) -> list[str]:
    return _emit("scoring", req, files, job_dir, settings)


def enumeration_argv(req: EnumerationRequest, files, job_dir, settings) -> list[str]:
    return _emit("enumeration", req, files, job_dir, settings)


def transfer_learning_argv(req: TransferLearningRequest, files, job_dir, settings) -> list[str]:
    return _emit("transfer_learning", req, files, job_dir, settings)


def staged_learning_argv(req: StagedLearningRequest, files, job_dir, settings) -> list[str]:
    return _emit("staged_learning", req, files, job_dir, settings)


__all__ = [
    "sampling_argv", "scoring_argv", "enumeration_argv",
    "transfer_learning_argv", "staged_learning_argv",
]
