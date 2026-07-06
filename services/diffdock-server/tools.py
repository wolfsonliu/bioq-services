"""Argv assembly for diffdock-server.

Wrapper script ``server/inference.py`` invokes upstream ``inference.main()``
and post-processes the results into ``confidence_scores.json``.  This module
builds the subprocess argv from a validated :class:`DockRequest` and the
resolved input paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .models import DockRequest
from .settings import DiffdockSettings


def dock_argv(
    *,
    protein_path: Optional[Path],
    protein_sequence: Optional[str],
    ligand_arg: str,
    out_dir: Path,
    params: DockRequest,
    settings: DiffdockSettings,
) -> list[str]:
    """Build the argv for the wrapper.

    ``ligand_arg`` is either an absolute file path (``str(resolved_path)``)
    or a raw SMILES string; upstream's ``ligand_description`` CLI arg
    accepts either.

    ``protein_path`` and ``protein_sequence`` are mutually exclusive:
    at least one must be non-None (endpoint validates before calling us).
    """
    if not ((protein_path is None) ^ (protein_sequence is None)):
        raise ValueError(
            "Exactly one of protein_path / protein_sequence must be provided"
        )
    argv = [
        settings.python,
        str(settings.inference_script),
        "--ligand", ligand_arg,
        "--complex_name", params.complex_name,
        "--out_dir", str(out_dir),
        "--samples_per_complex", str(params.samples_per_complex),
        "--inference_steps", str(params.inference_steps),
        "--actual_steps", str(params.actual_steps),
        "--batch_size", str(params.batch_size),
        "--no_final_step_noise", str(params.no_final_step_noise).lower(),
        "--save_visualisation", str(params.save_visualisation).lower(),
        "--seed", str(params.seed or 0),
        "--model_dir", str(settings.score_model_dir),
        "--confidence_model_dir", str(settings.confidence_model_dir),
        "--config", str(settings.config_yaml),
        "--torchhub_dir", str(settings.esm_cache_dir),
    ]
    if protein_path is not None:
        argv += ["--protein_path", str(protein_path)]
    else:
        argv += ["--protein_sequence", protein_sequence]
    return argv


__all__ = ["dock_argv"]
