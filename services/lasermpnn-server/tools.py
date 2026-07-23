"""argv builders for lasermpnn-server.

Both endpoints shell out to an upstream batch-inference module invoked as
`python -m LASErMPNN.<script> <input_pdb> <output_dir> <designs_per_input> ...`
from the LASErMPNN package parent dir (settings.root, the subprocess cwd).
Upstream needs no modification — every path/knob it exposes has a CLI flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .models import (
    LIGANDMPNN_WEIGHT,
    MODEL_VARIANT_WEIGHTS,
    DesignLigandMPNNRequest,
    DesignRequest,
)
from .settings import LASErMPNNSettings


def weight_file(model_variant: str, settings: LASErMPNNSettings) -> Path:
    """Absolute NAS path to the checkpoint for a LASErMPNN model_variant."""
    return settings.weights_dir / MODEL_VARIANT_WEIGHTS[model_variant]


def _common_flags(req, out_dir: Path) -> list[str]:
    """Flags shared by both batch scripts (temps + toggles + first-shell)."""
    flags: list[str] = [
        "--designs_per_batch", str(req.designs_per_batch),
        "--disabled_residues", req.disabled_residues,
        "--fs_calc_ca_distance", str(req.fs_calc_ca_distance),
    ]
    if req.sequence_temp is not None:
        flags += ["--sequence_temp", str(req.sequence_temp)]
    if req.first_shell_sequence_temp is not None:
        flags += ["--first_shell_sequence_temp", str(req.first_shell_sequence_temp)]
    if req.chi_temp is not None:
        flags += ["--chi_temp", str(req.chi_temp)]
    if req.fix_beta:
        flags.append("--fix_beta")
    if req.repack_only_input_sequence:
        flags.append("--repack_only_input_sequence")
    if req.ignore_ligand:
        flags.append("--ignore_ligand")
    if req.use_water:
        flags.append("--use_water")
    if req.noncanonical_aa_ligand:
        flags.append("--noncanonical_aa_ligand")
    if req.output_fasta:
        flags.append("--output_fasta")
    if req.fs_no_calc_burial:
        flags.append("--fs_no_calc_burial")
    if req.disable_charged_fs:
        flags.append("--disable_charged_fs")
    return flags


def _base_argv(module: str, input_pdb: Path, out_dir: Path, req,
               weight: Path, settings: LASErMPNNSettings) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        sys.executable, "-m", module,
        str(input_pdb), str(out_dir), str(req.designs_per_input),
        "-w", str(weight),
        "-d", settings.device,
        *_common_flags(req, out_dir),
    ]


def design_argv(
    req: DesignRequest,
    *,
    input_pdb: Path,
    job_dir: Path,
    settings: LASErMPNNSettings,
) -> list[str]:
    """Compose argv for `python -m LASErMPNN.run_batch_inference`."""
    out_dir = job_dir / "output"
    weight = weight_file(req.model_variant, settings)
    argv = _base_argv(
        "LASErMPNN.run_batch_inference", input_pdb, out_dir, req, weight, settings,
    )
    if req.constrain_ala_gly:
        argv += [
            "-c",
            "--ala_budget", str(req.ala_budget),
            "--gly_budget", str(req.gly_budget),
        ]
    return argv


def design_ligandmpnn_argv(
    req: DesignLigandMPNNRequest,
    *,
    input_pdb: Path,
    job_dir: Path,
    settings: LASErMPNNSettings,
) -> list[str]:
    """Compose argv for `python -m LASErMPNN.run_batch_inference_ligandmpnn`."""
    out_dir = job_dir / "output"
    weight = settings.weights_dir / LIGANDMPNN_WEIGHT
    return _base_argv(
        "LASErMPNN.run_batch_inference_ligandmpnn", input_pdb, out_dir, req, weight, settings,
    )
