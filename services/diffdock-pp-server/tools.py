"""Argv assembly for diffdock-pp-server.

Wraps our `services/diffdock-pp-server/inference.py`, which builds the
DB5-style temp layout and dispatches into upstream `main_inf.main()` in
process, then post-processes the resulting pickle into `dock_pose_<rank>.pdb`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .models import DockRequest
from .settings import DiffDockPPSettings


def dock_argv(
    req: DockRequest,
    *,
    job_dir: Path,
    receptor: Path,
    ligand: Path,
    settings: DiffDockPPSettings,
    seed: Optional[int] = None,
) -> list[str]:
    """Compose the inference.py argv.

    Output directory is `<job_dir>/output/` per project convention; the
    framework's `JobAdapter.output_dir()` and `detect_outputs()` are wired
    against this.

    `seed` overrides `req.seed` when provided (used by the framework to
    fill in a seed if the request left it None — keeps input_params echo
    honest about what actually ran).
    """
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_seed = seed if seed is not None else (req.seed if req.seed is not None else 0)

    return [
        settings.python,
        settings.inference_script,
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--output", str(output_dir),
        "--num_samples", str(req.num_samples),
        "--actual_steps", str(req.actual_steps),
        "--top_k", str(req.top_k),
        "--use_confidence_model", "true" if req.use_confidence_model else "false",
        "--seed", str(resolved_seed),
        "--mirror_ligand", "true" if req.mirror_ligand else "false",
        "--no_final_noise", "true" if req.no_final_noise else "false",
        "--score_model_dir", str(settings.weights_dir / "large_model_dips" / "fold_0"),
        "--confidence_model_dir", str(settings.weights_dir / "confidence_model_dips" / "fold_0"),
        "--config", str(settings.config_yaml),
        "--torchhub_dir", str(settings.weights_dir / "esm_cache"),
    ]


__all__ = ["dock_argv"]
