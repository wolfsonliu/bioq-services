"""Argv assembly for turbohopp-server.

Wraps ``services/turbohopp-server/inference.py`` — the custom single-input
adapter around upstream's dataset-only ``evaluate_consistency.py``.

The checkpoint is discovered by scanning ``settings.weights_dir`` for the
first ``*.ckpt`` file.  When the deployer stages multiple candidates
(e.g. distilled at different num_sampling_steps), set env
``TURBOHOPP_CHECKPOINT_NAME`` to pin a specific filename.
"""

from __future__ import annotations

import os
from pathlib import Path

from .models import GenerateRequest
from .settings import TurboHoppSettings


def _resolve_checkpoint(settings: TurboHoppSettings) -> Path:
    """Locate the .ckpt to use.

    Priority:
      1. ``TURBOHOPP_CHECKPOINT_NAME`` env override → weights_dir/<name>
      2. First ``*.ckpt`` found in weights_dir (rglob)

    Falls through with a placeholder path when weights are missing so the
    subprocess emits a clean SystemExit from ``inference.validate()`` rather
    than a Python ImportError.
    """
    wdir = settings.weights_dir
    override = os.environ.get("TURBOHOPP_CHECKPOINT_NAME")
    if override:
        return wdir / override
    if wdir.exists():
        candidates = sorted(wdir.rglob("*.ckpt"))
        if candidates:
            return candidates[0]
    return wdir / "missing.ckpt"


def generate_argv(
    req: GenerateRequest,
    *,
    job_dir: Path,
    input_protein: Path,
    input_molecule: Path,
    settings: TurboHoppSettings,
) -> list[str]:
    """Compose the inference.py argv.

    Output directory is ``<job_dir>/output/`` per project convention; the
    framework's ``JobAdapter.output_dir()`` and ``detect_outputs()`` are wired
    against this.
    """
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _resolve_checkpoint(settings)

    argv = [
        settings.python,
        settings.inference_script,
        "--input_protein", str(input_protein),
        "--input_molecule", str(input_molecule),
        "--output", str(output_dir),
        "--checkpoint", str(checkpoint),
        "--num_samples", str(req.num_samples),
        "--num_sampling_steps", str(req.num_sampling_steps),
    ]
    if req.find_best:
        argv.append("--find_best")
    if req.seed is not None:
        argv += ["--seed", str(req.seed)]
    return argv


__all__ = ["generate_argv"]
