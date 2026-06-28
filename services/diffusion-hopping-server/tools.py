"""Argv assembly for diffusion-hopping-server.

Wraps the `services/diffusion-hopping-server/inference.py` wrapper script,
which in turn imports the upstream `diffusion_hopping.*` modules.  Going
through our wrapper (instead of upstream's `generate_scaffolds.py`) lets us
parameterize the checkpoint path / variant — upstream hardcodes
`gvp_conditional.ckpt`.
"""

from __future__ import annotations

from pathlib import Path

from .models import GenerateRequest
from .settings import DiffusionHoppingSettings


def generate_argv(
    req: GenerateRequest,
    *,
    job_dir: Path,
    input_molecule: Path,
    input_protein: Path,
    settings: DiffusionHoppingSettings,
) -> list[str]:
    """Compose the inference.py argv.

    Output directory is `<job_dir>/output/` per project convention; the
    framework's `JobAdapter.output_dir()` and `detect_outputs()` are wired
    against this.
    """
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = settings.weights_dir / f"{req.model_variant}.ckpt"

    return [
        settings.python,
        settings.inference_script,
        "--input_molecule", str(input_molecule),
        "--input_protein", str(input_protein),
        "--output", str(output_dir),
        "--num_samples", str(req.num_samples),
        "--checkpoint", str(checkpoint),
        "--variant", req.model_variant,
    ]


__all__ = ["generate_argv"]
