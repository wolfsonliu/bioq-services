"""Argv assembly for flowmol-server.

Wraps `services/flowmol-server/inference.py`, which imports
`flowmol.models.flowmol.FlowMol` directly.  Going through our wrapper
(instead of upstream's `test.py`) lets us:
  - bypass `flowmol.load_pretrained`'s built-in `wget` subprocess
    (unusable behind FC egress restrictions),
  - stub out `wandb` before importing anything that transitively imports
    `flowmol.analysis.metrics` (which has `import wandb` at top level),
  - parameterise the output path to `<job_dir>/output/` — upstream defaults
    to writing under `model_dir/samples/`, but model_dir is on read-only NAS.
"""

from __future__ import annotations

from pathlib import Path

from .models import GenerateRequest
from .settings import FlowMolSettings


def generate_argv(
    req: GenerateRequest,
    *,
    job_dir: Path,
    settings: FlowMolSettings,
) -> list[str]:
    """Compose the inference.py argv.

    The variant's model directory lives at
    `<weights_dir>/trained_models/<variant>/` and must contain
    `checkpoints/last.ckpt` + `config.yaml`.
    """
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = settings.weights_dir / "trained_models" / req.model_variant

    argv = [
        settings.python,
        settings.inference_script,
        "--model-dir", str(model_dir),
        "--output-file", str(output_dir / "molecules.sdf"),
        "--stats-file", str(output_dir / "sampling_stats.json"),
        "--n-mols", str(req.n_mols),
        "--n-timesteps", str(req.n_timesteps),
        "--max-batch-size", str(req.max_batch_size),
    ]
    if req.n_atoms_per_mol is not None:
        argv += ["--n-atoms-per-mol", str(req.n_atoms_per_mol)]
    if req.stochasticity is not None:
        argv += ["--stochasticity", str(req.stochasticity)]
    if req.hc_thresh is not None:
        argv += ["--hc-thresh", str(req.hc_thresh)]
    if req.seed is not None:
        argv += ["--seed", str(req.seed)]
    return argv


__all__ = ["generate_argv"]
