"""Argv assembly for megalodon-server.

Wraps `server/inference.py`, which loads the per-job config, runs its own
sampling loop (so n_atoms_per_mol can be fixed), writes the SDF, and reuses
the upstream metric components to dump metrics.json.

`model_name` selects a (dataset, config, checkpoint) bundle. A per-job config
is synthesized here (statistics paths repointed at NAS) so the subprocess
gets an interpolation-free YAML.
"""

from __future__ import annotations

from pathlib import Path

from .configs import build_config
from .models import MODEL_REGISTRY, GenerateRequest
from .settings import MegalodonSettings


def generate_argv(
    req: GenerateRequest,
    *,
    job_dir: Path,
    settings: MegalodonSettings,
) -> list[str]:
    """Compose the inference.py argv and write the per-job config.yaml."""
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = MODEL_REGISTRY[req.model_name]
    info = settings.get_model(req.model_name)
    ckpt_path = info.ckpt_path if info else settings.ckpt_path(spec.dataset, spec.ckpt_file)
    stats_dir = info.stats_dir if info else settings.stats_dir(spec.dataset)

    job_config = build_config(
        src_config=settings.conf_dir / spec.config_rel,
        stats_dir=stats_dir,
        out_path=job_dir / "config.yaml",
    )

    argv = [
        settings.python,
        settings.inference_script,
        "--config-path", str(job_config),
        "--ckpt-path", str(ckpt_path),
        "--stats-dir", str(stats_dir),
        "--stats-split", "train",
        "--save-dir", str(output_dir),
        "--n-molecules", str(req.n_molecules),
        "--timesteps", str(req.timesteps),
        "--batch-size", str(req.batch_size),
        "--device", "cuda",
    ]
    if req.n_atoms_per_mol is not None:
        argv += ["--n-atoms-per-mol", str(req.n_atoms_per_mol)]
    if req.seed is not None:
        argv += ["--seed", str(req.seed)]
    return argv


__all__ = ["generate_argv"]
