"""Argv assembly for semlaflow-server.

Wraps `services/semlaflow-server/inference.py`, which reuses upstream
`semlaflow.predict` module-level functions (load_model / dm_from_ckpt /
save_rdkit_sdf / save_raw_smol) and additionally dumps the generative
metrics table to metrics.json — upstream only prints it to stdout.

The model_name selects a checkpoint + reference dataset bundle on NAS:
  <weights_dir>/<model_name>/model.ckpt
  <weights_dir>/<model_name>/smol/{train,val,test}.smol
The dataset kind (qm9 | geom-drugs) is resolved from the model registry
and passed as --dataset (it fixes coord_std + bucket_limits upstream).
"""

from __future__ import annotations

from pathlib import Path

from .models import GenerateRequest
from .settings import SemlaFlowSettings


def generate_argv(
    req: GenerateRequest,
    *,
    job_dir: Path,
    settings: SemlaFlowSettings,
) -> list[str]:
    """Compose the inference.py argv.

    Resolves model_name -> (ckpt, data_dir, dataset) via the registry when
    available, else falls back to the conventional NAS layout + name-based
    dataset inference so argv assembly stays testable without NAS present.
    """
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    info = settings.get_model(req.model_name)
    if info is not None:
        ckpt_path = info.ckpt_path
        data_dir = info.data_dir
        dataset = info.dataset
    else:
        model_dir = settings.weights_dir / req.model_name
        ckpt_path = model_dir / "model.ckpt"
        data_dir = model_dir / "smol"
        # req.model_name is a Literal["qm9", "geom-drugs"]; it IS the dataset.
        dataset = req.model_name

    argv = [
        settings.python,
        settings.inference_script,
        "--ckpt-path", str(ckpt_path),
        "--data-path", str(data_dir),
        "--dataset", dataset,
        "--save-dir", str(output_dir),
        "--n-molecules", str(req.n_molecules),
        "--integration-steps", str(req.integration_steps),
        "--dataset-split", req.dataset_split,
        "--ode-sampling-strategy", req.ode_sampling_strategy,
        "--cat-sampling-noise-level", str(req.cat_sampling_noise_level),
        "--batch-cost", str(req.batch_cost),
        "--bucket-cost-scale", req.bucket_cost_scale,
    ]
    if req.seed is not None:
        argv += ["--seed", str(req.seed)]
    return argv


__all__ = ["generate_argv"]
