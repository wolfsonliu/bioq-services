"""argv builders for boltzgen-server.

Constructs the `boltzgen run` command line for the design and inverse_fold
endpoints. Uses `--no_subprocess` to run all pipeline steps in-process
(single GPU, no nested subprocesses).
"""

from __future__ import annotations

from pathlib import Path

from .models import DesignRequest, InverseFoldRequest
from .settings import BoltzGenSettings


def design_argv(
    req: DesignRequest,
    *,
    job_dir: Path,
    yaml_path: Path,
    settings: BoltzGenSettings,
) -> list[str]:
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        settings.cli,
        "run", str(yaml_path),
        "--output", str(output_dir),
        "--protocol", req.protocol,
        "--num_designs", str(req.num_designs),
        "--budget", str(req.budget),
        "--devices", "1",
        "--num_workers", "1",
        "--use_kernels", req.use_kernels,
        "--no_subprocess",
        "--moldir", str(settings.moldir),
        "--cache", str(settings.weights_dir),
        "--design_checkpoints",
        str(settings.weights_dir / "boltzgen1_diverse.ckpt"),
        str(settings.weights_dir / "boltzgen1_adherence.ckpt"),
        "--inverse_fold_checkpoint", str(settings.weights_dir / "boltzgen1_ifold.ckpt"),
        "--folding_checkpoint", str(settings.weights_dir / "boltz2_conf_final.ckpt"),
        "--affinity_checkpoint", str(settings.weights_dir / "boltz2_aff.ckpt"),
    ]

    if req.diffusion_batch_size is not None:
        argv += ["--diffusion_batch_size", str(req.diffusion_batch_size)]
    if req.step_scale is not None:
        argv += ["--step_scale", req.step_scale]
    if req.noise_scale is not None:
        argv += ["--noise_scale", req.noise_scale]
    if req.skip_inverse_folding:
        argv.append("--skip_inverse_folding")
    if req.inverse_fold_avoid:
        argv += ["--inverse_fold_avoid", req.inverse_fold_avoid]
    if req.inverse_fold_num_sequences != 1:
        argv += ["--inverse_fold_num_sequences", str(req.inverse_fold_num_sequences)]
    if req.alpha is not None:
        argv += ["--alpha", str(req.alpha)]
    if req.filter_biased != "true":
        argv += ["--filter_biased", req.filter_biased]
    if req.reuse:
        argv.append("--reuse")

    # `--config <step> key=value` is boltzgen's escape hatch for overriding
    # per-step Hydra config. We use it here to lower the analysis step's
    # parallel worker count, which defaults to 32 upstream and OOMs on HPC
    # nodes with ≤64 GB RAM (see DesignRequest.analysis_num_processes docs).
    # Format requires a separate `--config` invocation per step.
    if req.analysis_num_processes is not None:
        argv += [
            "--config", "analysis",
            f"num_processes={req.analysis_num_processes}",
        ]

    return argv


def inverse_fold_argv(
    req: InverseFoldRequest,
    *,
    job_dir: Path,
    yaml_path: Path,
    settings: BoltzGenSettings,
) -> list[str]:
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        settings.cli,
        "run", str(yaml_path),
        "--output", str(output_dir),
        "--protocol", req.protocol,
        "--only_inverse_fold",
        "--budget", str(req.budget),
        "--devices", "1",
        "--num_workers", "1",
        "--use_kernels", req.use_kernels,
        "--no_subprocess",
        "--moldir", str(settings.moldir),
        "--cache", str(settings.weights_dir),
        "--inverse_fold_checkpoint", str(settings.weights_dir / "boltzgen1_ifold.ckpt"),
        "--folding_checkpoint", str(settings.weights_dir / "boltz2_conf_final.ckpt"),
        "--affinity_checkpoint", str(settings.weights_dir / "boltz2_aff.ckpt"),
    ]

    if req.inverse_fold_num_sequences != 1:
        argv += ["--inverse_fold_num_sequences", str(req.inverse_fold_num_sequences)]
    if req.inverse_fold_avoid:
        argv += ["--inverse_fold_avoid", req.inverse_fold_avoid]
    if req.alpha is not None:
        argv += ["--alpha", str(req.alpha)]
    if req.filter_biased != "true":
        argv += ["--filter_biased", req.filter_biased]
    if req.reuse:
        argv.append("--reuse")

    # Override analysis-step parallelism to fit memory-constrained nodes.
    # See design_argv() for full rationale; inverse_fold also runs the
    # analysis step so the same OOM trigger applies.
    if req.analysis_num_processes is not None:
        argv += [
            "--config", "analysis",
            f"num_processes={req.analysis_num_processes}",
        ]

    return argv
