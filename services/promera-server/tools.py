"""Subprocess argv builders for promera-server.

Generates command-line arguments for ``python -m promera`` with appropriate
OmegaConf overrides for each endpoint.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import CofoldRequest, DesignRequest
from .settings import PromeraSettings

_VHH_FRAMEWORK = (
    "EVQLVESGGGLVQPGGSLRLSCAA<cdrh1>MGWFRQAPGKGRELVAA<cdrh2>"
    "YYPDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYC<cdrh3>WGQGTQVTVSS"
)

_VHH_CDR_LENGTHS = {
    "cdrh1": [5, 7],
    "cdrh2": [7, 12],
    "cdrh3": [9, 15],
}


def cofold_argv(
    req: CofoldRequest,
    *,
    job_dir: Path,
    schema_path: Path,
    settings: PromeraSettings,
) -> list[str]:
    """Build argv for cofolding (structure prediction)."""
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    return [
        settings.python,
        "-m",
        "promera",
        "--weights",
        settings.weights,
        f"input={schema_path.parent}",
        f"output={output_dir}",
        f"num_seeds={req.num_seeds}",
        f"diffusion_samples={req.diffusion_samples}",
        f"diffusion_steps={req.diffusion_steps}",
        f"recycling_steps={req.recycling_steps}",
        "assert_msa=false",
        "skip_existing=false",
        f"save_traj={'true' if req.save_trajectory else 'false'}",
        f"save_full_confidence={'true' if req.save_full_confidence else 'false'}",
        f"save_distogram={'true' if req.save_distogram else 'false'}",
    ]


def build_design_config(
    req: DesignRequest,
    *,
    target_dir: Path,
    output_dir: Path,
    template_path: Path | None = None,
    settings: PromeraSettings,
) -> dict:
    """Build the OmegaConf task_config dict for the Design task."""
    cfg: dict = {
        "input": str(target_dir),
        "output": str(output_dir),
        "msa_dir": None,
        "recycling_steps": req.recycling_steps,
        "num_backbones": req.num_backbones,
        "diffusion_steps": req.diffusion_steps,
        "skip_existing": False,
        "save_traj": False,
        "save_distogram": False,
        "save_full_confidence": req.save_full_confidence,
        "epitope_chain": req.epitope_chain,
        "epitope_residues": (
            [int(x.strip()) for x in req.epitope_residues.split(",") if x.strip()]
            if req.epitope_residues
            else []
        ),
        "target_chains": req.target_chains or None,
        "inverse_folder": {
            "type": req.inverse_folder_type,
            "num_seqs": req.inverse_folder_num_seqs,
        },
        "diffusion": {
            "sigma_min": 0.0004,
            "sigma_max": 160.0,
            "sigma_data": 16.0,
            "rho": 7,
            "gamma_0": 0.8,
            "gamma_min": 1.0,
            "noise_scale": 1.003,
            "step_scale": 1.0 if req.design_type == "minibinder" else 1.5,
            "edm_churn": True,
        },
    }

    if req.design_type == "minibinder":
        cfg["binder"] = {
            "chain": req.binder_chain,
            "type": "protein",
            "length": [req.binder_length_min, req.binder_length_max],
        }
    else:
        cfg["binder"] = {
            "chain": req.binder_chain,
            "type": "vhh",
            "framework": _VHH_FRAMEWORK,
            "cdr_lengths": _VHH_CDR_LENGTHS,
            "paratope_from_cdrs": True,
        }

    if template_path is not None:
        cfg["target_template"] = {
            "path": str(template_path),
            "chain": req.target_template_chain,
            "subsample_frac": req.target_template_subsample_frac,
        }

    return cfg


def write_design_config(cfg: dict, config_path: Path) -> Path:
    """Write the design task config to a YAML file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return config_path


def design_argv(
    req: DesignRequest,
    *,
    job_dir: Path,
    config_path: Path,
    settings: PromeraSettings,
) -> list[str]:
    """Build argv for binder design."""
    return [
        settings.python,
        "-m",
        "promera",
        "--task",
        "promera.inference.Design",
        "--task_config",
        str(config_path),
        "--weights",
        settings.weights,
    ]
