"""CLI batch-mode entry point for drughive-server.

Same Docker image runs the HTTP service (via uvicorn) and this CLI batch
mode (via ``python -m server <endpoint> ...``).  See
engineering/decisions/2026-05-29-cli-batch-mode.md.

Usage::

    python -m server generate \\
        --target /data/pocket.pdb \\
        --ligand /data/ref_ligand.sdf \\
        --output-dir /scratch/results/ \\
        --params-json '{"n_samples": 10, "pdb_id": "5d3h"}'

    python -m server generate_spatial \\
        --target /data/pocket.pdb \\
        --ligand /data/ref_ligand.sdf \\
        --substruct-modify /data/frag.sdf \\
        --output-dir /scratch/results/ \\
        --params-json '{"n_samples": 10, "pdb_id": "4w9f"}'

    python -m server optimize \\
        --target /data/pocket.pdb \\
        --ligand /data/ref_ligand.sdf \\
        --target-pdbqt /data/pocket.pdbqt \\
        --output-dir /scratch/results/ \\
        --params-json '{"key_opt": "affinity_qvina", "n_cycles": 2}'
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import DrughiveAdapter
from .configs import (
    build_generate_config,
    build_generate_spatial_config,
    build_optimize_config,
)
from .models import GenerateRequest, GenerateSpatialRequest, OptimizeRequest
from .settings import DrughiveSettings
from .tools import generate_argv, optimize_argv

settings = DrughiveSettings()
adapter = DrughiveAdapter(settings=settings)


def _write_cfg(cfg: dict, job_dir: Path) -> Path:
    (job_dir / "input").mkdir(parents=True, exist_ok=True)
    path = job_dir / "input" / "config.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _generate_build(
    req: GenerateRequest,
    inputs: dict[str, Path],
    job_dir: Path,
    settings: DrughiveSettings,
) -> list[str]:
    cfg = build_generate_config(
        req=req,
        target_path=inputs["target"],
        ligand_path=inputs["ligand"],
        output_dir=job_dir / "output",
        settings=settings,
    )
    cfg_path = _write_cfg(cfg, job_dir)
    return generate_argv(cfg_path=cfg_path, settings=settings)


def _generate_spatial_build(
    req: GenerateSpatialRequest,
    inputs: dict[str, Path],
    job_dir: Path,
    settings: DrughiveSettings,
) -> list[str]:
    frag: Optional[Path] = inputs.get("substruct_modify")
    # CLI-level XOR check: exactly one of {file, pattern} must be set.
    if frag is not None and req.substruct_modify_pattern is not None:
        raise SystemExit(
            "ERROR: provide either --substruct-modify FILE or "
            "substruct_modify_pattern in --params-json, not both."
        )
    if frag is None and req.substruct_modify_pattern is None:
        raise SystemExit(
            "ERROR: generate_spatial requires either --substruct-modify FILE "
            "or substruct_modify_pattern in --params-json."
        )
    cfg = build_generate_spatial_config(
        req=req,
        target_path=inputs["target"],
        ligand_path=inputs["ligand"],
        output_dir=job_dir / "output",
        settings=settings,
        substruct_modify_path=frag,
    )
    cfg_path = _write_cfg(cfg, job_dir)
    return generate_argv(cfg_path=cfg_path, settings=settings)


def _optimize_build(
    req: OptimizeRequest,
    inputs: dict[str, Path],
    job_dir: Path,
    settings: DrughiveSettings,
) -> list[str]:
    if req.key_opt == "affinity_qvina" and "target_pdbqt" not in inputs:
        raise SystemExit(
            "ERROR: --target-pdbqt is required when key_opt='affinity_qvina'."
        )
    cfg = build_optimize_config(
        req=req,
        target_path=inputs["target"],
        ligand_path=inputs["ligand"],
        target_pdbqt_path=inputs.get("target_pdbqt"),
        output_dir=job_dir / "output",
        settings=settings,
    )
    cfg_path = _write_cfg(cfg, job_dir)
    return optimize_argv(cfg_path=cfg_path, settings=settings)


endpoints = {
    "generate": CLIEndpoint(
        name="generate",
        help="De novo ligand generation from pocket + reference ligand",
        request_model=GenerateRequest,
        build_argv=_generate_build,
        inputs={
            "target": ("Input pocket PDB (.pdb)", True),
            "ligand": ("Input reference ligand (.sdf)", True),
        },
    ),
    "generate_spatial": CLIEndpoint(
        name="generate_spatial",
        help="Scaffold hopping — provide --substruct-modify FILE or "
             "substruct_modify_pattern in --params-json",
        request_model=GenerateSpatialRequest,
        build_argv=_generate_spatial_build,
        inputs={
            "target": ("Input pocket PDB (.pdb)", True),
            "ligand": ("Input reference ligand (.sdf)", True),
            "substruct_modify": (
                "Preserved substructure fragment (.sdf) — alternative to "
                "substruct_modify_pattern SMARTS in --params-json",
                False,
            ),
        },
    ),
    "optimize": CLIEndpoint(
        name="optimize",
        help="Multi-cycle QVina2 affinity/property optimization",
        request_model=OptimizeRequest,
        build_argv=_optimize_build,
        inputs={
            "target": ("Input pocket PDB (.pdb)", True),
            "ligand": ("Input reference ligand (.sdf)", True),
            "target_pdbqt": (
                "Target PDBQT for QVina docking — required when "
                "key_opt='affinity_qvina'",
                False,
            ),
        },
    ),
}


create_cli(adapter, settings, endpoints, version="0.0.1")
