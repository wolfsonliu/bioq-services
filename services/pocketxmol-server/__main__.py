"""CLI batch-mode entry point for pocketxmol-server.

Same Docker image runs the HTTP service (via uvicorn) and this CLI batch
mode (via ``python -m server <endpoint> ...``).  See
engineering/decisions/2026-05-29-cli-batch-mode.md.

Usage::

    python -m server dock \\
        --protein /data/8C7Y_TXV_protein.pdb \\
        --ligand /data/8C7Y_TXV_ligand.sdf \\
        --output-dir /scratch/results/ \\
        --params-json '{"num_samples": 10, "pocket_coord": [-8.257, 85.181, 19.050], "pocket_radius": 15}'

    python -m server sbdd \\
        --protein /data/2ar9_A.pdb \\
        --output-dir /scratch/results/ \\
        --params-json '{"pocket_coord": [-8.16, 36.70, 38.77], "num_samples": 50}'

    python -m server linking \\
        --protein /data/target.pdb \\
        --input-ligand /data/parent.sdf \\
        --output-dir /scratch/results/ \\
        --params-json '{"fragments": [[0,1,2,3,4,5,6]], "mol_size_mean": 28}'

    python -m server pepdesign \\
        --protein /data/3bik_A.pdb \\
        --ref-ligand /data/3bik_A_pocket_coord.sdf \\
        --output-dir /scratch/results/ \\
        --params-json '{"mode": "denovo_linear", "pep_length": 10, "pocket_radius": 20}'
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import PocketXMolAdapter
from .configs import (
    build_dock_config,
    build_linking_config,
    build_model_config,
    build_optimize_config,
    build_pepdesign_config,
    build_sbdd_config,
    confidence_yaml_path,
)
from .models import (
    ConfidenceRequest,
    DockRequest,
    LinkingRequest,
    OptimizeRequest,
    PepDesignMode,
    PepDesignRequest,
    SbddRequest,
)
from .settings import PocketXMolSettings
from .tools import confidence_argv, sample_argv

settings = PocketXMolSettings()
adapter = PocketXMolAdapter(settings=settings)


def _write_yaml(cfg: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _dump_cfg_pair(
    task_cfg: dict, job_dir: Path, settings_: PocketXMolSettings,
) -> tuple[Path, Path]:
    task_path = _write_yaml(task_cfg, job_dir / "input" / "task_config.yml")
    model_cfg = build_model_config(settings_)
    model_path = _write_yaml(model_cfg, job_dir / "input" / "model_config.yml")
    return task_path, model_path


# ---------------------------------------------------------------------------
# Per-endpoint build_argv callbacks.
# ---------------------------------------------------------------------------
def _dock_build(
    req: DockRequest,
    inputs: dict[str, Path],
    job_dir: Path,
    settings_: PocketXMolSettings,
) -> list[str]:
    have_ligand = "ligand" in inputs
    have_smiles = req.smiles is not None
    have_pepseq = req.pep_sequence is not None
    if sum([have_ligand, have_smiles, have_pepseq]) != 1:
        raise SystemExit(
            "ERROR: provide exactly one of --ligand, smiles, or pep_sequence."
        )
    output_dir = job_dir / "output"
    cfg = build_dock_config(
        req=req,
        protein_path=inputs["protein"],
        ligand_path=inputs.get("ligand"),
        ref_ligand_path=inputs.get("ref_ligand"),
        output_dir=output_dir,
    )
    task_yml, model_yml = _dump_cfg_pair(cfg, job_dir, settings_)
    return sample_argv(
        task_config_path=task_yml, model_config_path=model_yml,
        output_dir=output_dir, settings=settings_, batch_size=req.batch_size,
    )


def _sbdd_build(
    req: SbddRequest,
    inputs: dict[str, Path],
    job_dir: Path,
    settings_: PocketXMolSettings,
) -> list[str]:
    output_dir = job_dir / "output"
    cfg = build_sbdd_config(
        req=req, protein_path=inputs["protein"], output_dir=output_dir,
    )
    task_yml, model_yml = _dump_cfg_pair(cfg, job_dir, settings_)
    return sample_argv(
        task_config_path=task_yml, model_config_path=model_yml,
        output_dir=output_dir, settings=settings_, batch_size=req.batch_size,
    )


def _linking_build(
    req: LinkingRequest,
    inputs: dict[str, Path],
    job_dir: Path,
    settings_: PocketXMolSettings,
) -> list[str]:
    output_dir = job_dir / "output"
    cfg = build_linking_config(
        req=req,
        protein_path=inputs["protein"],
        input_ligand_path=inputs["input_ligand"],
        output_dir=output_dir,
    )
    task_yml, model_yml = _dump_cfg_pair(cfg, job_dir, settings_)
    return sample_argv(
        task_config_path=task_yml, model_config_path=model_yml,
        output_dir=output_dir, settings=settings_, batch_size=req.batch_size,
    )


def _optimize_build(
    req: OptimizeRequest,
    inputs: dict[str, Path],
    job_dir: Path,
    settings_: PocketXMolSettings,
) -> list[str]:
    output_dir = job_dir / "output"
    cfg = build_optimize_config(
        req=req,
        protein_path=inputs["protein"],
        input_ligand_path=inputs["input_ligand"],
        output_dir=output_dir,
    )
    task_yml, model_yml = _dump_cfg_pair(cfg, job_dir, settings_)
    return sample_argv(
        task_config_path=task_yml, model_config_path=model_yml,
        output_dir=output_dir, settings=settings_, batch_size=req.batch_size,
    )


def _pepdesign_build(
    req: PepDesignRequest,
    inputs: dict[str, Path],
    job_dir: Path,
    settings_: PocketXMolSettings,
) -> list[str]:
    pep_path: Optional[Path] = inputs.get("input_peptide")
    if req.mode in (PepDesignMode.inverse_fold, PepDesignMode.sc_pack) and pep_path is None:
        raise SystemExit(
            f"ERROR: --input-peptide is required for mode={req.mode.value}."
        )
    output_dir = job_dir / "output"
    cfg = build_pepdesign_config(
        req=req,
        protein_path=inputs["protein"],
        input_peptide_path=pep_path,
        ref_ligand_path=inputs.get("ref_ligand"),
        output_dir=output_dir,
    )
    task_yml, model_yml = _dump_cfg_pair(cfg, job_dir, settings_)
    return sample_argv(
        task_config_path=task_yml, model_config_path=model_yml,
        output_dir=output_dir, settings=settings_, batch_size=req.batch_size,
    )


def _confidence_build(
    req: ConfidenceRequest,
    inputs: dict[str, Path],
    job_dir: Path,
    settings_: PocketXMolSettings,
) -> list[str]:
    # CLI mode: user must have their source job's output/ dir locally.
    # We accept `--source-exp-dir` pointing to the timestamped subdir.
    src_exp = inputs.get("source_exp_dir")
    if src_exp is None or not src_exp.is_dir():
        raise SystemExit(
            "ERROR: --source-exp-dir must point to the timestamped experiment "
            "directory produced by a prior sample_use.py run."
        )
    yaml_path = confidence_yaml_path(req.variant, settings_)
    return confidence_argv(
        req=req, source_output_dir=src_exp,
        confidence_yaml_path=yaml_path, settings=settings_,
    )


# ---------------------------------------------------------------------------
# Endpoint registry.
# ---------------------------------------------------------------------------
endpoints = {
    "dock": CLIEndpoint(
        name="dock",
        help="Molecular docking (small-molecule via ligand/SMILES, "
             "or peptide via ligand PDB / pep_sequence).",
        request_model=DockRequest,
        build_argv=_dock_build,
        inputs={
            "protein": ("Input protein PDB (.pdb)", True),
            "ligand": (
                "Input ligand (.sdf for small-mol, .pdb for peptide) — "
                "alternative to smiles / pep_sequence in --params-json.",
                False,
            ),
            "ref_ligand": (
                "Reference ligand for pocket extraction (optional).",
                False,
            ),
        },
    ),
    "sbdd": CLIEndpoint(
        name="sbdd",
        help="De novo structure-based drug design (requires pocket_coord "
             "in --params-json).",
        request_model=SbddRequest,
        build_argv=_sbdd_build,
        inputs={
            "protein": ("Input protein PDB (.pdb)", True),
        },
    ),
    "linking": CLIEndpoint(
        name="linking",
        help="Fragment linking / growing / PROTAC (specify fragments "
             "list-of-lists in --params-json).",
        request_model=LinkingRequest,
        build_argv=_linking_build,
        inputs={
            "protein": ("Input protein PDB (.pdb)", True),
            "input_ligand": ("Input SDF containing the fragment(s)", True),
        },
    ),
    "optimize": CLIEndpoint(
        name="optimize",
        help="Molecular optimization (local refinement of an input ligand).",
        request_model=OptimizeRequest,
        build_argv=_optimize_build,
        inputs={
            "protein": ("Input protein PDB (.pdb)", True),
            "input_ligand": ("Input SDF to optimize", True),
        },
    ),
    "pepdesign": CLIEndpoint(
        name="pepdesign",
        help="Peptide design (mode = denovo_linear / denovo_cyclic / "
             "inverse_fold / sc_pack).",
        request_model=PepDesignRequest,
        build_argv=_pepdesign_build,
        inputs={
            "protein": ("Input protein PDB (.pdb)", True),
            "input_peptide": (
                "Input peptide PDB — required for inverse_fold / sc_pack.",
                False,
            ),
            "ref_ligand": (
                "Reference ligand for pocket extraction (optional).",
                False,
            ),
        },
    ),
    "confidence": CLIEndpoint(
        name="confidence",
        help="Tuned-ranker confidence scoring on a previously generated "
             "experiment directory.",
        request_model=ConfidenceRequest,
        build_argv=_confidence_build,
        inputs={
            "source_exp_dir": (
                "Timestamped experiment directory produced by a prior "
                "generation run (dock / sbdd / …).",
                True,
            ),
        },
    ),
}


create_cli(adapter, settings, endpoints, version="0.0.1")
