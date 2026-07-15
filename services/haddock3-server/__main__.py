"""CLI batch-mode entry point for haddock3-server.

Same image, no uvicorn — for Slurm / sbatch one-shot runs::

    apptainer exec haddock3-server.sif \\
        python -m server dock-protein-protein \\
        --mol1 receptor.pdb --mol2 ligand.pdb --ambig air.tbl \\
        --output-dir /scratch/$SLURM_JOB_ID/ --sampling 200

    python -m server score --complex complex.pdb --output-dir out/ --full

    python -m server restrain-bodies --structure complex.pdb --output-dir out/

    python -m server dock --config workflow.cfg --output-dir out/
        # general runner: molecules referenced by absolute path inside the config
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import Haddock3Adapter
from .configs import build_protein_protein_cfg, write_cfg
from .models import (
    ActpassToAmbigRequest,
    DockRequest,
    ProteinProteinRequest,
    RestrainBodiesRequest,
    ScoreRequest,
)
from .settings import Haddock3Settings
from .tools import (
    actpass_to_ambig_argv,
    dock_argv,
    restrain_bodies_argv,
    score_argv,
)

settings = Haddock3Settings()
adapter = Haddock3Adapter(settings=settings)


def _dock_build(req: DockRequest, inputs, job_dir: Path, s: Haddock3Settings) -> list[str]:
    # HPC passthrough: the config already references molecules by absolute path.
    return dock_argv(
        req, config_path=inputs["config"], workdir=job_dir, job_dir=job_dir, settings=s,
    )


def _pp_build(
    req: ProteinProteinRequest, inputs, job_dir: Path, s: Haddock3Settings,
) -> list[str]:
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    run_dir = job_dir / "output" / "run"
    cfg = build_protein_protein_cfg(
        molecules=[str(inputs["mol1"]), str(inputs["mol2"])],
        run_dir=str(run_dir),
        ncores=req.ncores or s.default_ncores,
        sampling=req.sampling,
        do_flexref=req.do_flexref,
        do_emref=req.do_emref,
        clustering=req.clustering,
        top_models=req.top_models,
        ambig_fname=str(inputs["ambig"]) if inputs.get("ambig") else None,
        reference_fname=str(inputs["reference"]) if inputs.get("reference") else None,
    )
    cfg_path = write_cfg(cfg, input_dir / "workflow.cfg")
    return dock_argv(
        req, config_path=cfg_path, workdir=input_dir, job_dir=job_dir, settings=s,
    )


def _score_build(req: ScoreRequest, inputs, job_dir: Path, s: Haddock3Settings) -> list[str]:
    return score_argv(req, pdb=inputs["complex"], job_dir=job_dir, settings=s)


def _restrain_build(
    req: RestrainBodiesRequest, inputs, job_dir: Path, s: Haddock3Settings,
) -> list[str]:
    return restrain_bodies_argv(req, pdb=inputs["structure"], job_dir=job_dir, settings=s)


def _actpass_build(
    req: ActpassToAmbigRequest, inputs, job_dir: Path, s: Haddock3Settings,
) -> list[str]:
    return actpass_to_ambig_argv(
        req, actpass1=inputs["actpass1"], actpass2=inputs["actpass2"],
        job_dir=job_dir, settings=s,
    )


endpoints = {
    "dock": CLIEndpoint(
        name="dock",
        help="Run an arbitrary haddock3 workflow config (HPC passthrough)",
        request_model=DockRequest,
        build_argv=_dock_build,
        inputs={"config": ("HADDOCK3 workflow config (.cfg)", True)},
    ),
    "dock-protein-protein": CLIEndpoint(
        name="dock-protein-protein",
        help="Curated two-body protein-protein docking",
        request_model=ProteinProteinRequest,
        build_argv=_pp_build,
        inputs={
            "mol1": ("First molecule PDB", True),
            "mol2": ("Second molecule PDB", True),
            "ambig": ("Ambiguous interaction restraints (.tbl)", False),
            "reference": ("Reference complex PDB for CAPRI eval", False),
        },
    ),
    "score": CLIEndpoint(
        name="score",
        help="Standalone HADDOCK scoring of a complex",
        request_model=ScoreRequest,
        build_argv=_score_build,
        inputs={"complex": ("Complex PDB to score", True)},
    ),
    "restrain-bodies": CLIEndpoint(
        name="restrain-bodies",
        help="CNS-free body restraints from a multi-chain PDB",
        request_model=RestrainBodiesRequest,
        build_argv=_restrain_build,
        inputs={"structure": ("Multi-chain PDB", True)},
    ),
    "actpass-to-ambig": CLIEndpoint(
        name="actpass-to-ambig",
        help="CNS-free ambig restraints from two actpass files",
        request_model=ActpassToAmbigRequest,
        build_argv=_actpass_build,
        inputs={
            "actpass1": ("First active/passive residue file", True),
            "actpass2": ("Second active/passive residue file", True),
        },
    ),
}


create_cli(adapter, settings, endpoints, version="0.0.1")
