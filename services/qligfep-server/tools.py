"""Argv builders for qligfep-server subprocess wrappers.

Each function returns the argv list to be run by SubprocessRunner (HTTP mode)
or the CLI batch entry (__main__.py).  All wrappers are invoked as
``python -m server.<endpoint>_cli`` for consistency.
"""
from __future__ import annotations

from pathlib import Path

from .models import (
    AnalyzeFepRequest, AnalyzeLieRequest, CogRequest, LigprepRequest,
    ProtprepRequest, RunFepRequest, SetupLieRequest, SetupLigfepRequest,
    SetupResfepRequest,
)
from .settings import QligfepSettings


def _base(module: str, job_dir: Path, settings: QligfepSettings) -> list[str]:
    return [
        str(settings.python), "-m", module,
        "--work-dir", str(job_dir / "work"),
        "--output-dir", str(job_dir / "output"),
    ]


def ligprep_argv(req: LigprepRequest, ligand_path: Path,
                 job_dir: Path, settings: QligfepSettings) -> list[str]:
    argv = _base("server.ligprep_cli", job_dir, settings) + [
        "--ligand", str(ligand_path),
        "--ligand-name", req.ligand_name,
        "--forcefield", req.forcefield,
    ]
    if req.net_charge is not None:
        argv += ["--net-charge", str(req.net_charge)]
    return argv


def protprep_argv(req: ProtprepRequest, protein_pdb: Path,
                  job_dir: Path, settings: QligfepSettings) -> list[str]:
    argv = _base("server.protprep_cli", job_dir, settings) + [
        "--protein-pdb", str(protein_pdb),
        "--sphere-radius", str(req.sphere_radius),
        "--sphere-center", req.sphere_center,
        "--forcefield", req.forcefield,
        "--preplocation", req.preplocation,
    ]
    if req.mutchain:
        argv += ["--mutchain", req.mutchain]
    if req.nowater:
        argv += ["--nowater"]
    if req.noclean:
        argv += ["--noclean"]
    return argv


def cog_argv(req: CogRequest, pdb: Path,
             job_dir: Path, settings: QligfepSettings) -> list[str]:
    argv = _base("server.cog_cli", job_dir, settings) + [
        "--pdb", str(pdb),
        "--mode", req.mode,
    ]
    if req.atom_range:
        argv += ["--atom-range", req.atom_range]
    return argv


def _common_setup_flags(req) -> list[str]:
    flags = [
        "--forcefield", req.forcefield,
        "--system", req.system,
        "--start", req.start,
        "--temperature", str(req.temperature),
        "--replicates", str(req.replicates),
        "--windows", str(req.windows),
        "--sampling", req.sampling,
        "--timestep", req.timestep,
        "--cluster", req.cluster,
    ]
    return flags


def setup_ligfep_argv(req: SetupLigfepRequest, ligprep_dir: Path,
                      protprep_dir: Path, job_dir: Path,
                      settings: QligfepSettings) -> list[str]:
    argv = _base("server.setup_ligfep_cli", job_dir, settings) + [
        "--lig1-name", req.lig1_name,
        "--lig2-name", req.lig2_name,
        "--ligprep-dir", str(ligprep_dir),
        "--protprep-dir", str(protprep_dir),
    ] + _common_setup_flags(req)
    if req.to_clean:
        argv += ["--to-clean"]
    return argv


def setup_resfep_argv(req: SetupResfepRequest, protprep_dir: Path,
                      job_dir: Path, settings: QligfepSettings) -> list[str]:
    argv = _base("server.setup_resfep_cli", job_dir, settings) + [
        "--mutation", req.mutation,
        "--mutchain", req.mutchain,
        "--protprep-dir", str(protprep_dir),
        "--shell-rest", str(req.shell_rest),
    ] + _common_setup_flags(req)
    if req.dual:
        argv += ["--dual"]
    if req.tripeptide:
        argv += ["--tripeptide"]
    if req.cofactors:
        for c in req.cofactors:
            argv += ["--cofactor", c]
    return argv


def setup_lie_argv(req: SetupLieRequest, ligprep_dir: Path,
                   protprep_dir: Path, job_dir: Path,
                   settings: QligfepSettings) -> list[str]:
    argv = _base("server.setup_lie_cli", job_dir, settings) + [
        "--ligand-name", req.ligand_name,
        "--forcefield", req.forcefield,
        "--system", req.system,
        "--radius", str(req.radius),
        "--time-ns", str(req.time_ns),
        "--temperature", str(req.temperature),
        "--replicates", str(req.replicates),
        "--cluster", req.cluster,
        "--preplocation", req.preplocation,
        "--ligprep-dir", str(ligprep_dir),
        "--protprep-dir", str(protprep_dir),
    ]
    if req.cofactor:
        for c in req.cofactor:
            argv += ["--cofactor", c]
    return argv


def run_fep_argv(req: RunFepRequest, setup_dir: Path,
                 job_dir: Path, settings: QligfepSettings) -> list[str]:
    argv = _base("server.run_fep_cli", job_dir, settings) + [
        "--setup-dir", str(setup_dir),
        "--window-idx", str(req.window_idx),
        "--leg", req.leg,
        "--replicate-idx", str(req.replicate_idx),
        "--device", req.device,
        "--nprocs", str(req.nprocs),
        "--stage", req.stage,
    ]
    if req.keep_dcd:
        argv += ["--keep-dcd"]
    return argv


def analyze_fep_argv(req: AnalyzeFepRequest, run_dir: Path,
                     job_dir: Path, settings: QligfepSettings) -> list[str]:
    argv = _base("server.analyze_fep_cli", job_dir, settings) + [
        "--run-dir", str(run_dir),
        "--temperature", str(req.temperature),
        "--start", req.start,
        "--end-state-catastrophe", str(req.end_state_catastrophe),
    ]
    if req.use_pdb:
        argv += ["--use-pdb"]
    return argv


def analyze_lie_argv(req: AnalyzeLieRequest, run_dir: Path,
                     job_dir: Path, settings: QligfepSettings) -> list[str]:
    argv = _base("server.analyze_lie_cli", job_dir, settings) + [
        "--run-dir", str(run_dir),
        "--radius", str(req.radius),
    ]
    if req.cofactor:
        for c in req.cofactor:
            argv += ["--cofactor", c]
    return argv


__all__ = [
    "ligprep_argv", "protprep_argv", "cog_argv",
    "setup_ligfep_argv", "setup_resfep_argv", "setup_lie_argv",
    "run_fep_argv", "analyze_fep_argv", "analyze_lie_argv",
]
