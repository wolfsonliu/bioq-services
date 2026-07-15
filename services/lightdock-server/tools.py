"""argv builder for lightdock-server.

`dock_argv` composes the command line for our `docking.py` driver, which
orchestrates the multi-step LightDock protocol (setup -> run -> conformations
-> cluster -> rank -> top). The driver lives in this package so the venv can
import/run it without extra installation.
"""

from __future__ import annotations

from pathlib import Path

from .models import DockRequest
from .settings import LightdockSettings


def dock_argv(
    req: DockRequest,
    *,
    job_dir: Path,
    receptor_path: Path,
    ligand_path: Path,
    settings: LightdockSettings,
    restraints_path: Path | None = None,
) -> list[str]:
    """Compose argv for the docking driver.

    The driver stages receptor/ligand into `job_dir/work/`, chdir's there,
    runs the lgd_* pipeline, and copies ranked results into `job_dir/output/`.
    """
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = job_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    cores = req.cores if req.cores is not None else settings.default_cores
    scoring = req.scoring_function or settings.default_scoring

    argv: list[str] = [
        settings.python,
        settings.driver_script,
        "dock",
        "--receptor", str(receptor_path),
        "--ligand", str(ligand_path),
        "--workdir", str(work_dir),
        "--output-dir", str(out_dir),
        "--bin-dir", str(settings.bin_dir),
        "--swarms", str(req.swarms),
        "--glowworms", str(req.glowworms),
        "--steps", str(req.steps),
        "--scoring", scoring,
        "--cores", str(cores),
        "--top", str(req.top),
        "--swarm-seed", str(req.swarm_seed),
        "--gso-seed", str(req.gso_seed),
        "--cluster-cutoff", str(req.cluster_cutoff),
    ]
    if restraints_path is not None:
        argv += ["--restraints", str(restraints_path)]
    if req.use_anm:
        argv.append("--anm")
    if req.noxt:
        argv.append("--noxt")
    if req.noh:
        argv.append("--noh")
    if req.now:
        argv.append("--now")
    return argv
