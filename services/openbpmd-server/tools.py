"""Argv assembly for openbpmd-server.

Wraps `services/openbpmd-server/inference.py`, which injects the
simtk->openmm shim + a configurable OpenMM platform, then dispatches into
the vendored `openbpmd.main()`.
"""

from __future__ import annotations

from pathlib import Path

from .models import ScoreRequest
from .settings import OpenBPMDSettings


def score_argv(
    req: ScoreRequest,
    *,
    job_dir: Path,
    structure: Path,
    parameters: Path,
    settings: OpenBPMDSettings,
) -> list[str]:
    """Compose the inference.py argv.

    Output goes to `<job_dir>/output/` per project convention; the adapter's
    `output_dir()` / `detect_outputs()` are wired against this.
    """
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        settings.python,
        settings.inference_script,
        "--structure", str(structure),
        "--parameters", str(parameters),
        "--output-dir", str(output_dir),
        "--lig-resname", req.lig_resname,
        "--nreps", str(req.nreps),
        "--hill-height", str(req.hill_height),
        "--platform", settings.platform,
    ]

    if req.system_format is not None:
        argv += ["--system-format", req.system_format]
    # Advanced/testing knobs — only forwarded when explicitly set.
    if req.sim_ns is not None:
        argv += ["--sim-ns", str(req.sim_ns)]
    if req.equil_steps is not None:
        argv += ["--equil-steps", str(req.equil_steps)]

    return argv


__all__ = ["score_argv"]
