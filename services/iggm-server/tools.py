"""Argv assembly for iggm-server.

Two subprocess wrappers, both vendored under /opt/iggm/server/:

- run_design.py  — thin wrapper over upstream design.py (design /
  inverse_design / fr_design / affinity_maturation).  Reuses design.predict;
  injects seed and pre-validates the FASTA.  Outputs land in <job>/output/.
- epitope.py     — reuses IgGM.protein.cal_ppi to dump epitope.json.

Checkpoints are resolved by upstream from ./checkpoints/<name>.pth relative to
the subprocess cwd (/opt/iggm), which is symlinked to the NAS weights_dir.
"""

from __future__ import annotations

from pathlib import Path

from .models import AffinityMaturationRequest, DesignRequest
from .settings import IgGMSettings


def _output_dir(job_dir: Path) -> Path:
    out = job_dir / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


def design_argv(
    req: DesignRequest | AffinityMaturationRequest,
    *,
    job_dir: Path,
    fasta_path: Path,
    antigen_path: Path,
    settings: IgGMSettings,
    run_task: str,
    fasta_origin_path: Path | None = None,
) -> list[str]:
    """Compose the run_design.py argv for any of the four design tasks."""
    out = _output_dir(job_dir)
    argv = [
        settings.python,
        settings.design_script,
        "--fasta", str(fasta_path),
        "--antigen", str(antigen_path),
        "--output", str(out),
        "--run_task", run_task,
        "--steps", str(req.steps),
        "--num_samples", str(req.num_samples),
        "--chunk_size", str(req.chunk_size),
        "--temperature", str(req.temperature),
        "--max_antigen_size", str(req.max_antigen_size),
    ]
    if req.epitope:
        argv.append("--epitope")
        argv.extend(str(e) for e in req.epitope)
    if fasta_origin_path is not None:
        argv += ["--fasta_origin", str(fasta_origin_path)]
    if req.seed is not None:
        argv += ["--seed", str(req.seed)]
    return argv


def epitope_argv(
    *,
    job_dir: Path,
    fasta_path: Path,
    antigen_path: Path,
    settings: IgGMSettings,
) -> list[str]:
    """Compose the epitope.py argv (interface residue calculation)."""
    out = _output_dir(job_dir)
    return [
        settings.python,
        settings.epitope_script,
        "--fasta", str(fasta_path),
        "--antigen", str(antigen_path),
        "--output", str(out),
    ]


__all__ = ["design_argv", "epitope_argv"]
