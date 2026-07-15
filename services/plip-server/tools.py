"""Subprocess argv builder for plip-server.

`profile_argv` assembles a single `python -m plip.plipcmd` invocation. PLIP is
launched as a module (its `plipcmd.py` has a clean argparse + `__main__` guard),
with `PYTHONPATH` pointing at the vendored upstream root so `import plip` works.
The upstream source is never modified — this is a pure argv wrapper.
"""

from __future__ import annotations

from pathlib import Path

from .models import ProfileRequest
from .settings import PlipSettings

# report_format → PLIP CLI flag.
_FORMAT_FLAG = {"xml": "-x", "txt": "-t"}


def profile_argv(
    req: ProfileRequest,
    *,
    job_dir: Path,
    input_pdb: Path,
    settings: PlipSettings,
) -> list[str]:
    """Build `python -m plip.plipcmd -f <pdb> -o output/ ...` for one complex."""
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    argv: list[str] = [
        settings.python, "-m", "plip.plipcmd",
        "-f", str(input_pdb),
        "-o", str(out_dir),
        "--name", req.name,
        "--maxthreads", str(req.maxthreads or settings.threads),
        "--model", str(req.model),
    ]

    # Report formats (at least one guaranteed non-empty by the model validator).
    for fmt in req.report_formats:
        argv.append(_FORMAT_FLAG[fmt])

    # Visualization.
    if req.pymol_session:
        argv.append("-y")
    if req.render_images:
        argv.append("-p")

    # Mutually-exclusive detection modes.
    if req.mode == "peptide":
        argv.append("--peptides")
        argv.extend(req.peptide_chains)
    elif req.mode == "intra":
        argv.extend(["--intra", req.intra_chain])
    elif req.mode == "dnareceptor":
        argv.append("--dnareceptor")

    # Passthrough boolean switches.
    if req.breakcomposite:
        argv.append("--breakcomposite")
    if req.altlocation:
        argv.append("--altlocation")
    if req.nofix:
        argv.append("--nofix")
    if req.keepmod:
        argv.append("--keepmod")
    if req.nohydro:
        argv.append("--nohydro")

    return argv


__all__ = ["profile_argv"]
