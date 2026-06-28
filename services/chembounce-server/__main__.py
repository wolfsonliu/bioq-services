"""CLI batch-mode entry point for chembounce-server.

Same Docker image runs the HTTP service (via uvicorn) and this CLI batch
mode (via `python -m server scaffold_hop ...`).

Usage::

    python -m server scaffold_hop \\
        --input-smiles "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl" \\
        --output-dir /scratch/results/ \\
        --params-json '{"frag_max_n": 100, "tanimoto_threshold": 0.5, "database": "250mw"}'

ChemBounce takes the SMILES as a parameter (not a file), so there are
no `inputs={...}` files declared — everything goes through
`--params-json` or per-field flags.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import ChemBounceAdapter
from .models import ScaffoldHopRequest
from .settings import ChemBounceSettings
from .tools import scaffold_hop_argv

settings = ChemBounceSettings()
adapter = ChemBounceAdapter(settings=settings)


def _scaffold_hop_build(
    req: ScaffoldHopRequest,
    _inputs: dict[str, Path],
    job_dir: Path,
    settings: ChemBounceSettings,
) -> list[str]:
    # Persist SMILES alongside (so the job directory is self-documenting).
    in_dir = job_dir / "input"
    in_dir.mkdir(parents=True, exist_ok=True)
    (in_dir / "input_smiles.txt").write_text(req.input_smiles + "\n")
    return scaffold_hop_argv(req, job_dir=job_dir, settings=settings)


endpoints = {
    "scaffold_hop": CLIEndpoint(
        name="scaffold_hop",
        help="Run ChemBounce scaffold hopping from a SMILES + thresholds",
        request_model=ScaffoldHopRequest,
        build_argv=_scaffold_hop_build,
        inputs={},  # No file inputs — SMILES is a parameter.
    ),
}


create_cli(adapter, settings, endpoints, version="0.0.1")
