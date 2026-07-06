"""CLI batch-mode build helpers, split out of ``__main__.py`` so tests can
import them without triggering ``create_cli(...)`` at module import time
(which would try to parse ``pytest``'s argv).
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service.cli import CLIEndpoint

from .models import DockRequest
from .settings import DiffdockSettings
from .tools import dock_argv


def dock_build(
    req: DockRequest,
    inputs: dict[str, Path],
    job_dir: Path,
    settings: DiffdockSettings,
) -> list[str]:
    """CLI-level: assemble ``dock_argv`` from the (already resolved) file
    inputs + req text params.

    Mutex validation:
    - Protein: exactly one of {--protein file, req.protein_uri (unusual for
      CLI mode but allowed), req.protein_sequence}
    - Ligand: exactly one of {--ligand file, req.ligand_uri,
      req.ligand_description}
    """
    protein_file = inputs.get("protein")
    ligand_file = inputs.get("ligand")

    protein_flags = [
        protein_file is not None,
        req.protein_uri is not None,
        req.protein_sequence is not None,
    ]
    if sum(protein_flags) != 1:
        raise SystemExit(
            "ERROR: exactly one of --protein FILE, protein_uri, or "
            "protein_sequence (via --params-json) must be provided."
        )
    ligand_flags = [
        ligand_file is not None,
        req.ligand_uri is not None,
        req.ligand_description is not None,
    ]
    if sum(ligand_flags) != 1:
        raise SystemExit(
            "ERROR: exactly one of --ligand FILE, ligand_uri, or "
            "ligand_description (via --params-json) must be provided."
        )

    if ligand_file is not None:
        ligand_arg = str(ligand_file)
    elif req.ligand_description is not None:
        ligand_arg = req.ligand_description
    else:
        raise SystemExit(
            "ERROR: ligand_uri is only supported over HTTP; "
            "for CLI batch mode, pass --ligand FILE or ligand_description."
        )

    protein_path = None
    protein_sequence = None
    if protein_file is not None:
        protein_path = protein_file
    elif req.protein_sequence is not None:
        protein_sequence = req.protein_sequence
    else:
        raise SystemExit(
            "ERROR: protein_uri is only supported over HTTP; "
            "for CLI batch mode, pass --protein FILE or protein_sequence."
        )

    return dock_argv(
        protein_path=protein_path,
        protein_sequence=protein_sequence,
        ligand_arg=ligand_arg,
        out_dir=job_dir / "output",
        params=req,
        settings=settings,
    )


def build_endpoints() -> dict[str, CLIEndpoint]:
    """Return the CLIEndpoint mapping used by ``__main__.py``.

    Broken out so tests can construct the same registry without invoking
    ``create_cli`` (which parses ``sys.argv``).
    """
    return {
        "dock": CLIEndpoint(
            name="dock",
            help="Single-complex DiffDock-L blind docking",
            request_model=DockRequest,
            build_argv=dock_build,
            inputs={
                "protein": (
                    "Input protein structure (.pdb) — omit when using "
                    "protein_sequence text param",
                    False,
                ),
                "ligand": (
                    "Input ligand file (.sdf/.mol2) — omit when using "
                    "ligand_description SMILES",
                    False,
                ),
            },
        ),
    }
