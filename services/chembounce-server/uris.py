"""Input URI resolution for chembounce-server.

ChemBounce takes SMILES as a string, not a file — so this resolver is slim: it
fetches an `input_smiles_uri` (for chained pipelines, e.g. a previous job's
output SMILES) via the shared `bioagent_service.uris.resolve_uri` and returns
the first non-empty line as the SMILES string. Most calls pass `input_smiles`
as a plain form field and never touch this.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import HTTPException

from bioagent_service.uris import resolve_uri

from .settings import ChemBounceSettings


def resolve_smiles_uri(uri: str, settings: ChemBounceSettings) -> str:
    """Fetch a SMILES from a URI and return its first non-empty line."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="chembounce_smiles_"))
    try:
        dest = resolve_uri(uri, tmp_dir / "smiles.txt", settings)
        return _read_first_smiles(dest)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _read_first_smiles(path: Path) -> str:
    """Read the first non-empty, non-comment line from a file as the SMILES."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    raise HTTPException(
        status_code=422,
        detail=f"No SMILES content in {path}; first non-empty line is required.",
    )
