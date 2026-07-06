"""Real OpenFF ligprep smoke test — run manually.

Requires openff-toolkit and a small ligand fixture; skipped when either is missing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data" / "ligand" / "17.pdb"

openff = pytest.importorskip("openff.toolkit")


@pytest.mark.slow
def test_ligprep_openff_real(tmp_path):
    if not DATA.exists():
        pytest.skip(f"fixture missing: {DATA}")
    from server import ligprep_cli
    rc = ligprep_cli.run(
        ligand=DATA, ligand_name="17", forcefield="openff-2.1.0",
        net_charge=None,
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
    )
    assert rc == 0
    for ext in ("lib", "prm", "pdb"):
        p = tmp_path / "out" / f"17.{ext}"
        assert p.exists() and p.stat().st_size > 0
