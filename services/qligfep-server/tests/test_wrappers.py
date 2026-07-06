"""Wrapper CLI unit tests (subprocess mocked)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


def _work_out(tmp_path):
    w = tmp_path / "work"; w.mkdir()
    o = tmp_path / "output"; o.mkdir()
    return w, o


def test_cog_wrapper_writes_result_json(tmp_path):
    work, out = _work_out(tmp_path)
    pdb = tmp_path / "p.pdb"
    pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000\n")

    from server import cog_cli

    fake_stdout = "COG all: 0.535 26.772 8.819\n"
    with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess(
        args=[], returncode=0, stdout=fake_stdout, stderr="")):
        rc = cog_cli.run(
            pdb=pdb, mode="all", atom_range=None,
            work_dir=work, output_dir=out,
        )
    assert rc == 0
    result = json.loads((out / "result.json").read_text())
    assert result == {"cx": 0.535, "cy": 26.772, "cz": 8.819, "mode": "all"}


def test_protprep_wrapper_stages_and_collects(tmp_path):
    work, out = _work_out(tmp_path)
    protein = tmp_path / "in" / "prot.pdb"; protein.parent.mkdir()
    protein.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000\n")

    from server import protprep_cli

    def fake_run(argv, cwd, *args, **kw):
        # simulate upstream protprep.py creating protein.pdb + water.pdb + protprep.log in cwd
        (Path(cwd) / "protein.pdb").write_text("ATOM protein_out\n")
        (Path(cwd) / "water.pdb").write_text("HETATM water_out\n")
        (Path(cwd) / "protprep.log").write_text("done\n")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    with patch.object(subprocess, "run", side_effect=fake_run):
        rc = protprep_cli.run(
            protein_pdb=protein,
            sphere_radius=22.0, sphere_center="0.5:1.0:2.0",
            forcefield="OPLSAAM", mutchain=None,
            nowater=False, noclean=False, preplocation="LOCAL",
            work_dir=work, output_dir=out,
        )
    assert rc == 0
    assert (out / "protein.pdb").exists()
    assert (out / "water.pdb").exists()
    assert (out / "system.json").exists()
    meta = json.loads((out / "system.json").read_text())
    assert meta["sphere_center"] == "0.5:1.0:2.0"
    assert meta["forcefield"] == "OPLSAAM"
