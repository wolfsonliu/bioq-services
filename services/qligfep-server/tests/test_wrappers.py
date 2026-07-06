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


def test_ligprep_wrapper_calls_openff2q(tmp_path):
    work, out = _work_out(tmp_path)
    lig = tmp_path / "in" / "17.mol2"; lig.parent.mkdir()
    lig.write_text("@<TRIPOS>MOLECULE\n17\n1 0 0 0 0\n")

    from server import ligprep_cli

    calls: list[list[str]] = []

    def fake_run(argv, cwd=None, *args, **kw):
        calls.append(list(argv))
        for ext in ("lib", "prm", "pdb"):
            (Path(cwd) / f"17.{ext}").write_text(f"{ext} content\n")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    with patch.object(ligprep_cli, "_generate_offxml", return_value=None):
        with patch.object(subprocess, "run", side_effect=fake_run):
            rc = ligprep_cli.run(
                ligand=lig, ligand_name="17", forcefield="openff-2.1.0",
                net_charge=None, work_dir=work, output_dir=out,
            )
    assert rc == 0
    for ext in ("lib", "prm", "pdb"):
        assert (out / f"17.{ext}").exists()
    assert any("openff2Q.py" in " ".join(c) for c in calls)


def test_setup_ligfep_stages_and_collects(tmp_path):
    work, out = _work_out(tmp_path)
    ligprep = tmp_path / "lp"; ligprep.mkdir()
    protprep = tmp_path / "pp"; protprep.mkdir()
    for n in ("17.lib", "17.prm", "17.pdb", "18.lib", "18.prm", "18.pdb"):
        (ligprep / n).write_text(n)
    for n in ("protein.pdb", "water.pdb"):
        (protprep / n).write_text(n)

    from server import setup_ligfep_cli

    def fake_run(argv, cwd=None, *args, **kw):
        base = Path(cwd) / "17-18"
        for leg in ("1.protein", "2.water"):
            fep1 = base / leg / "FEP1"
            fep1.mkdir(parents=True)
            (fep1 / "md_0500_0500.inp").write_text("")
            (base / leg / "FEP_submit.sh").write_text("#!/bin/bash\n")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    with patch.object(subprocess, "run", side_effect=fake_run):
        rc = setup_ligfep_cli.run(
            lig1_name="17", lig2_name="18",
            ligprep_dir=ligprep, protprep_dir=protprep,
            forcefield="OPLSAAM", system="protein", start="0.5",
            temperature=298.15, replicates=10, windows=51,
            sampling="linear", timestep="2fs", cluster="LOCAL",
            to_clean=True,
            work_dir=work, output_dir=out,
        )
    assert rc == 0
    assert (out / "1.protein" / "FEP_submit.sh").exists()
    assert (out / "2.water" / "FEP_submit.sh").exists()
    assert (out / "setup.json").exists()
    meta = json.loads((out / "setup.json").read_text())
    assert meta["windows"] == 51 and meta["lig1"] == "17"


def test_setup_resfep_stages_and_collects(tmp_path):
    work, out = _work_out(tmp_path)
    protprep = tmp_path / "pp"; protprep.mkdir()
    for n in ("protein.pdb", "water.pdb"):
        (protprep / n).write_text(n)

    from server import setup_resfep_cli

    def fake_run(argv, cwd=None, *args, **kw):
        base = Path(cwd) / "A24V"
        for leg in ("1.protein", "2.water"):
            fep = base / leg / "FEP1"; fep.mkdir(parents=True)
            (fep / "md_0500_0500.inp").write_text("")
            (base / leg / "FEP_submit.sh").write_text("#!/bin/bash\n")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    with patch.object(subprocess, "run", side_effect=fake_run):
        rc = setup_resfep_cli.run(
            mutation="A24V", mutchain="A", protprep_dir=protprep,
            system="protein", dual=True, shell_rest=25.0,
            tripeptide=False, cofactors=None,
            forcefield="OPLSAAM", windows=51, sampling="linear",
            timestep="2fs", temperature=298.15, replicates=10,
            start="0.5", cluster="LOCAL",
            work_dir=work, output_dir=out,
        )
    assert rc == 0
    assert (out / "1.protein" / "FEP_submit.sh").exists()
    assert (out / "setup.json").exists()
    meta = json.loads((out / "setup.json").read_text())
    assert meta["mutation"] == "A24V" and meta["dual"] is True
