"""Argv builder unit tests for qligfep-server."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from server.models import (
    AnalyzeFepRequest, AnalyzeLieRequest, CogRequest, LigprepRequest,
    ProtprepRequest, RunFepRequest, SetupLieRequest, SetupLigfepRequest,
    SetupResfepRequest,
)
from server.settings import QligfepSettings
from server.tools import (
    analyze_fep_argv, analyze_lie_argv, cog_argv, ligprep_argv,
    protprep_argv, run_fep_argv, setup_lie_argv, setup_ligfep_argv,
    setup_resfep_argv,
)


class _Off(QligfepSettings):
    model_config = SettingsConfigDict(env_prefix="QLIGFEP_TEST_", env_file=None, extra="ignore")


@pytest.fixture
def settings(tmp_path):
    return _Off(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "root",
        upstream_dir=tmp_path / "upstream" / "qligfep",
        q_bin_dir=tmp_path / "Q6bin",
        python=tmp_path / "python",
    )


@pytest.fixture
def job_dir(tmp_path):
    d = tmp_path / "job"
    (d / "work").mkdir(parents=True)
    (d / "output").mkdir()
    return d


def test_ligprep_argv_basic(settings, job_dir):
    req = LigprepRequest(ligand_name="17", forcefield="openff-2.1.0")
    lig = job_dir / "inputs" / "lig.mol2"
    lig.parent.mkdir()
    lig.write_text("x")
    argv = ligprep_argv(req, lig, job_dir, settings)
    assert argv[0] == str(settings.python)
    assert "server.ligprep_cli" in argv
    assert "--ligand" in argv and str(lig) in argv
    assert "--ligand-name" in argv and "17" in argv
    assert "--forcefield" in argv and "openff-2.1.0" in argv
    assert "--work-dir" in argv and str(job_dir / "work") in argv
    assert "--output-dir" in argv and str(job_dir / "output") in argv
    assert "--net-charge" not in argv


def test_ligprep_argv_net_charge(settings, job_dir):
    req = LigprepRequest(ligand_name="17", net_charge=-1)
    argv = ligprep_argv(req, job_dir / "in.mol2", job_dir, settings)
    assert "--net-charge" in argv and "-1" in argv


def test_protprep_argv(settings, job_dir):
    req = ProtprepRequest(sphere_center="0.5:1.0:2.0", sphere_radius=22.0, forcefield="OPLSAAM")
    pdb = job_dir / "prot.pdb"
    pdb.write_text("ATOM")
    argv = protprep_argv(req, pdb, job_dir, settings)
    assert "server.protprep_cli" in argv
    assert "--sphere-radius" in argv and "22.0" in argv
    assert "--sphere-center" in argv and "0.5:1.0:2.0" in argv
    assert "--forcefield" in argv and "OPLSAAM" in argv


def test_cog_argv_all(settings, job_dir):
    req = CogRequest(mode="all")
    pdb = job_dir / "p.pdb"
    pdb.write_text("ATOM")
    argv = cog_argv(req, pdb, job_dir, settings)
    assert "--mode" in argv and "all" in argv
    assert "--atom-range" not in argv


def test_cog_argv_range(settings, job_dir):
    req = CogRequest(mode="atomrange", atom_range="1:25")
    argv = cog_argv(req, job_dir / "p.pdb", job_dir, settings)
    assert "--atom-range" in argv and "1:25" in argv


def test_setup_ligfep_argv(settings, job_dir):
    req = SetupLigfepRequest(lig1_name="17", lig2_name="18", windows=51)
    lig = job_dir / "lig_dir"; lig.mkdir()
    prot = job_dir / "prot_dir"; prot.mkdir()
    argv = setup_ligfep_argv(req, lig, prot, job_dir, settings)
    assert "server.setup_ligfep_cli" in argv
    assert "--lig1-name" in argv and "17" in argv
    assert "--lig2-name" in argv and "18" in argv
    assert "--windows" in argv and "51" in argv


def test_setup_resfep_argv(settings, job_dir):
    req = SetupResfepRequest(mutation="A24V", mutchain="A", dual=True)
    argv = setup_resfep_argv(req, job_dir / "prot", job_dir, settings)
    assert "server.setup_resfep_cli" in argv
    assert "--mutation" in argv and "A24V" in argv
    assert "--mutchain" in argv and "A" in argv
    assert "--dual" in argv  # flag


def test_setup_lie_argv(settings, job_dir):
    req = SetupLieRequest(ligand_name="17", radius=25.0)
    argv = setup_lie_argv(req, job_dir / "lig", job_dir / "prot", job_dir, settings)
    assert "server.setup_lie_cli" in argv
    assert "--ligand-name" in argv and "17" in argv
    assert "--radius" in argv and "25.0" in argv


def test_run_fep_argv_mpi(settings, job_dir):
    req = RunFepRequest(window_idx=5, leg="protein", device="mpi", nprocs=4)
    setup = job_dir / "setup_dir"; setup.mkdir()
    argv = run_fep_argv(req, setup, job_dir, settings)
    assert "server.run_fep_cli" in argv
    assert "--window-idx" in argv and "5" in argv
    assert "--leg" in argv and "protein" in argv
    assert "--device" in argv and "mpi" in argv
    assert "--nprocs" in argv and "4" in argv


def test_run_fep_argv_gpu(settings, job_dir):
    req = RunFepRequest(window_idx=0, leg="water", device="gpu")
    argv = run_fep_argv(req, job_dir / "s", job_dir, settings)
    assert "--device" in argv and "gpu" in argv


def test_analyze_fep_argv(settings, job_dir):
    req = AnalyzeFepRequest(temperature=310.0, use_pdb=True)
    argv = analyze_fep_argv(req, job_dir / "run", job_dir, settings)
    assert "server.analyze_fep_cli" in argv
    assert "--temperature" in argv and "310.0" in argv
    assert "--use-pdb" in argv


def test_analyze_lie_argv(settings, job_dir):
    req = AnalyzeLieRequest(radius=22.0)
    argv = analyze_lie_argv(req, job_dir / "run", job_dir, settings)
    assert "server.analyze_lie_cli" in argv
    assert "--radius" in argv and "22.0" in argv
