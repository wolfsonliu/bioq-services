"""CLI batch-mode tests for bindflow-server."""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioagent_service.cli import CLIEndpoint, create_cli

from server.adapter import BindFlowAdapter
from server.models import FepCalculateRequest, MmpbsaCalculateRequest
from server.settings import BindFlowSettings
from server.tools import calculate_argv


def _tiny_pdb() -> bytes:
    return b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"


def _tiny_sdf() -> bytes:
    return (
        b"L\n  test\n\n"
        b"  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        b"    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        b"M  END\n$$$$\n"
    )


class _Off(BindFlowSettings):
    model_config = SettingsConfigDict(
        env_prefix="BINDFLOW_TEST_",
        env_file=None,
        extra="ignore",
    )


def _fep_build(req, inputs, job_dir, settings):
    return calculate_argv(
        req,
        calculation_type="fep",
        job_dir=job_dir,
        protein_path=inputs["protein"],
        ligands_dir=inputs["ligands_dir"],
        cofactor_path=inputs.get("cofactor"),
        settings=settings,
    )


def _mmpbsa_build(req, inputs, job_dir, settings):
    return calculate_argv(
        req,
        calculation_type="mmpbsa",
        job_dir=job_dir,
        protein_path=inputs["protein"],
        ligands_dir=inputs["ligands_dir"],
        cofactor_path=inputs.get("cofactor"),
        settings=settings,
    )


INPUTS = {
    "protein": ("Protein PDB file", True),
    "ligands_dir": ("Directory containing ligand SDF/MOL files", True),
    "cofactor": ("Cofactor (optional)", False),
}

ENDPOINTS = {
    "fep": CLIEndpoint(
        name="fep", help="FEP",
        request_model=FepCalculateRequest,
        build_argv=_fep_build,
        inputs=INPUTS,
    ),
    "mmpbsa": CLIEndpoint(
        name="mmpbsa", help="MMPBSA",
        request_model=MmpbsaCalculateRequest,
        build_argv=_mmpbsa_build,
        inputs=INPUTS,
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"fep", "mmpbsa"}


def test_fep_endpoint_declares_ligands_dir():
    ep = ENDPOINTS["fep"]
    assert ep.request_model is FepCalculateRequest
    assert "ligands_dir" in ep.inputs
    assert ep.inputs["protein"][1] is True
    assert ep.inputs["cofactor"][1] is False


def test_mmpbsa_endpoint_declares_samples_field():
    ep = ENDPOINTS["mmpbsa"]
    assert "samples" in ep.request_model.model_fields


# ---- build_argv ----


def test_fep_build_argv(tmp_path):
    s = _Off(python="/bin/true")
    protein = tmp_path / "p.pdb"
    protein.write_bytes(_tiny_pdb())
    ligands_dir = tmp_path / "ligs"
    ligands_dir.mkdir()
    (ligands_dir / "a.sdf").write_bytes(_tiny_sdf())
    job_dir = tmp_path / "j"
    job_dir.mkdir()

    argv = _fep_build(
        FepCalculateRequest(replicas=1, num_jobs=1),
        {"protein": protein, "ligands_dir": ligands_dir},
        job_dir, s,
    )
    assert argv[0] == "/bin/true"
    assert "--calculation-type" in argv and "fep" in argv
    assert str(protein) in argv
    assert str(ligands_dir) in argv


def test_mmpbsa_build_argv_with_cofactor(tmp_path):
    s = _Off(python="/bin/true")
    protein = tmp_path / "p.pdb"
    protein.write_bytes(_tiny_pdb())
    ligands_dir = tmp_path / "ligs"
    ligands_dir.mkdir()
    (ligands_dir / "a.sdf").write_bytes(_tiny_sdf())
    cofactor = tmp_path / "c.sdf"
    cofactor.write_bytes(_tiny_sdf())
    job_dir = tmp_path / "j"
    job_dir.mkdir()

    argv = _mmpbsa_build(
        MmpbsaCalculateRequest(samples=8, replicas=1),
        {"protein": protein, "ligands_dir": ligands_dir, "cofactor": cofactor},
        job_dir, s,
    )
    assert "--samples" in argv and "8" in argv
    assert "--cofactor" in argv and str(cofactor) in argv


# ---- End-to-end create_cli ----


def _stage_cli_inputs(tmp_path):
    """Standard test fixture layout."""
    protein = tmp_path / "p.pdb"
    protein.write_bytes(_tiny_pdb())
    ligands_dir = tmp_path / "ligs"
    ligands_dir.mkdir()
    (ligands_dir / "a.sdf").write_bytes(_tiny_sdf())
    output_dir = tmp_path / "run" / "output"
    return protein, ligands_dir, output_dir


def test_cli_fep_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = BindFlowAdapter(settings=s)
    protein, ligands_dir, output_dir = _stage_cli_inputs(tmp_path)

    with patch.object(sys, "argv", [
        "prog", "fep",
        "--protein", str(protein),
        "--ligands-dir", str(ligands_dir),
        "--output-dir", str(output_dir),
        "--replicas", "1",
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_mmpbsa_json_output(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = BindFlowAdapter(settings=s)
    protein, ligands_dir, output_dir = _stage_cli_inputs(tmp_path)

    with patch.object(sys, "argv", [
        "prog", "mmpbsa",
        "--protein", str(protein),
        "--ligands-dir", str(ligands_dir),
        "--params-json", '{"samples": 12, "replicas": 1}',
        "--json", "--output-dir", str(output_dir),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert result["return_code"] == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = BindFlowAdapter(settings=s)
    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)


def test_cli_missing_input_exits_2(tmp_path):
    """--protein pointing to a non-existent path → argparse-level error."""
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = BindFlowAdapter(settings=s)
    with patch.object(sys, "argv", [
        "prog", "fep",
        "--protein", str(tmp_path / "no-such.pdb"),
        "--ligands-dir", str(tmp_path),
        "--output-dir", str(tmp_path / "out"),
    ]):
        with pytest.raises(SystemExit):
            create_cli(adapter, s, ENDPOINTS)
