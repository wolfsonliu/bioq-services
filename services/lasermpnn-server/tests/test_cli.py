"""CLI batch-mode tests for lasermpnn-server.

Tests endpoint registration, build_argv callbacks, and end-to-end create_cli.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioq_service.cli import CLIEndpoint, create_cli

from server.adapter import LASErMPNNAdapter
from server.models import DesignLigandMPNNRequest, DesignRequest
from server.settings import LASErMPNNSettings
from server.tools import design_argv, design_ligandmpnn_argv


class _Off(LASErMPNNSettings):
    model_config = SettingsConfigDict(env_prefix="LASERMPNN_TEST_", env_file=None, extra="ignore")


def _copy_input(inputs, job_dir):
    import shutil
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    dest = input_dir / "input.pdb"
    shutil.copy2(inputs["pdb"], dest)
    return dest


def _build_design(req, inputs, job_dir, settings):
    return design_argv(req, input_pdb=_copy_input(inputs, job_dir), job_dir=job_dir, settings=settings)


def _build_design_ligandmpnn(req, inputs, job_dir, settings):
    return design_ligandmpnn_argv(
        req, input_pdb=_copy_input(inputs, job_dir), job_dir=job_dir, settings=settings,
    )


def _make_endpoints():
    return {
        "design": CLIEndpoint(
            name="design",
            help="LASErMPNN ligand-conditioned batch design",
            request_model=DesignRequest,
            build_argv=_build_design,
            inputs={"pdb": ("Input PDB with a protonated ligand", True)},
        ),
        "design_ligandmpnn": CLIEndpoint(
            name="design_ligandmpnn",
            help="Retrained LigandMPNN variant batch design",
            request_model=DesignLigandMPNNRequest,
            build_argv=_build_design_ligandmpnn,
            inputs={"pdb": ("Input PDB with a protonated ligand", True)},
        ),
    }


ENDPOINTS = _make_endpoints()


# ---- registration ----

def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"design", "design_ligandmpnn"}


def test_design_endpoint_fields():
    ep = ENDPOINTS["design"]
    assert ep.request_model is DesignRequest
    assert ep.inputs["pdb"][1] is True


def test_design_ligandmpnn_endpoint_fields():
    ep = ENDPOINTS["design_ligandmpnn"]
    assert ep.request_model is DesignLigandMPNNRequest


# ---- build_argv callbacks ----

def test_design_build_argv(tmp_path):
    s = _Off(weights_dir=tmp_path / "w", device="cpu")
    job_dir = tmp_path / "job"
    pdb = tmp_path / "in.pdb"
    pdb.write_text("ATOM\n")
    argv = _build_design(DesignRequest(designs_per_input=3), {"pdb": pdb}, job_dir, s)
    assert argv[2] == "LASErMPNN.run_batch_inference"
    assert "3" in argv
    assert (job_dir / "input" / "input.pdb").exists()


def test_design_ligandmpnn_build_argv(tmp_path):
    s = _Off(weights_dir=tmp_path / "w")
    job_dir = tmp_path / "job"
    pdb = tmp_path / "in.pdb"
    pdb.write_text("ATOM\n")
    argv = _build_design_ligandmpnn(DesignLigandMPNNRequest(), {"pdb": pdb}, job_dir, s)
    assert argv[2] == "LASErMPNN.run_batch_inference_ligandmpnn"


# ---- end-to-end create_cli ----

def test_cli_design_success(tmp_path):
    s = _Off(root=tmp_path / "root", weights_dir=tmp_path / "w")
    s.jobs_base_dir = tmp_path / "jobs"
    adapter = LASErMPNNAdapter(settings=s)
    pdb = tmp_path / "in.pdb"
    pdb.write_text("ATOM\n")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "design",
        "--pdb", str(pdb),
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc.value.code == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(root=tmp_path / "root", weights_dir=tmp_path / "w")
    adapter = LASErMPNNAdapter(settings=s)
    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
