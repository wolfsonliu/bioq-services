"""CLI batch-mode tests for rfdiffusion2-server.

Tests endpoint registration, build_argv callbacks, and end-to-end create_cli.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from bioagent_service.cli import CLIEndpoint, create_cli

from server.adapter import RFdiffusion2Adapter
from server.models import ActiveSiteRequest, CustomRequest, SmallMoleculeBinderRequest
from server.settings import RFdiffusion2Settings
from server.tools import active_site_argv, custom_argv, small_molecule_binder_argv


@pytest.fixture
def settings(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    return RFdiffusion2Settings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "rfdiffusion2",
        models_dir=models,
        inference_script=tmp_path / "rfdiffusion2" / "rf_diffusion" / "run_inference.py",
        python=tmp_path / "rfdiffusion2" / ".venv" / "bin" / "python",
        pythonpath=tmp_path / "rfdiffusion2",
    )


def _active_site_build(req, inputs, job_dir, settings):
    return active_site_argv(req, inputs["input_pdb"], job_dir, settings)


def _sm_binder_build(req, inputs, job_dir, settings):
    return small_molecule_binder_argv(req, inputs["input_pdb"], job_dir, settings)


def _custom_build(req, inputs, job_dir, settings):
    return custom_argv(req, inputs.get("input_pdb"), job_dir, settings)


ENDPOINTS = {
    "active_site": CLIEndpoint(
        name="active_site",
        help="Active-site scaffolding around an atomic motif + ligand",
        request_model=ActiveSiteRequest,
        build_argv=_active_site_build,
        inputs={"input_pdb": ("Input PDB with motif + ligand", True)},
    ),
    "small_molecule_binder": CLIEndpoint(
        name="small_molecule_binder",
        help="Small-molecule binder design, optionally RASA-conditioned",
        request_model=SmallMoleculeBinderRequest,
        build_argv=_sm_binder_build,
        inputs={"input_pdb": ("Input PDB with small molecule", True)},
    ),
    "custom": CLIEndpoint(
        name="custom",
        help="Raw contig + freeform Hydra overrides",
        request_model=CustomRequest,
        build_argv=_custom_build,
        inputs={"input_pdb": ("Optional input PDB", False)},
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"active_site", "small_molecule_binder", "custom"}


def test_active_site_input_required():
    assert ENDPOINTS["active_site"].inputs["input_pdb"][1] is True
    assert ENDPOINTS["active_site"].request_model is ActiveSiteRequest


def test_sm_binder_input_required():
    assert ENDPOINTS["small_molecule_binder"].inputs["input_pdb"][1] is True
    assert ENDPOINTS["small_molecule_binder"].request_model is SmallMoleculeBinderRequest


def test_custom_input_optional():
    assert ENDPOINTS["custom"].inputs["input_pdb"][1] is False
    assert ENDPOINTS["custom"].request_model is CustomRequest


# ---- Build_argv callbacks ----


def test_active_site_build_argv(settings, tmp_path):
    pdb = tmp_path / "motif.pdb"
    pdb.write_text("ATOM")
    argv = _active_site_build(
        ActiveSiteRequest(
            contigs="46,A106-106,46",
            ligand="NAD",
            contig_atoms={"A106": "NE,CD,CZ"},
            num_designs=5,
        ),
        {"input_pdb": pdb},
        tmp_path / "j",
        settings,
    )
    joined = " ".join(argv)
    assert "--config-name=aa" in argv
    assert "inference.ligand='NAD'" in argv
    assert "inference.num_designs=5" in argv


def test_sm_binder_build_argv(settings, tmp_path):
    pdb = tmp_path / "target.pdb"
    pdb.write_text("ATOM")
    argv = _sm_binder_build(
        SmallMoleculeBinderRequest(
            contigs="150",
            ligand="PH2",
            rasa_active=True,
            rasa_target=0.0,
        ),
        {"input_pdb": pdb},
        tmp_path / "j",
        settings,
    )
    assert "--config-name=aa" in argv
    assert "inference.ligand=PH2" in argv
    assert "inference.conditions.relative_sasa_v2.active=True" in argv


def test_custom_build_argv(settings, tmp_path):
    argv = _custom_build(
        CustomRequest(contigs="150", config_name="aa"),
        {},
        tmp_path / "j",
        settings,
    )
    assert "--config-name=aa" in argv
    assert "contigmap.contigs=['150']" in argv


# ---- End-to-end create_cli ----


def test_cli_active_site_success(settings, tmp_path):
    adapter = RFdiffusion2Adapter(settings=settings)
    pdb = tmp_path / "motif.pdb"
    pdb.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    params = json.dumps({
        "contigs": "46,A106-106,46",
        "ligand": "NAD",
        "contig_atoms": {"A106": "NE,CD,CZ"},
    })

    with patch.object(sys, "argv", [
        "prog", "active_site",
        "--input-pdb", str(pdb),
        "--params-json", params,
        "--output-dir", str(output_dir),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, settings, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_no_subcommand_exits_2(settings):
    adapter = RFdiffusion2Adapter(settings=settings)
    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, settings, ENDPOINTS)


def test_cli_custom_json_output(settings, tmp_path, capsys):
    adapter = RFdiffusion2Adapter(settings=settings)
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "custom",
        "--contigs", "100",
        "--json", "--output-dir", str(output_dir),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, settings, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
