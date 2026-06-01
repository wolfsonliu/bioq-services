"""CLI batch-mode tests for rfantibody-server.

Tests endpoint registration, build_argv callbacks, and end-to-end create_cli.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioagent_service.cli import CLIEndpoint, create_cli

from server.adapter import RFantibodyAdapter
from server.models import ProteinMPNNRequest, RF2Request, RFdiffusionRequest
from server.settings import RFantibodySettings
from server.tools import proteinmpnn_argv, rf2_argv, rfdiffusion_argv


class _Off(RFantibodySettings):
    model_config = SettingsConfigDict(
        env_prefix="RFANTIBODY_TEST_", env_file=None, extra="ignore", case_sensitive=False,
    )


def _rfdiffusion_build(req, inputs, job_dir, settings):
    return rfdiffusion_argv(req, inputs["target"], inputs["framework"], job_dir, settings)


def _proteinmpnn_build(req, inputs, job_dir, settings):
    return proteinmpnn_argv(req, inputs["input_quiver"], job_dir, settings)


def _rf2_build(req, inputs, job_dir, settings):
    return rf2_argv(req, inputs["input_quiver"], job_dir, settings)


ENDPOINTS = {
    "rfdiffusion": CLIEndpoint(
        name="rfdiffusion",
        help="RFdiffusion antibody-framework backbone design",
        request_model=RFdiffusionRequest,
        build_argv=_rfdiffusion_build,
        inputs={
            "target": ("Target antigen PDB file", True),
            "framework": ("Antibody framework PDB file", True),
        },
    ),
    "proteinmpnn": CLIEndpoint(
        name="proteinmpnn",
        help="ProteinMPNN CDR sequence design over RFdiffusion backbones",
        request_model=ProteinMPNNRequest,
        build_argv=_proteinmpnn_build,
        inputs={"input_quiver": ("Input Quiver file (from RFdiffusion)", True)},
    ),
    "rf2": CLIEndpoint(
        name="rf2",
        help="RF2 structure prediction + filtering over MPNN-designed sequences",
        request_model=RF2Request,
        build_argv=_rf2_build,
        inputs={"input_quiver": ("Input Quiver file (from ProteinMPNN)", True)},
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"rfdiffusion", "proteinmpnn", "rf2"}


def test_rfdiffusion_endpoint_fields():
    ep = ENDPOINTS["rfdiffusion"]
    assert ep.request_model is RFdiffusionRequest
    assert "target" in ep.inputs
    assert "framework" in ep.inputs
    assert ep.inputs["target"][1] is True
    assert ep.inputs["framework"][1] is True


def test_proteinmpnn_endpoint_fields():
    ep = ENDPOINTS["proteinmpnn"]
    assert ep.request_model is ProteinMPNNRequest
    assert "input_quiver" in ep.inputs
    assert ep.inputs["input_quiver"][1] is True


def test_rf2_endpoint_fields():
    ep = ENDPOINTS["rf2"]
    assert ep.request_model is RF2Request
    assert "input_quiver" in ep.inputs


# ---- Build_argv callbacks ----


def test_rfdiffusion_build_argv(tmp_path):
    s = _Off(root=tmp_path, weights_dir=tmp_path / "weights", scripts_dir=tmp_path / "scripts")
    job_dir = tmp_path / "j"
    (job_dir / "output").mkdir(parents=True)
    target = tmp_path / "target.pdb"
    framework = tmp_path / "framework.pdb"
    target.write_text("ATOM")
    framework.write_text("ATOM")

    argv = _rfdiffusion_build(
        RFdiffusionRequest(num_designs=5),
        {"target": target, "framework": framework},
        job_dir,
        s,
    )
    assert any("rfdiffusion_inference.py" in str(a) for a in argv)
    joined = " ".join(str(a) for a in argv)
    assert "inference.num_designs=5" in joined
    assert "antibody.target_pdb=" in joined


def test_proteinmpnn_build_argv(tmp_path):
    s = _Off(root=tmp_path, weights_dir=tmp_path / "weights", scripts_dir=tmp_path / "scripts")
    job_dir = tmp_path / "j"
    (job_dir / "output").mkdir(parents=True)
    quiver = tmp_path / "input.qv"
    quiver.write_text("dummy")

    argv = _proteinmpnn_build(
        ProteinMPNNRequest(),
        {"input_quiver": quiver},
        job_dir,
        s,
    )
    assert "-quiver" in argv
    assert "-outquiver" in argv


def test_rf2_build_argv(tmp_path):
    s = _Off(root=tmp_path, weights_dir=tmp_path / "weights", scripts_dir=tmp_path / "scripts")
    job_dir = tmp_path / "j"
    (job_dir / "output").mkdir(parents=True)
    quiver = tmp_path / "input.qv"
    quiver.write_text("dummy")

    argv = _rf2_build(
        RF2Request(),
        {"input_quiver": quiver},
        job_dir,
        s,
    )
    joined = " ".join(str(a) for a in argv)
    assert "output.quiver=" in joined
    assert "inference.cautious=False" in joined


# ---- End-to-end create_cli ----


def test_cli_rfdiffusion_success(tmp_path):
    s = _Off(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path,
        weights_dir=tmp_path / "weights",
        scripts_dir=tmp_path / "scripts",
    )
    adapter = RFantibodyAdapter(settings=s)

    target = tmp_path / "target.pdb"
    framework = tmp_path / "framework.pdb"
    target.write_text("ATOM")
    framework.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "rfdiffusion",
        "--target", str(target), "--framework", str(framework),
        "--output-dir", str(output_dir),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.2.0")
                assert exc_info.value.code == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path,
        weights_dir=tmp_path / "weights",
        scripts_dir=tmp_path / "scripts",
    )
    adapter = RFantibodyAdapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)


def test_cli_proteinmpnn_json_output(tmp_path, capsys):
    s = _Off(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path,
        weights_dir=tmp_path / "weights",
        scripts_dir=tmp_path / "scripts",
    )
    adapter = RFantibodyAdapter(settings=s)

    quiver = tmp_path / "input.qv"
    quiver.write_text("dummy")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "proteinmpnn",
        "--input-quiver", str(quiver),
        "--json", "--output-dir", str(output_dir),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.2.0")
                assert exc_info.value.code == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
