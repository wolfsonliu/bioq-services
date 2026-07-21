"""CLI batch-mode tests for diffusion-hopping-server."""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioq_service.cli import CLIEndpoint, create_cli

from server.adapter import DiffusionHoppingAdapter
from server.models import GenerateRequest
from server.settings import DiffusionHoppingSettings
from server.tools import generate_argv


class _Off(DiffusionHoppingSettings):
    model_config = SettingsConfigDict(
        env_prefix="DIFFUSION_HOPPING_TEST_",
        env_file=None,
        extra="ignore",
    )


def _generate_build(req, inputs, job_dir, settings):
    return generate_argv(
        req,
        job_dir=job_dir,
        input_molecule=inputs["reference_ligand"],
        input_protein=inputs["protein"],
        settings=settings,
    )


ENDPOINTS = {
    "generate": CLIEndpoint(
        name="generate",
        help="Generate scaffold-hopping candidates",
        request_model=GenerateRequest,
        build_argv=_generate_build,
        inputs={
            "protein": ("Input protein (.pdb)", True),
            "reference_ligand": ("Input reference ligand", True),
        },
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"generate"}


def test_generate_endpoint_fields():
    ep = ENDPOINTS["generate"]
    assert ep.request_model is GenerateRequest
    assert ep.inputs["protein"][1] is True
    assert ep.inputs["reference_ligand"][1] is True


# ---- build_argv ----


def test_generate_build_argv(tmp_path):
    s = _Off(
        python="/bin/true",
        inference_script="/opt/inference.py",
        weights_dir=tmp_path / "weights",
    )
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    protein = tmp_path / "input.pdb"
    protein.write_text("ATOM")
    ligand = tmp_path / "ref.sdf"
    ligand.write_text("$$$$")

    argv = _generate_build(
        GenerateRequest(num_samples=5),
        {"protein": protein, "reference_ligand": ligand},
        job_dir,
        s,
    )
    assert argv[0] == "/bin/true"
    assert "/opt/inference.py" in argv
    assert "--num_samples" in argv and "5" in argv
    assert "--variant" in argv and "gvp_conditional" in argv  # default


# ---- End-to-end create_cli ----


def test_cli_generate_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = DiffusionHoppingAdapter(settings=s)

    protein = tmp_path / "p.pdb"
    protein.write_text("ATOM")
    ligand = tmp_path / "l.sdf"
    ligand.write_text("$$$$")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "generate",
        "--protein", str(protein),
        "--reference-ligand", str(ligand),
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_generate_json_output(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = DiffusionHoppingAdapter(settings=s)

    protein = tmp_path / "p.pdb"
    protein.write_text("ATOM")
    ligand = tmp_path / "l.sdf"
    ligand.write_text("$$$$")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "generate",
        "--protein", str(protein),
        "--reference-ligand", str(ligand),
        "--json", "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
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
    adapter = DiffusionHoppingAdapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
