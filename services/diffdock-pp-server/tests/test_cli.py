"""CLI batch-mode tests for diffdock-pp-server."""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioagent_service.cli import CLIEndpoint, create_cli

from server.adapter import DiffDockPPAdapter
from server.models import DockRequest
from server.settings import DiffDockPPSettings
from server.tools import dock_argv


class _Off(DiffDockPPSettings):
    model_config = SettingsConfigDict(
        env_prefix="DIFFDOCK_PP_TEST_",
        env_file=None,
        extra="ignore",
    )


def _dock_build(req, inputs, job_dir, settings):
    return dock_argv(
        req,
        job_dir=job_dir,
        receptor=inputs["receptor"],
        ligand=inputs["ligand"],
        settings=settings,
    )


ENDPOINTS = {
    "dock": CLIEndpoint(
        name="dock",
        help="Rigid protein-protein docking",
        request_model=DockRequest,
        build_argv=_dock_build,
        inputs={
            "receptor": ("Receptor protein (.pdb)", True),
            "ligand": ("Ligand protein (.pdb)", True),
        },
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"dock"}


def test_dock_endpoint_fields():
    ep = ENDPOINTS["dock"]
    assert ep.request_model is DockRequest
    assert ep.inputs["receptor"][1] is True
    assert ep.inputs["ligand"][1] is True


# ---- build_argv ----


def test_dock_build_argv(tmp_path):
    s = _Off(
        python="/bin/true",
        inference_script="/opt/inference.py",
        config_yaml=tmp_path / "c.yaml",
        weights_dir=tmp_path / "w",
    )
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    rec = tmp_path / "r.pdb"
    rec.write_text("ATOM")
    lig = tmp_path / "l.pdb"
    lig.write_text("ATOM")

    argv = _dock_build(
        DockRequest(num_samples=5, top_k=2),
        {"receptor": rec, "ligand": lig},
        job_dir,
        s,
    )
    assert argv[0] == "/bin/true"
    assert "/opt/inference.py" in argv
    assert "--num_samples" in argv and "5" in argv
    assert "--top_k" in argv and "2" in argv
    assert "--use_confidence_model" in argv and "true" in argv  # default


# ---- End-to-end create_cli ----


def test_cli_dock_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = DiffDockPPAdapter(settings=s)

    rec = tmp_path / "r.pdb"
    rec.write_text("ATOM")
    lig = tmp_path / "l.pdb"
    lig.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "dock",
        "--receptor", str(rec),
        "--ligand", str(lig),
        "--output-dir", str(output_dir),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_dock_json_output(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = DiffDockPPAdapter(settings=s)

    rec = tmp_path / "r.pdb"
    rec.write_text("ATOM")
    lig = tmp_path / "l.pdb"
    lig.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "dock",
        "--receptor", str(rec),
        "--ligand", str(lig),
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
    adapter = DiffDockPPAdapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
