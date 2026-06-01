"""CLI batch-mode tests for dockq-server.

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

from server.adapter import DockQAdapter
from server.models import ScoreBatchRequest, ScoreRequest
from server.settings import DockQSettings
from server.tools import batch_argv, score_argv


class _Off(DockQSettings):
    model_config = SettingsConfigDict(env_prefix="DOCKQ_TEST_", env_file=None, extra="ignore")


def _score_build(req, inputs, job_dir, settings):
    return score_argv(
        req,
        job_dir=job_dir,
        model_path=inputs["model"],
        native_path=inputs["native"],
        settings=settings,
    )


def _batch_build(req, inputs, job_dir, settings):
    return batch_argv(
        req,
        job_dir=job_dir,
        native_path=inputs["native"],
        models_dir=inputs["models_dir"],
        settings=settings,
    )


ENDPOINTS = {
    "score": CLIEndpoint(
        name="score",
        help="Score a single (model, native) pair via DockQ",
        request_model=ScoreRequest,
        build_argv=_score_build,
        inputs={
            "model": ("Model PDB/CIF file", True),
            "native": ("Native/reference PDB/CIF file", True),
        },
    ),
    "score_batch": CLIEndpoint(
        name="score_batch",
        help="Score N candidate models against 1 reference native",
        request_model=ScoreBatchRequest,
        build_argv=_batch_build,
        inputs={
            "native": ("Native/reference PDB/CIF file", True),
            "models_dir": ("Directory containing candidate model PDB/CIF files", True),
        },
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"score", "score_batch"}


def test_score_endpoint_fields():
    ep = ENDPOINTS["score"]
    assert ep.request_model is ScoreRequest
    assert ep.inputs["model"] == ("Model PDB/CIF file", True)
    assert ep.inputs["native"] == ("Native/reference PDB/CIF file", True)


def test_batch_endpoint_fields():
    ep = ENDPOINTS["score_batch"]
    assert ep.request_model is ScoreBatchRequest
    assert ep.inputs["native"][1] is True
    assert ep.inputs["models_dir"][1] is True


# ---- Build_argv callbacks ----


def test_score_build_argv(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    model = tmp_path / "model.pdb"
    native = tmp_path / "native.pdb"
    model.write_text("ATOM")
    native.write_text("ATOM")

    argv = _score_build(
        ScoreRequest(name="test"),
        {"model": model, "native": native},
        job_dir,
        s,
    )
    assert argv[0] == "DockQ"
    assert "--json" in argv
    assert "--short" in argv
    assert "--n_cpu" in argv


def test_batch_build_argv(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    native = tmp_path / "native.pdb"
    models_dir = tmp_path / "models"
    native.write_text("ATOM")
    models_dir.mkdir()

    argv = _batch_build(
        ScoreBatchRequest(sort_by="iRMSD"),
        {"native": native, "models_dir": models_dir},
        job_dir,
        s,
    )
    assert argv[0] == sys.executable
    assert "--native" in argv
    assert "--models-dir" in argv
    assert "--sort-by" in argv
    assert argv[argv.index("--sort-by") + 1] == "iRMSD"


# ---- End-to-end create_cli ----


def test_cli_score_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = DockQAdapter(settings=s)

    model = tmp_path / "model.pdb"
    native = tmp_path / "native.pdb"
    model.write_text("ATOM")
    native.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "score",
        "--model", str(model), "--native", str(native),
        "--output-dir", str(output_dir),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_score_json_output(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = DockQAdapter(settings=s)

    model = tmp_path / "model.pdb"
    native = tmp_path / "native.pdb"
    model.write_text("ATOM")
    native.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "score",
        "--model", str(model), "--native", str(native),
        "--json", "--output-dir", str(output_dir),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == "completed"
    assert result["return_code"] == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = DockQAdapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)


def test_cli_score_failure(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = DockQAdapter(settings=s)

    model = tmp_path / "model.pdb"
    native = tmp_path / "native.pdb"
    model.write_text("ATOM")
    native.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "score",
        "--model", str(model), "--native", str(native),
        "--output-dir", str(output_dir),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 1
            with pytest.raises(SystemExit) as exc_info:
                create_cli(adapter, s, ENDPOINTS, version="0.0.1")
            assert exc_info.value.code == 1


def test_cli_write_job_json(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = DockQAdapter(settings=s)

    model = tmp_path / "model.pdb"
    native = tmp_path / "native.pdb"
    model.write_text("ATOM")
    native.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "score",
        "--model", str(model), "--native", str(native),
        "--write-job-json", "--output-dir", str(output_dir),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit):
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")

    job_json = tmp_path / "run" / "job.json"
    assert job_json.exists()
    data = json.loads(job_json.read_text())
    assert data["status"] == "completed"
