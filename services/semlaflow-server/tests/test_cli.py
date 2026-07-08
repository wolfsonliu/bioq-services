"""CLI batch-mode tests for semlaflow-server.

Unconditional generation has no file inputs — `CLIEndpoint.inputs={}`, so
all params flow through argparse-generated flags or `--params-json`.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioagent_service.cli import CLIEndpoint, create_cli

from server.adapter import SemlaFlowAdapter
from server.models import GenerateRequest
from server.settings import SemlaFlowSettings
from server.tools import generate_argv


class _Off(SemlaFlowSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEMLAFLOW_TEST_",
        env_file=None,
        extra="ignore",
    )


def _generate_build(req, inputs, job_dir, settings):
    return generate_argv(req, job_dir=job_dir, settings=settings)


ENDPOINTS = {
    "generate": CLIEndpoint(
        name="generate",
        help="Generate 3D small molecules unconditionally with SemlaFlow",
        request_model=GenerateRequest,
        build_argv=_generate_build,
        inputs={},
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"generate"}


def test_generate_endpoint_fields():
    ep = ENDPOINTS["generate"]
    assert ep.request_model is GenerateRequest
    assert ep.inputs == {}


# ---- build_argv ----


def test_generate_build_argv(tmp_path):
    s = _Off(
        python="/bin/true",
        inference_script="/opt/inference.py",
        weights_dir=tmp_path / "weights",
    )
    job_dir = tmp_path / "j"
    job_dir.mkdir()

    argv = _generate_build(
        GenerateRequest(model_name="qm9", n_molecules=50, integration_steps=80),
        {},
        job_dir,
        s,
    )
    assert argv[0] == "/bin/true"
    assert "/opt/inference.py" in argv
    assert "--n-molecules" in argv and "50" in argv
    assert "--integration-steps" in argv and "80" in argv
    assert "--dataset" in argv
    idx = argv.index("--dataset")
    assert argv[idx + 1] == "qm9"


# ---- End-to-end create_cli ----


def test_cli_generate_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = SemlaFlowAdapter(settings=s)
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "generate",
        "--model-name", "qm9",
        "--n-molecules", "10",
        "--integration-steps", "50",
        "--output-dir", str(output_dir),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_generate_json_output(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = SemlaFlowAdapter(settings=s)
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "generate",
        "--n-molecules", "5",
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


def test_cli_generate_via_params_json(tmp_path):
    """Complex params can go through --params-json as an alternative."""
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = SemlaFlowAdapter(settings=s)
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "generate",
        "--params-json",
        '{"model_name": "qm9", "n_molecules": 20, "integration_steps": 60, "seed": 7}',
        "--output-dir", str(output_dir),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = SemlaFlowAdapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
