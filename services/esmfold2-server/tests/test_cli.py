"""CLI batch-mode tests for esmfold2-server.

Tests endpoint registration, build_argv callbacks, and end-to-end create_cli.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioq_service.cli import CLIEndpoint, create_cli

from server.adapter import ESMFold2Adapter
from server.models import FoldRequest
from server.settings import ESMFold2Settings
from server.tools import build_input_json, fold_argv


class _Off(ESMFold2Settings):
    model_config = SettingsConfigDict(
        env_prefix="ESMFOLD2_TEST_", env_file=None, extra="ignore",
    )


def _fold_build(req, inputs, job_dir, settings):
    input_json = build_input_json(req, job_dir=job_dir, saved_msa_paths={})
    return fold_argv(req, job_dir=job_dir, input_json=input_json, settings=settings)


ENDPOINTS = {
    "fold": CLIEndpoint(
        name="fold",
        help="Predict 3D structure of a biomolecular complex",
        request_model=FoldRequest,
        build_argv=_fold_build,
        inputs={},
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"fold"}


def test_fold_endpoint_fields():
    ep = ENDPOINTS["fold"]
    assert ep.request_model is FoldRequest
    assert ep.inputs == {}


# ---- Build_argv callbacks ----


def test_fold_build_argv(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()

    req = FoldRequest(
        sequences=[{"type": "protein", "id": "A", "sequence": "MKTL"}],
        num_loops=3,
        num_sampling_steps=50,
    )
    argv = _fold_build(req, {}, job_dir, s)
    assert len(argv) > 0
    assert "--input-json" in argv
    assert "--num-loops" in argv


# ---- End-to-end create_cli ----


def test_cli_fold_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = ESMFold2Adapter(settings=s)

    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "fold",
        "--params-json", json.dumps({
            "sequences": [{"type": "protein", "id": "A", "sequence": "MKTL"}],
        }),
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_fold_json_output(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = ESMFold2Adapter(settings=s)

    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "fold",
        "--params-json", json.dumps({
            "sequences": [{"type": "protein", "id": "A", "sequence": "MKTL"}],
        }),
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
    adapter = ESMFold2Adapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
