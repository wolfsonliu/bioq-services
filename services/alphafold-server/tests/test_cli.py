"""CLI batch-mode tests for alphafold-server.

Tests endpoint registration, build_argv callbacks, and end-to-end create_cli.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioq_service.cli import CLIEndpoint, create_cli

from server.adapter import AlphaFoldAdapter
from server.models import FoldRequest
from server.settings import AlphaFoldSettings
from server.tools import fold_argv


class _Off(AlphaFoldSettings):
    model_config = SettingsConfigDict(
        env_prefix="ALPHAFOLD_TEST_", env_file=None, extra="ignore",
    )


def _fold_build(req, inputs, job_dir, settings):
    fasta_path = inputs["input_fasta"]
    return fold_argv(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings)


ENDPOINTS = {
    "fold": CLIEndpoint(
        name="fold",
        help="Predict protein structure using AlphaFold v2.3.2",
        request_model=FoldRequest,
        build_argv=_fold_build,
        inputs={"input_fasta": ("Input FASTA file path", True)},
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"fold"}


def test_fold_endpoint_fields():
    ep = ENDPOINTS["fold"]
    assert ep.request_model is FoldRequest
    assert "input_fasta" in ep.inputs
    assert ep.inputs["input_fasta"][1] is True


# ---- Build_argv callbacks ----


def test_fold_build_argv(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()

    fasta = tmp_path / "input.fasta"
    fasta.write_text(">A\nMKTL\n")

    req = FoldRequest(model_preset="monomer_ptm", db_preset="reduced_dbs")
    argv = _fold_build(req, {"input_fasta": fasta}, job_dir, s)
    assert len(argv) > 0
    assert "--fasta_paths" in argv
    assert "--model_preset" in argv


# ---- End-to-end create_cli ----


def test_cli_fold_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = AlphaFoldAdapter(settings=s)

    fasta = tmp_path / "input.fasta"
    fasta.write_text(">A\nMKTL\n")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "fold",
        "--input-fasta", str(fasta),
        "--params-json", json.dumps({"model_preset": "monomer_ptm"}),
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
    adapter = AlphaFoldAdapter(settings=s)

    fasta = tmp_path / "input.fasta"
    fasta.write_text(">A\nMKTL\n")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "fold",
        "--input-fasta", str(fasta),
        "--params-json", json.dumps({"model_preset": "monomer_ptm"}),
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
    adapter = AlphaFoldAdapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
