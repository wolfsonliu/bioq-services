"""CLI batch-mode tests for boltzgen-server.

Tests endpoint registration, build_argv callbacks, and end-to-end create_cli.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioq_service.cli import CLIEndpoint, create_cli

from server.adapter import BoltzGenAdapter
from server.models import DesignRequest, InverseFoldRequest
from server.settings import BoltzGenSettings
from server.tools import design_argv, inverse_fold_argv


class _Off(BoltzGenSettings):
    model_config = SettingsConfigDict(env_prefix="BOLTZGEN_TEST_", env_file=None, extra="ignore")


DATA_DIR = Path(__file__).resolve().parent / "data"


def _design_build(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = input_dir / "design_spec.yaml"
    shutil.copy2(inputs["design_yaml"], yaml_path)
    return design_argv(req, job_dir=job_dir, yaml_path=yaml_path, settings=settings)


def _inverse_fold_build(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = input_dir / "design_spec.yaml"
    shutil.copy2(inputs["design_yaml"], yaml_path)
    return inverse_fold_argv(req, job_dir=job_dir, yaml_path=yaml_path, settings=settings)


ENDPOINTS = {
    "design": CLIEndpoint(
        name="design",
        help="Full BoltzGen binder design pipeline",
        request_model=DesignRequest,
        build_argv=_design_build,
        inputs={"design_yaml": ("Design specification YAML file", True)},
    ),
    "inverse_fold": CLIEndpoint(
        name="inverse_fold",
        help="BoltzGen inverse-fold-only mode",
        request_model=InverseFoldRequest,
        build_argv=_inverse_fold_build,
        inputs={"design_yaml": ("Design specification YAML file", True)},
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"design", "inverse_fold"}


def test_design_endpoint_fields():
    ep = ENDPOINTS["design"]
    assert ep.request_model is DesignRequest
    assert ep.inputs["design_yaml"][1] is True


def test_inverse_fold_endpoint_fields():
    ep = ENDPOINTS["inverse_fold"]
    assert ep.request_model is InverseFoldRequest
    assert ep.inputs["design_yaml"][1] is True


# ---- Build_argv callbacks ----


def test_design_build_argv(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    yaml_file = tmp_path / "spec.yaml"
    yaml_file.write_text("entities: []\n")

    argv = _design_build(
        DesignRequest(protocol="protein-anything", num_designs=10),
        {"design_yaml": yaml_file},
        job_dir,
        s,
    )
    assert argv[0] == s.cli
    assert "run" in argv
    assert "--protocol" in argv
    assert "protein-anything" in argv
    assert "--num_designs" in argv
    assert "10" in argv
    assert "--no_subprocess" in argv


def test_design_build_with_data_file(tmp_path):
    if not DATA_DIR.exists():
        pytest.skip("test data directory not found")
    s = _Off()
    job_dir = tmp_path / "j"

    argv = _design_build(
        DesignRequest(protocol="protein-anything", num_designs=5, budget=3),
        {"design_yaml": DATA_DIR / "vanilla.yaml"},
        job_dir,
        s,
    )
    assert "--budget" in argv
    assert "3" in argv


def test_inverse_fold_build_argv(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    yaml_file = tmp_path / "spec.yaml"
    yaml_file.write_text("entities: []\n")

    argv = _inverse_fold_build(
        InverseFoldRequest(),
        {"design_yaml": yaml_file},
        job_dir,
        s,
    )
    assert argv[0] == s.cli
    assert "--only_inverse_fold" in argv
    assert "--no_subprocess" in argv


# ---- End-to-end create_cli ----


def test_cli_design_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = BoltzGenAdapter(settings=s)

    yaml_file = tmp_path / "spec.yaml"
    yaml_file.write_text("entities: []\n")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "design",
        "--design-yaml", str(yaml_file),
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_inverse_fold_json_output(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = BoltzGenAdapter(settings=s)

    yaml_file = tmp_path / "spec.yaml"
    yaml_file.write_text("entities: []\n")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "inverse_fold",
        "--design-yaml", str(yaml_file),
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


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = BoltzGenAdapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
