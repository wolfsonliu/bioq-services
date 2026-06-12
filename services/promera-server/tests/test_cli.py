"""CLI batch-mode tests for promera-server.

Tests endpoint registration, build_argv callbacks, and end-to-end create_cli.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioagent_service.cli import CLIEndpoint, create_cli

from server.adapter import PromeraAdapter
from server.models import CofoldRequest, DesignRequest
from server.settings import PromeraSettings
from server.tools import cofold_argv, build_design_config, design_argv, write_design_config

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_TARGET = DATA_DIR / "test_target.json"


class _Off(PromeraSettings):
    model_config = SettingsConfigDict(
        env_prefix="PROMERA_CLI_TEST_", env_file=None, extra="ignore"
    )


def _cofold_build(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    schema_path = input_dir / "input.json"
    shutil.copy2(inputs["input_schema"], schema_path)
    return cofold_argv(
        req, job_dir=job_dir, schema_path=schema_path, settings=settings
    )


def _design_build(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    target_dir = input_dir / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs["target_schema"], target_dir / "target.json")
    template_path = None
    if inputs.get("target_template"):
        template_path = input_dir / "target_template.cif"
        shutil.copy2(inputs["target_template"], template_path)
    cfg = build_design_config(
        req,
        target_dir=target_dir,
        output_dir=output_dir,
        template_path=template_path,
        settings=settings,
    )
    config_path = write_design_config(cfg, input_dir / "task_config.yaml")
    return design_argv(
        req, job_dir=job_dir, config_path=config_path, settings=settings
    )


ENDPOINTS = {
    "cofold": CLIEndpoint(
        name="cofold",
        help="Run structure prediction on a tinyprot JSON schema",
        request_model=CofoldRequest,
        build_argv=_cofold_build,
        inputs={"input_schema": ("Input JSON schema file", True)},
    ),
    "design": CLIEndpoint(
        name="design",
        help="Run de novo binder design on a target JSON schema",
        request_model=DesignRequest,
        build_argv=_design_build,
        inputs={
            "target_schema": ("Target JSON schema file", True),
            "target_template": ("Target template CIF file (optional)", False),
        },
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"cofold", "design"}


def test_cofold_endpoint_fields():
    ep = ENDPOINTS["cofold"]
    assert ep.request_model is CofoldRequest
    assert ep.inputs["input_schema"][1] is True


def test_design_endpoint_fields():
    ep = ENDPOINTS["design"]
    assert ep.request_model is DesignRequest
    assert ep.inputs["target_schema"][1] is True
    assert ep.inputs["target_template"][1] is False


# ---- Build_argv callbacks ----


def test_cofold_build_argv(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()

    argv = _cofold_build(
        CofoldRequest(num_seeds=2, diffusion_samples=3),
        {"input_schema": TEST_TARGET},
        job_dir,
        s,
    )
    assert len(argv) > 0
    assert any("num_seeds=2" in a for a in argv)
    assert any("promera" in a for a in argv)


def test_design_build_argv(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()

    argv = _design_build(
        DesignRequest(design_type="minibinder", num_backbones=5),
        {"target_schema": TEST_TARGET},
        job_dir,
        s,
    )
    assert len(argv) > 0
    assert "promera.inference.Design" in argv

    config_path = job_dir / "input" / "task_config.yaml"
    assert config_path.exists()


# ---- End-to-end create_cli ----


def test_cli_cofold_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = PromeraAdapter(settings=s)

    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "cofold",
        "--input-schema", str(TEST_TARGET),
        "--output-dir", str(output_dir),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_design_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = PromeraAdapter(settings=s)

    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "design",
        "--target-schema", str(TEST_TARGET),
        "--output-dir", str(output_dir),
        "--design-type", "vhh",
        "--num-backbones", "3",
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = PromeraAdapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
