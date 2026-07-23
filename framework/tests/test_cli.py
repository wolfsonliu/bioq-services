"""Tests for the CLI batch-mode module (bioq_service.cli).

Covers: pydantic→argparse conversion, request building, input resolution,
CLIEndpoint wiring, and end-to-end create_cli flow (with mocked subprocess).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from bioq_service import ServiceSettings
from bioq_service.adapter import JobAdapter
from bioq_service.cli import (
    CLIEndpoint,
    _UNSET,
    _add_model_args,
    _build_request,
    _resolve_inputs,
    _unwrap_optional,
    create_cli,
)


# ---- Test models ----


class SimpleRequest(BaseModel):
    name: str = Field(default="run", description="Job name")
    count: int = Field(default=10)
    temperature: float = Field(default=0.5)
    verbose: bool = False
    no_align: bool = True


class OptionalFieldsRequest(BaseModel):
    message: str = "hello"
    seed: Optional[int] = None
    label: Optional[str] = None


# ---- Test settings / adapter ----


class _TestSettings(ServiceSettings):
    model_config = SettingsConfigDict(env_file=None, env_prefix="CLI_TEST_", extra="ignore")


class _TestAdapter(JobAdapter):
    name = "test-cli"


# ---- Tests: type unwrapping ----


def test_unwrap_optional_int():
    inner, is_opt = _unwrap_optional(Optional[int])
    assert inner is int
    assert is_opt is True


def test_unwrap_non_optional():
    inner, is_opt = _unwrap_optional(str)
    assert inner is str
    assert is_opt is False


# ---- Tests: argparse flag generation ----


def test_add_model_args_simple():
    """Simple model fields produce correct argparse flags."""
    import argparse

    parser = argparse.ArgumentParser()
    _add_model_args(parser, SimpleRequest)

    # Parse defaults — all fields should be _UNSET when not passed
    ns = parser.parse_args([])
    assert ns.name is _UNSET
    assert ns.count is _UNSET
    assert ns.temperature is _UNSET
    assert ns.verbose is _UNSET
    assert ns.no_align is _UNSET

    # Parse overrides
    ns = parser.parse_args(["--name", "test", "--count", "20", "--verbose", "--no-no-align"])
    assert ns.name == "test"
    assert ns.count == 20
    assert ns.verbose is True
    assert ns.no_align is False


def test_add_model_args_optional():
    """Optional fields default to _UNSET and accept values."""
    import argparse

    parser = argparse.ArgumentParser()
    _add_model_args(parser, OptionalFieldsRequest)

    ns = parser.parse_args([])
    assert ns.seed is _UNSET

    ns = parser.parse_args(["--seed", "42"])
    assert ns.seed == 42


# ---- Tests: request building ----


def test_build_request_from_namespace():
    """Build a pydantic model from argparse namespace."""
    import argparse

    ns = argparse.Namespace(name="test", count=5, temperature=0.8, verbose=True, no_align=False)
    req = _build_request(SimpleRequest, ns, params_json=None)
    assert req.name == "test"
    assert req.count == 5
    assert req.verbose is True


def test_build_request_from_json_string():
    """JSON string overrides defaults."""
    import argparse

    ns = argparse.Namespace(
        name=_UNSET, count=_UNSET, temperature=_UNSET, verbose=_UNSET, no_align=_UNSET,
    )
    json_str = '{"name": "from_json", "count": 99}'
    req = _build_request(SimpleRequest, ns, params_json=json_str)
    assert req.name == "from_json"
    assert req.count == 99
    assert req.temperature == 0.5  # pydantic default


def test_build_request_from_json_file(tmp_path: Path):
    """JSON file overrides defaults."""
    import argparse

    json_file = tmp_path / "params.json"
    json_file.write_text('{"name": "file_test", "count": 7}')

    ns = argparse.Namespace(
        name=_UNSET, count=_UNSET, temperature=_UNSET, verbose=_UNSET, no_align=_UNSET,
    )
    req = _build_request(SimpleRequest, ns, params_json=str(json_file))
    assert req.name == "file_test"
    assert req.count == 7


def test_build_request_cli_overrides_json():
    """Explicit CLI args take priority over JSON values."""
    import argparse

    ns = argparse.Namespace(
        name="cli_wins", count=_UNSET, temperature=_UNSET, verbose=_UNSET, no_align=_UNSET,
    )
    json_str = '{"name": "from_json", "count": 99}'
    req = _build_request(SimpleRequest, ns, params_json=json_str)
    assert req.name == "cli_wins"
    assert req.count == 99


# ---- Tests: input resolution ----


def test_resolve_inputs_ok(tmp_path: Path):
    """Existing files are resolved to absolute paths."""
    import argparse

    pdb = tmp_path / "model.pdb"
    pdb.write_text("ATOM ...")

    ns = argparse.Namespace(model=str(pdb))
    resolved = _resolve_inputs(ns, {"model": ("Model PDB", True)})
    assert resolved["model"] == pdb.resolve()


def test_resolve_inputs_missing_required(tmp_path: Path):
    """Missing required input causes sys.exit(2)."""
    import argparse

    ns = argparse.Namespace(model=None)
    with pytest.raises(SystemExit, match="2"):
        _resolve_inputs(ns, {"model": ("Model PDB", True)})


def test_resolve_inputs_missing_optional(tmp_path: Path):
    """Missing optional input is skipped."""
    import argparse

    ns = argparse.Namespace(input_pdb=None)
    resolved = _resolve_inputs(ns, {"input_pdb": ("Optional PDB", False)})
    assert "input_pdb" not in resolved


def test_resolve_inputs_nonexistent_file(tmp_path: Path):
    """Non-existent file path causes sys.exit(2)."""
    import argparse

    ns = argparse.Namespace(model=str(tmp_path / "nope.pdb"))
    with pytest.raises(SystemExit, match="2"):
        _resolve_inputs(ns, {"model": ("Model PDB", True)})


# ---- Tests: CLIEndpoint ----


def test_cli_endpoint_dataclass():
    """CLIEndpoint holds all required fields."""

    def _build(req, inputs, job_dir, settings):
        return ["echo", "hello"]

    ep = CLIEndpoint(
        name="test",
        help="Test endpoint",
        request_model=SimpleRequest,
        build_argv=_build,
        inputs={"model": ("Model file", True)},
    )
    assert ep.name == "test"
    assert ep.inputs["model"] == ("Model file", True)


# ---- Tests: end-to-end create_cli ----


def test_create_cli_no_args(tmp_path: Path):
    """No arguments prints help and exits 2."""
    settings = _TestSettings(jobs_base_dir=tmp_path / "jobs")
    adapter = _TestAdapter(settings=settings)

    def _build(req, inputs, job_dir, settings):
        return ["echo", "ok"]

    endpoints = {
        "echo": CLIEndpoint(
            name="echo", help="Echo test",
            request_model=SimpleRequest, build_argv=_build,
        ),
    }

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, settings, endpoints)


def test_create_cli_success(tmp_path: Path):
    """Successful run exits 0 with outputs detected."""
    settings = _TestSettings(jobs_base_dir=tmp_path / "jobs")
    adapter = _TestAdapter(settings=settings)

    def _build(req, inputs, job_dir, settings):
        out = job_dir / "output"
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.txt").write_text("done")
        return ["true"]

    endpoints = {
        "echo": CLIEndpoint(
            name="echo", help="Echo test",
            request_model=SimpleRequest, build_argv=_build,
        ),
    }

    output_dir = tmp_path / "run" / "output"
    with patch.object(sys, "argv", ["prog", "echo", "--output-dir", str(output_dir)]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with pytest.raises(SystemExit) as exc_info:
                create_cli(adapter, settings, endpoints)
            assert exc_info.value.code == 0


def test_create_cli_failure(tmp_path: Path):
    """Failed subprocess returns non-zero exit."""
    settings = _TestSettings(jobs_base_dir=tmp_path / "jobs")
    adapter = _TestAdapter(settings=settings)

    def _build(req, inputs, job_dir, settings):
        return ["false"]

    endpoints = {
        "echo": CLIEndpoint(
            name="echo", help="Echo test",
            request_model=SimpleRequest, build_argv=_build,
        ),
    }

    output_dir = tmp_path / "run" / "output"
    with patch.object(sys, "argv", ["prog", "echo", "--output-dir", str(output_dir)]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 1
            with pytest.raises(SystemExit) as exc_info:
                create_cli(adapter, settings, endpoints)
            assert exc_info.value.code == 1


def test_create_cli_json_output(tmp_path: Path, capsys):
    """--json flag produces JSON output to stdout."""
    settings = _TestSettings(jobs_base_dir=tmp_path / "jobs")
    adapter = _TestAdapter(settings=settings)

    def _build(req, inputs, job_dir, settings):
        out = job_dir / "output"
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.txt").write_text("done")
        return ["true"]

    endpoints = {
        "echo": CLIEndpoint(
            name="echo", help="Echo test",
            request_model=SimpleRequest, build_argv=_build,
        ),
    }

    output_dir = tmp_path / "run" / "output"
    with patch.object(sys, "argv", ["prog", "echo", "--json", "--output-dir", str(output_dir)]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with pytest.raises(SystemExit) as exc_info:
                create_cli(adapter, settings, endpoints)
            assert exc_info.value.code == 0

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == "completed"
    assert result["return_code"] == 0
    assert result["has_outputs"] is True


def test_create_cli_write_job_json(tmp_path: Path):
    """--write-job-json writes a job.json sidecar."""
    settings = _TestSettings(jobs_base_dir=tmp_path / "jobs")
    adapter = _TestAdapter(settings=settings)

    def _build(req, inputs, job_dir, settings):
        out = job_dir / "output"
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.txt").write_text("done")
        return ["true"]

    endpoints = {
        "echo": CLIEndpoint(
            name="echo", help="Echo test",
            request_model=SimpleRequest, build_argv=_build,
        ),
    }

    output_dir = tmp_path / "run" / "output"
    with patch.object(sys, "argv", [
        "prog", "echo", "--write-job-json", "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with pytest.raises(SystemExit):
                create_cli(adapter, settings, endpoints)

    job_json = tmp_path / "run" / "job.json"
    assert job_json.exists()
    data = json.loads(job_json.read_text())
    assert data["status"] == "completed"


def test_create_cli_with_input_files(tmp_path: Path):
    """Input files are resolved and passed to build_argv."""
    settings = _TestSettings(jobs_base_dir=tmp_path / "jobs")
    adapter = _TestAdapter(settings=settings)

    captured_inputs = {}

    def _build(req, inputs, job_dir, settings):
        captured_inputs.update(inputs)
        out = job_dir / "output"
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.txt").write_text("done")
        return ["true"]

    endpoints = {
        "score": CLIEndpoint(
            name="score", help="Score test",
            request_model=SimpleRequest, build_argv=_build,
            inputs={"model": ("Model PDB", True)},
        ),
    }

    model_pdb = tmp_path / "model.pdb"
    model_pdb.write_text("ATOM ...")

    output_dir = tmp_path / "run" / "output"
    with patch.object(sys, "argv", [
        "prog", "score", "--model", str(model_pdb), "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with pytest.raises(SystemExit):
                create_cli(adapter, settings, endpoints)

    assert "model" in captured_inputs
    assert captured_inputs["model"] == model_pdb.resolve()
