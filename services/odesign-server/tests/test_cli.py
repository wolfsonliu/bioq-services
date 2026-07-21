"""CLI batch-mode tests for odesign-server.

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

from server.adapter import ODesignAdapter
from server.models import DesignRequest
from server.settings import ODesignSettings
from server.tools import design_argv


class _Off(ODesignSettings):
    model_config = SettingsConfigDict(env_prefix="ODESIGN_TEST_", env_file=None, extra="ignore")


DATA_DIR = Path(__file__).resolve().parent / "data"


def _design_build(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    json_path = input_dir / "input.json"
    shutil.copy2(inputs["input_json"], json_path)
    return design_argv(req, job_dir=job_dir, json_path=json_path, settings=settings)


ENDPOINTS = {
    "design": CLIEndpoint(
        name="design",
        help="ODesign biomolecular interaction design",
        request_model=DesignRequest,
        build_argv=_design_build,
        inputs={"input_json": ("JSON specification file", True)},
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"design"}


def test_design_endpoint_fields():
    ep = ENDPOINTS["design"]
    assert ep.request_model is DesignRequest
    assert ep.inputs["input_json"][1] is True


# ---- Build_argv callbacks ----


def test_design_build_argv_prot_flex(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    json_file = tmp_path / "input.json"
    json_file.write_text('[{"name":"test"}]')

    argv = _design_build(
        DesignRequest(model="odesign_base_prot_flex", n_sample=3),
        {"input_json": json_file},
        job_dir,
        s,
    )
    assert argv[0] == s.python
    assert argv[1] == s.inference_script
    assert "exp=train_odesign_base_prot_flex" in argv
    assert "exp.design_modality=protein" in argv
    assert "exp.model.sample_diffusion.N_sample=3" in argv


def test_design_build_argv_na_rigid(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    json_file = tmp_path / "input.json"
    json_file.write_text('[{"name":"test"}]')

    argv = _design_build(
        DesignRequest(model="odesign_base_na_rigid", design_modality="rna"),
        {"input_json": json_file},
        job_dir,
        s,
    )
    assert "exp=train_odesign_base_na_rigid" in argv
    assert "exp.design_modality=rna" in argv


def test_design_build_with_data_file(tmp_path):
    if not (DATA_DIR / "fc_design.json").exists():
        pytest.skip("test data not found")
    s = _Off()
    job_dir = tmp_path / "j"

    argv = _design_build(
        DesignRequest(model="odesign_base_prot_flex", n_sample=2),
        {"input_json": DATA_DIR / "fc_design.json"},
        job_dir,
        s,
    )
    assert "exp.model.sample_diffusion.N_sample=2" in argv


def test_design_build_argv_partial_diff(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    json_file = tmp_path / "input.json"
    json_file.write_text('[{"name":"test"}]')

    argv = _design_build(
        DesignRequest(enable_partial_diff=True, partial_diff_snr=0.5),
        {"input_json": json_file},
        job_dir,
        s,
    )
    assert any("partial_diffusion.enable=true" in a for a in argv)
    assert any("partial_diffusion.snr=0.5" in a for a in argv)


# ---- End-to-end create_cli ----


def test_cli_design_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = ODesignAdapter(settings=s)

    json_file = tmp_path / "input.json"
    json_file.write_text('[{"name":"test"}]')
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "design",
        "--input-json", str(json_file),
        "--model", "odesign_base_prot_flex",
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_design_json_output(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = ODesignAdapter(settings=s)

    json_file = tmp_path / "input.json"
    json_file.write_text('[{"name":"test"}]')
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "design",
        "--input-json", str(json_file),
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
    adapter = ODesignAdapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
