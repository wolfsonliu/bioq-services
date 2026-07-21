"""CLI batch-mode tests for megalodon-server.

Unconditional generation has no file inputs — `CLIEndpoint.inputs={}`, so
all params flow through argparse-generated flags or `--params-json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioq_service.cli import CLIEndpoint, create_cli

from server.adapter import MegalodonAdapter
from server.models import GenerateRequest
from server.settings import MegalodonSettings
from server.tools import generate_argv

SERVICE_DIR = Path(__file__).resolve().parent.parent
CONF_DIR = SERVICE_DIR / "upstream" / "scripts" / "conf"


class _Off(MegalodonSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEGALODON_TEST_", env_file=None, extra="ignore",
    )


def _generate_build(req, inputs, job_dir, settings):
    return generate_argv(req, job_dir=job_dir, settings=settings)


ENDPOINTS = {
    "generate": CLIEndpoint(
        name="generate",
        help="Generate 3D small molecules unconditionally with Megalodon",
        request_model=GenerateRequest,
        build_argv=_generate_build,
        inputs={},
    ),
}


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"generate"}


def test_generate_endpoint_fields():
    ep = ENDPOINTS["generate"]
    assert ep.request_model is GenerateRequest
    assert ep.inputs == {}


def test_generate_build_argv(tmp_path):
    s = _Off(
        python="/bin/true",
        inference_script="/opt/inference.py",
        weights_dir=tmp_path / "weights",
        conf_dir=CONF_DIR,
    )
    job_dir = tmp_path / "j"
    job_dir.mkdir()

    argv = _generate_build(
        GenerateRequest(model_name="drugs_diffusion", n_molecules=50,
                        timesteps=200, n_atoms_per_mol=25),
        {},
        job_dir,
        s,
    )
    assert argv[0] == "/bin/true"
    assert "/opt/inference.py" in argv
    assert "--n-molecules" in argv and "50" in argv
    assert "--timesteps" in argv and "200" in argv
    assert "--n-atoms-per-mol" in argv and "25" in argv
    # ckpt path resolves to the drugs diffusion filename.
    ci = argv.index("--ckpt-path")
    assert argv[ci + 1].endswith("ckpts/drugs/megalodon_large_diffusion.ckpt")
    # per-job config written.
    cfgi = argv.index("--config-path")
    assert Path(argv[cfgi + 1]).is_file()


def test_generate_build_argv_omits_optional(tmp_path):
    s = _Off(python="/bin/true", weights_dir=tmp_path / "w", conf_dir=CONF_DIR)
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    argv = _generate_build(GenerateRequest(model_name="qm9_fm"), {}, job_dir, s)
    assert "--n-atoms-per-mol" not in argv
    assert "--seed" not in argv


def test_cli_generate_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true", conf_dir=CONF_DIR,
             weights_dir=tmp_path / "w")
    adapter = MegalodonAdapter(settings=s)
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "generate",
        "--model-name", "drugs_diffusion",
        "--n-molecules", "10",
        "--timesteps", "100",
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_generate_json_output(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true", conf_dir=CONF_DIR,
             weights_dir=tmp_path / "w")
    adapter = MegalodonAdapter(settings=s)
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "generate",
        "--n-molecules", "5",
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


def test_cli_generate_via_params_json(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true", conf_dir=CONF_DIR,
             weights_dir=tmp_path / "w")
    adapter = MegalodonAdapter(settings=s)
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "generate",
        "--params-json",
        '{"model_name": "qm9_diffusion", "n_molecules": 20, "n_atoms_per_mol": 20, "seed": 7}',
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = MegalodonAdapter(settings=s)
    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
