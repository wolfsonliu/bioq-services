"""CLI batch-mode tests for chembounce-server."""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioagent_service.cli import CLIEndpoint, create_cli

from server.adapter import ChemBounceAdapter
from server.models import ScaffoldHopRequest
from server.settings import ChemBounceSettings
from server.tools import scaffold_hop_argv


LOSARTAN = "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl"


class _Off(ChemBounceSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHEMBOUNCE_TEST_",
        env_file=None,
        extra="ignore",
    )


def _scaffold_hop_build(req, _inputs, job_dir, settings):
    return scaffold_hop_argv(req, job_dir=job_dir, settings=settings)


ENDPOINTS = {
    "scaffold_hop": CLIEndpoint(
        name="scaffold_hop",
        help="Run ChemBounce scaffold hopping",
        request_model=ScaffoldHopRequest,
        build_argv=_scaffold_hop_build,
        inputs={},
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"scaffold_hop"}


def test_scaffold_hop_endpoint_fields():
    ep = ENDPOINTS["scaffold_hop"]
    assert ep.request_model is ScaffoldHopRequest
    assert ep.inputs == {}


# ---- build_argv ----


def test_scaffold_hop_build_argv(tmp_path):
    s = _Off(
        python="/bin/true",
        entrypoint="/opt/chembounce.py",
        weights_dir=tmp_path / "data",
    )
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    argv = _scaffold_hop_build(
        ScaffoldHopRequest(input_smiles=LOSARTAN, frag_max_n=5),
        {}, job_dir, s,
    )
    assert argv[0] == "/bin/true"
    assert "/opt/chembounce.py" in argv
    assert "-n" in argv and "5" in argv
    assert LOSARTAN in argv


# ---- End-to-end create_cli ----


def test_cli_scaffold_hop_success(tmp_path):
    s = _Off(
        jobs_base_dir=tmp_path / "jobs",
        python="/bin/true",
        weights_dir=tmp_path / "data",
    )
    adapter = ChemBounceAdapter(settings=s)

    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "scaffold_hop",
        "--input-smiles", LOSARTAN,
        "--output-dir", str(output_dir),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_scaffold_hop_json_output(tmp_path, capsys):
    s = _Off(
        jobs_base_dir=tmp_path / "jobs",
        python="/bin/true",
        weights_dir=tmp_path / "data",
    )
    adapter = ChemBounceAdapter(settings=s)

    output_dir = tmp_path / "run" / "output"
    with patch.object(sys, "argv", [
        "prog", "scaffold_hop",
        "--input-smiles", LOSARTAN,
        "--params-json", '{"frag_max_n": 7, "database": "250mw"}',
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
    adapter = ChemBounceAdapter(settings=s)
    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
