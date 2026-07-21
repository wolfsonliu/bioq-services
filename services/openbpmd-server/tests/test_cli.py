"""CLI batch-mode tests for openbpmd-server."""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioq_service.cli import CLIEndpoint, create_cli

from server.adapter import OpenBPMDAdapter
from server.models import ScoreRequest
from server.settings import OpenBPMDSettings
from server.tools import score_argv


def _rst7() -> bytes:
    return b"default_name\n     3\n  1.0  2.0  3.0\n"


def _prm7() -> bytes:
    return b"%VERSION  VERSION_STAMP = V0001.000\n"


def _gro() -> bytes:
    return b"test\n    1\n    1MOL C1 1 0.0 0.0 0.0\n 1 1 1\n"


def _top() -> bytes:
    return b"; top\n[ defaults ]\n"


class _Off(OpenBPMDSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENBPMD_TEST_",
        env_file=None,
        extra="ignore",
    )


def _score_build(req, inputs, job_dir, settings):
    return score_argv(
        req,
        job_dir=job_dir,
        structure=inputs["structure"],
        parameters=inputs["parameters"],
        settings=settings,
    )


INPUTS = {
    "structure": ("Coordinate file", True),
    "parameters": ("Topology/parameter file", True),
}

ENDPOINTS = {
    "score": CLIEndpoint(
        name="score", help="BPMD scoring",
        request_model=ScoreRequest,
        build_argv=_score_build,
        inputs=INPUTS,
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"score"}


def test_score_endpoint_declares_inputs():
    ep = ENDPOINTS["score"]
    assert ep.request_model is ScoreRequest
    assert ep.inputs["structure"][1] is True
    assert ep.inputs["parameters"][1] is True


# ---- build_argv ----


def test_score_build_argv_amber(tmp_path):
    s = _Off(python="/bin/true")
    structure = tmp_path / "s.rst7"
    structure.write_bytes(_rst7())
    parameters = tmp_path / "s.prm7"
    parameters.write_bytes(_prm7())
    job_dir = tmp_path / "j"
    job_dir.mkdir()

    argv = _score_build(
        ScoreRequest(nreps=1, lig_resname="MOL"),
        {"structure": structure, "parameters": parameters},
        job_dir, s,
    )
    assert argv[0] == "/bin/true"
    assert "--structure" in argv and str(structure) in argv
    assert "--parameters" in argv and str(parameters) in argv
    assert "--nreps" in argv and "1" in argv


def test_score_build_argv_gromacs(tmp_path):
    s = _Off(python="/bin/true")
    structure = tmp_path / "s.gro"
    structure.write_bytes(_gro())
    parameters = tmp_path / "s.top"
    parameters.write_bytes(_top())
    job_dir = tmp_path / "j"
    job_dir.mkdir()

    argv = _score_build(
        ScoreRequest(system_format="gromacs"),
        {"structure": structure, "parameters": parameters},
        job_dir, s,
    )
    assert str(structure) in argv and str(parameters) in argv
    assert "--system-format" in argv and "gromacs" in argv


# ---- End-to-end create_cli ----


def _stage(tmp_path):
    structure = tmp_path / "s.rst7"
    structure.write_bytes(_rst7())
    parameters = tmp_path / "s.prm7"
    parameters.write_bytes(_prm7())
    output_dir = tmp_path / "run" / "output"
    return structure, parameters, output_dir


def test_cli_score_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = OpenBPMDAdapter(settings=s)
    structure, parameters, output_dir = _stage(tmp_path)

    with patch.object(sys, "argv", [
        "prog", "score",
        "--structure", str(structure),
        "--parameters", str(parameters),
        "--output-dir", str(output_dir),
        "--nreps", "1",
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_score_json_output(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = OpenBPMDAdapter(settings=s)
    structure, parameters, output_dir = _stage(tmp_path)

    with patch.object(sys, "argv", [
        "prog", "score",
        "--structure", str(structure),
        "--parameters", str(parameters),
        "--params-json", '{"lig_resname": "LIG", "nreps": 1}',
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
    adapter = OpenBPMDAdapter(settings=s)
    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)


def test_cli_missing_input_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = OpenBPMDAdapter(settings=s)
    with patch.object(sys, "argv", [
        "prog", "score",
        "--structure", str(tmp_path / "no-such.rst7"),
        "--parameters", str(tmp_path / "no-such.prm7"),
        "--output-dir", str(tmp_path / "out"),
    ]):
        with pytest.raises(SystemExit):
            create_cli(adapter, s, ENDPOINTS)
