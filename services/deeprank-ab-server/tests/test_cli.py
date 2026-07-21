"""CLI batch-mode tests for deeprank-ab-server.

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

from server.adapter import DeepRankAbAdapter
from server.argv import score_argv
from server.models import ScoreRequest
from server.settings import DeepRankAbSettings


class _Off(DeepRankAbSettings):
    model_config = SettingsConfigDict(env_prefix="DEEPRANK_AB_TEST_", env_file=None, extra="ignore")


DATA_DIR = Path(__file__).resolve().parent / "data"


def _score_build(req, inputs, job_dir, settings):
    return score_argv(
        req,
        job_dir=job_dir,
        pdb_path=inputs["input_pdb"],
        settings=settings,
    )


ENDPOINTS = {
    "score": CLIEndpoint(
        name="score",
        help="Score an antibody-antigen docking complex via DeepRank-Ab",
        request_model=ScoreRequest,
        build_argv=_score_build,
        inputs={"input_pdb": ("Input PDB file with antibody-antigen complex", True)},
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"score"}


def test_score_endpoint_fields():
    ep = ENDPOINTS["score"]
    assert ep.request_model is ScoreRequest
    assert ep.inputs["input_pdb"][1] is True


# ---- Build_argv callbacks ----


def test_score_build_argv_basic(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    pdb = tmp_path / "complex.pdb"
    pdb.write_text("ATOM")

    argv = _score_build(
        ScoreRequest(),
        {"input_pdb": pdb},
        job_dir,
        s,
    )
    assert argv[0] == "bash"
    assert argv[1] == "-c"
    cmd = argv[2]
    assert "cd " in cmd
    assert str(job_dir / "output") in cmd
    assert "inference.py" in cmd
    assert " H " in cmd or "'H'" in cmd
    assert " L " in cmd or "'L'" in cmd
    assert " A" in cmd or "'A'" in cmd


def test_score_build_argv_custom_chains(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    pdb = tmp_path / "complex.pdb"
    pdb.write_text("ATOM")

    argv = _score_build(
        ScoreRequest(heavy_chain_id="B", light_chain_id="C", antigen_chain_id="D"),
        {"input_pdb": pdb},
        job_dir,
        s,
    )
    cmd = argv[2]
    assert " B " in cmd or "'B'" in cmd
    assert " C " in cmd or "'C'" in cmd
    assert " D" in cmd or "'D'" in cmd


def test_score_build_argv_nanobody(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()

    argv = _score_build(
        ScoreRequest(light_chain_id="-"),
        {"input_pdb": tmp_path / "vhh.pdb"},
        job_dir,
        s,
    )
    cmd = argv[2]
    assert " - " in cmd or "'-'" in cmd


def test_score_build_with_data_file(tmp_path):
    if not (DATA_DIR / "test.pdb").exists():
        pytest.skip("test data not found")
    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()

    argv = _score_build(
        ScoreRequest(heavy_chain_id="H", light_chain_id="L", antigen_chain_id="A"),
        {"input_pdb": DATA_DIR / "test.pdb"},
        job_dir,
        s,
    )
    assert argv[0] == "bash"
    cmd = argv[2]
    assert "test.pdb" in cmd


# ---- End-to-end create_cli ----


def test_cli_score_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = DeepRankAbAdapter(settings=s)

    pdb = tmp_path / "complex.pdb"
    pdb.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "score",
        "--input-pdb", str(pdb),
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_score_custom_chains(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = DeepRankAbAdapter(settings=s)

    pdb = tmp_path / "complex.pdb"
    pdb.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "score",
        "--input-pdb", str(pdb),
        "--heavy-chain-id", "B",
        "--light-chain-id", "C",
        "--antigen-chain-id", "D",
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_score_json_output(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = DeepRankAbAdapter(settings=s)

    pdb = tmp_path / "complex.pdb"
    pdb.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "score",
        "--input-pdb", str(pdb),
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


def test_cli_score_failure(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = DeepRankAbAdapter(settings=s)

    pdb = tmp_path / "complex.pdb"
    pdb.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "score",
        "--input-pdb", str(pdb),
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 1
            with pytest.raises(SystemExit) as exc_info:
                create_cli(adapter, s, ENDPOINTS, version="0.0.1")
            assert exc_info.value.code == 1


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = DeepRankAbAdapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
