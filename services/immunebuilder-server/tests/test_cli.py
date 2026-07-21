"""CLI batch-mode tests for immunebuilder-server.

Tests endpoint registration, build_argv callbacks, and end-to-end create_cli.
No input files needed — sequences come from the request model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioq_service.cli import CLIEndpoint, create_cli

from server.adapter import ImmuneBuilderAdapter
from server.models import AntibodyRequest, NanobodyRequest, TCRRequest
from server.settings import ImmuneBuilderSettings
from server.tools import (
    predict_antibody_argv,
    predict_nanobody_argv,
    predict_tcr_argv,
    write_fasta,
)


class _Off(ImmuneBuilderSettings):
    model_config = SettingsConfigDict(env_prefix="IMMUNEBUILDER_TEST_", env_file=None, extra="ignore")


HEAVY_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
LIGHT_SEQ = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
NANOBODY_SEQ = "QVQLVESGGGLVQAGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAINSGGGSTYYPDSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAKDGYGGSFDYWGQGTQVTVSS"
ALPHA_SEQ = "METLLGVSLVILWLQLARVNSQQGEEDPQALSIQEGENATMNCSYKTSINNLQWYRQNSGRGLVHLILIRSNEREKHSGRLRVTLDTSKKSSSLLITASRAADTASYFCAVNFGGGKLIFGQGTELSVKP"
BETA_SEQ = "NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSFSTCSANYGYTFGSGTRLTVVEDL"


def _antibody_build(req, inputs, job_dir, settings):
    fasta_path = write_fasta(
        {"H": req.heavy_sequence, "L": req.light_sequence},
        job_dir / "input" / "input.fasta",
    )
    return predict_antibody_argv(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings)


def _nanobody_build(req, inputs, job_dir, settings):
    fasta_path = write_fasta(
        {"H": req.heavy_sequence},
        job_dir / "input" / "input.fasta",
    )
    return predict_nanobody_argv(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings)


def _tcr_build(req, inputs, job_dir, settings):
    fasta_path = write_fasta(
        {"A": req.alpha_sequence, "B": req.beta_sequence},
        job_dir / "input" / "input.fasta",
    )
    return predict_tcr_argv(req, job_dir=job_dir, fasta_path=fasta_path, settings=settings)


ENDPOINTS = {
    "predict_antibody": CLIEndpoint(
        name="predict_antibody",
        help="Predict antibody structure from heavy + light chain sequences",
        request_model=AntibodyRequest,
        build_argv=_antibody_build,
    ),
    "predict_nanobody": CLIEndpoint(
        name="predict_nanobody",
        help="Predict nanobody structure from heavy chain sequence",
        request_model=NanobodyRequest,
        build_argv=_nanobody_build,
    ),
    "predict_tcr": CLIEndpoint(
        name="predict_tcr",
        help="Predict TCR structure from alpha + beta chain sequences",
        request_model=TCRRequest,
        build_argv=_tcr_build,
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"predict_antibody", "predict_nanobody", "predict_tcr"}


def test_antibody_endpoint_has_no_file_inputs():
    assert ENDPOINTS["predict_antibody"].inputs == {}
    assert ENDPOINTS["predict_antibody"].request_model is AntibodyRequest


def test_nanobody_endpoint_has_no_file_inputs():
    assert ENDPOINTS["predict_nanobody"].inputs == {}
    assert ENDPOINTS["predict_nanobody"].request_model is NanobodyRequest


def test_tcr_endpoint_has_no_file_inputs():
    assert ENDPOINTS["predict_tcr"].inputs == {}
    assert ENDPOINTS["predict_tcr"].request_model is TCRRequest


# ---- Build_argv callbacks ----


def test_antibody_build_argv(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"

    argv = _antibody_build(
        AntibodyRequest(heavy_sequence=HEAVY_SEQ, light_sequence=LIGHT_SEQ),
        {},
        job_dir,
        s,
    )
    assert "ABodyBuilder2" in argv[0]
    assert "-f" in argv
    assert "-n" in argv
    idx = argv.index("-n")
    assert argv[idx + 1] == "imgt"

    fasta = job_dir / "input" / "input.fasta"
    assert fasta.exists()
    content = fasta.read_text()
    assert ">H\n" in content
    assert ">L\n" in content


def test_nanobody_build_argv(tmp_path):
    s = _Off()
    argv = _nanobody_build(
        NanobodyRequest(heavy_sequence=NANOBODY_SEQ),
        {},
        tmp_path / "j",
        s,
    )
    assert "NanoBodyBuilder2" in argv[0]


def test_tcr_build_argv(tmp_path):
    s = _Off()
    argv = _tcr_build(
        TCRRequest(alpha_sequence=ALPHA_SEQ, beta_sequence=BETA_SEQ),
        {},
        tmp_path / "j",
        s,
    )
    assert "TCRBuilder2" in argv[0]

    fasta = tmp_path / "j" / "input" / "input.fasta"
    content = fasta.read_text()
    assert ">A\n" in content
    assert ">B\n" in content


def test_antibody_build_argv_chothia(tmp_path):
    s = _Off()
    argv = _antibody_build(
        AntibodyRequest(
            heavy_sequence=HEAVY_SEQ,
            light_sequence=LIGHT_SEQ,
            numbering_scheme="chothia",
        ),
        {},
        tmp_path / "j",
        s,
    )
    idx = argv.index("-n")
    assert argv[idx + 1] == "chothia"


# ---- End-to-end create_cli ----


def test_cli_antibody_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = ImmuneBuilderAdapter(settings=s)
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "predict_antibody",
        "--heavy-sequence", HEAVY_SEQ,
        "--light-sequence", LIGHT_SEQ,
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_nanobody_json_output(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = ImmuneBuilderAdapter(settings=s)
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "predict_nanobody",
        "--heavy-sequence", NANOBODY_SEQ,
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


def test_cli_tcr_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = ImmuneBuilderAdapter(settings=s)
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "predict_tcr",
        "--alpha-sequence", ALPHA_SEQ,
        "--beta-sequence", BETA_SEQ,
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
    adapter = ImmuneBuilderAdapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
