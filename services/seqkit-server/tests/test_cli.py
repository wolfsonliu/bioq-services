"""CLI batch-mode tests for seqkit-server (endpoint registration + argv builders).

The endpoint dict is rebuilt here (rather than imported from `server.__main__`,
which would call `create_cli` at import time).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from bioq_service.cli import CLIEndpoint, create_cli
from pydantic_settings import SettingsConfigDict
from server.adapter import SeqkitAdapter
from server.models import RevcompRequest, StatsRequest
from server.settings import SeqkitSettings
from server.tools import revcomp_argv, stats_argv

DATA_DIR = Path(__file__).resolve().parent / "data"
FASTA = DATA_DIR / "input.fasta"


class _Off(SeqkitSettings):
    model_config = SettingsConfigDict(env_prefix="SEQKIT_TEST_", env_file=None, extra="ignore")


ENDPOINTS = {
    "stats": CLIEndpoint(
        name="stats", help="", request_model=StatsRequest,
        build_argv=lambda req, inputs, jd, s: stats_argv(req, job_dir=jd, input_fasta=inputs["input_fasta"], settings=s),
        inputs={"input_fasta": ("Input FASTA/FASTQ file", True)},
    ),
    "revcomp": CLIEndpoint(
        name="revcomp", help="", request_model=RevcompRequest,
        build_argv=lambda req, inputs, jd, s: revcomp_argv(req, job_dir=jd, input_fasta=inputs["input_fasta"], settings=s),
        inputs={"input_fasta": ("Input FASTA/FASTQ file", True)},
    ),
}


# ---- Endpoint registration ----

def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"stats", "revcomp"}


def test_inputs_required():
    assert ENDPOINTS["stats"].inputs["input_fasta"][1] is True
    assert ENDPOINTS["revcomp"].inputs["input_fasta"][1] is True


# ---- argv builders ----

def test_stats_argv_default(tmp_path):
    s = _Off(bin="/opt/seqkit/bin/seqkit", threads=3, jobs_base_dir=tmp_path)
    argv = stats_argv(StatsRequest(), job_dir=tmp_path / "j", input_fasta=FASTA, settings=s)
    assert argv[0] == "/opt/seqkit/bin/seqkit"
    assert argv[1] == "stats"
    assert "--tabular" in argv and "--all" in argv
    assert "-j" in argv and "3" in argv
    assert "-o" in argv and argv[argv.index("-o") + 1].endswith("stats.tsv")
    assert argv[-1] == str(FASTA)


def test_stats_argv_core_only(tmp_path):
    s = _Off(bin="seqkit", jobs_base_dir=tmp_path)
    argv = stats_argv(StatsRequest(all_stats=False), job_dir=tmp_path / "j", input_fasta=FASTA, settings=s)
    assert "--tabular" in argv
    assert "--all" not in argv


def test_revcomp_argv_auto(tmp_path):
    s = _Off(bin="seqkit", jobs_base_dir=tmp_path)
    argv = revcomp_argv(RevcompRequest(), job_dir=tmp_path / "j", input_fasta=FASTA, settings=s)
    assert argv[1] == "seq"
    assert "--reverse" in argv and "--complement" in argv
    assert "-t" not in argv  # auto => let seqkit guess
    assert argv[argv.index("-o") + 1].endswith("revcomp.fasta")


def test_revcomp_argv_dna(tmp_path):
    s = _Off(bin="seqkit", jobs_base_dir=tmp_path)
    argv = revcomp_argv(RevcompRequest(seq_type="dna"), job_dir=tmp_path / "j", input_fasta=FASTA, settings=s)
    assert "-t" in argv
    assert argv[argv.index("-t") + 1] == "dna"


# ---- End-to-end create_cli (subprocess mocked) ----

def test_cli_stats_success(tmp_path):
    s = _Off(bin="seqkit", jobs_base_dir=tmp_path / "jobs")
    adapter = SeqkitAdapter(settings=s)
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "stats", "--input-fasta", str(FASTA), "--output-dir", str(output_dir),
    ]), patch("bioq_service.cli.SubprocessRunner") as mock_runner:
        mock_runner.run.return_value = 0
        with patch.object(adapter, "detect_outputs", return_value=True):
            with pytest.raises(SystemExit) as exc:
                create_cli(adapter, s, ENDPOINTS, version="0.0.1")
            assert exc.value.code == 0


def test_cli_revcomp_success(tmp_path):
    s = _Off(bin="seqkit", jobs_base_dir=tmp_path / "jobs")
    adapter = SeqkitAdapter(settings=s)
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "revcomp", "--input-fasta", str(FASTA),
        "--params-json", '{"seq_type": "dna"}', "--output-dir", str(output_dir),
    ]), patch("bioq_service.cli.SubprocessRunner") as mock_runner:
        mock_runner.run.return_value = 0
        with patch.object(adapter, "detect_outputs", return_value=True):
            with pytest.raises(SystemExit) as exc:
                create_cli(adapter, s, ENDPOINTS, version="0.0.1")
            assert exc.value.code == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = SeqkitAdapter(settings=s)
    with patch.object(sys, "argv", ["prog"]), pytest.raises(SystemExit, match="2"):
        create_cli(adapter, s, ENDPOINTS)
