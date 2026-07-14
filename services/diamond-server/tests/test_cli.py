"""CLI batch-mode tests for diamond-server (endpoint registration + create_cli).

Covers the CLI-only `makedb` command plus blastp/blastx/cluster/msa. The
endpoint dict is rebuilt here (rather than imported from `server.__main__`,
which would call `create_cli` at import time).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioagent_service.cli import CLIEndpoint, create_cli

from server.adapter import DiamondAdapter
from server.models import (
    BlastpRequest,
    BlastxRequest,
    ClusterRequest,
    MakedbRequest,
    MsaRequest,
)
from server.settings import DiamondSettings
from server.tools import blastp_argv, blastx_argv, cluster_argv, makedb_argv, msa_argv

DATA_DIR = Path(__file__).resolve().parent / "data"
QUERY = DATA_DIR / "query.faa"
SUBJECT = DATA_DIR / "subject.faa"


class _Off(DiamondSettings):
    model_config = SettingsConfigDict(env_prefix="DIAMOND_TEST_", env_file=None, extra="ignore")


ENDPOINTS = {
    "makedb": CLIEndpoint(
        name="makedb", help="", request_model=MakedbRequest,
        build_argv=lambda req, inputs, jd, s: makedb_argv(req, job_dir=jd, sequences_path=inputs["sequences"], settings=s),
        inputs={"sequences": ("Protein FASTA to index", True)},
    ),
    "blastp": CLIEndpoint(
        name="blastp", help="", request_model=BlastpRequest,
        build_argv=lambda req, inputs, jd, s: blastp_argv(req, job_dir=jd, query_path=inputs["query"], db_path=inputs.get("db"), subject_path=inputs.get("subject"), settings=s),
        inputs={"query": ("q", True), "db": ("db", False), "subject": ("subj", False)},
    ),
    "blastx": CLIEndpoint(
        name="blastx", help="", request_model=BlastxRequest,
        build_argv=lambda req, inputs, jd, s: blastx_argv(req, job_dir=jd, query_path=inputs["query"], db_path=inputs.get("db"), subject_path=inputs.get("subject"), settings=s),
        inputs={"query": ("q", True), "db": ("db", False), "subject": ("subj", False)},
    ),
    "cluster": CLIEndpoint(
        name="cluster", help="", request_model=ClusterRequest,
        build_argv=lambda req, inputs, jd, s: cluster_argv(req, job_dir=jd, sequences_path=inputs["sequences"], settings=s),
        inputs={"sequences": ("seqs", True)},
    ),
    "msa": CLIEndpoint(
        name="msa", help="", request_model=MsaRequest,
        build_argv=lambda req, inputs, jd, s: msa_argv(req, job_dir=jd, query_path=inputs["query"], db_path=inputs["db"], settings=s),
        inputs={"query": ("q", True), "db": ("db", True)},
    ),
}


# ---- Endpoint registration ----

def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"makedb", "blastp", "blastx", "cluster", "msa"}


def test_blastp_optional_db_and_subject():
    ep = ENDPOINTS["blastp"]
    assert ep.inputs["query"][1] is True
    assert ep.inputs["db"][1] is False
    assert ep.inputs["subject"][1] is False


def test_msa_requires_db():
    ep = ENDPOINTS["msa"]
    assert ep.inputs["query"][1] is True
    assert ep.inputs["db"][1] is True


# ---- build_argv callbacks ----

def test_makedb_build(tmp_path):
    s = _Off(binary="diamond", jobs_base_dir=tmp_path)
    ep = ENDPOINTS["makedb"]
    argv = ep.build_argv(ep.request_model(name="ref"), {"sequences": SUBJECT}, tmp_path / "j", s)
    assert argv[1] == "makedb"


def test_blastp_build_with_db(tmp_path):
    s = _Off(binary="diamond", jobs_base_dir=tmp_path)
    ep = ENDPOINTS["blastp"]
    argv = ep.build_argv(
        ep.request_model(name="h"),
        {"query": QUERY, "db": tmp_path / "ref.dmnd"},
        tmp_path / "j", s,
    )
    assert "--db" in argv
    assert "--subject" not in argv


# ---- End-to-end create_cli (subprocess mocked) ----

def test_cli_makedb_success(tmp_path):
    s = _Off(binary="diamond", jobs_base_dir=tmp_path / "jobs")
    adapter = DiamondAdapter(settings=s)
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "makedb", "--sequences", str(SUBJECT), "--output-dir", str(output_dir),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc.value.code == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = DiamondAdapter(settings=s)
    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
