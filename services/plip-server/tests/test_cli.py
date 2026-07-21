"""CLI batch-mode tests for plip-server (endpoint registration + argv builder).

The endpoint dict is rebuilt here (rather than imported from `server.__main__`,
which would call `create_cli` at import time).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioq_service.cli import CLIEndpoint, create_cli

from server.adapter import PlipAdapter
from server.models import ProfileRequest
from server.settings import PlipSettings
from server.tools import profile_argv

DATA_DIR = Path(__file__).resolve().parent / "data"
PDB = DATA_DIR / "1vsn.pdb"


class _Off(PlipSettings):
    model_config = SettingsConfigDict(env_prefix="PLIP_TEST_", env_file=None, extra="ignore")


ENDPOINTS = {
    "profile": CLIEndpoint(
        name="profile", help="", request_model=ProfileRequest,
        build_argv=lambda req, inputs, jd, s: profile_argv(req, job_dir=jd, input_pdb=inputs["input_pdb"], settings=s),
        inputs={"input_pdb": ("Input PDB complex", True)},
    ),
}


# ---- Endpoint registration ----

def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"profile"}


def test_profile_requires_pdb():
    ep = ENDPOINTS["profile"]
    assert ep.inputs["input_pdb"][1] is True


# ---- argv builder ----

def test_profile_argv_default(tmp_path):
    s = _Off(python="python", threads=3, jobs_base_dir=tmp_path)
    argv = profile_argv(ProfileRequest(name="vsn"), job_dir=tmp_path / "j", input_pdb=PDB, settings=s)
    assert argv[:3] == ["python", "-m", "plip.plipcmd"]
    assert "-f" in argv and str(PDB) in argv
    assert "-x" in argv and "-t" in argv  # default report_formats
    assert "--name" in argv and "vsn" in argv
    assert "--maxthreads" in argv and "3" in argv


def test_profile_argv_peptide_mode(tmp_path):
    s = _Off(python="python", jobs_base_dir=tmp_path)
    req = ProfileRequest(name="x", mode="peptide", peptide_chains=["I", "J"])
    argv = profile_argv(req, job_dir=tmp_path / "j", input_pdb=PDB, settings=s)
    assert "--peptides" in argv
    idx = argv.index("--peptides")
    assert argv[idx + 1] == "I" and argv[idx + 2] == "J"


def test_profile_argv_intra_mode(tmp_path):
    s = _Off(python="python", jobs_base_dir=tmp_path)
    req = ProfileRequest(name="x", mode="intra", intra_chain="A")
    argv = profile_argv(req, job_dir=tmp_path / "j", input_pdb=PDB, settings=s)
    assert "--intra" in argv
    assert argv[argv.index("--intra") + 1] == "A"


def test_profile_argv_dnareceptor_and_flags(tmp_path):
    s = _Off(python="python", jobs_base_dir=tmp_path)
    req = ProfileRequest(
        name="x", mode="dnareceptor", report_formats=["xml"],
        pymol_session=True, render_images=True, nofix=True, nohydro=True,
    )
    argv = profile_argv(req, job_dir=tmp_path / "j", input_pdb=PDB, settings=s)
    assert "--dnareceptor" in argv
    assert "-x" in argv and "-t" not in argv
    assert "-y" in argv and "-p" in argv
    assert "--nofix" in argv and "--nohydro" in argv


# ---- End-to-end create_cli (subprocess mocked) ----

def test_cli_profile_success(tmp_path):
    s = _Off(python="python", jobs_base_dir=tmp_path / "jobs")
    adapter = PlipAdapter(settings=s)
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "profile", "--input-pdb", str(PDB), "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc.value.code == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = PlipAdapter(settings=s)
    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
