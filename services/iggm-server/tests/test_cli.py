"""CLI batch-mode tests for iggm-server.

Note: importing server.__main__ would run create_cli() at module import time
(parsing pytest's argv), so we reconstruct the endpoint table locally from the
same tools/adapters — mirroring the semlaflow-server test pattern.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioq_service.cli import CLIEndpoint, create_cli

from server.adapter import IgGMAdapter
from server.models import AffinityMaturationRequest, DesignRequest, EpitopeRequest
from server.settings import IgGMSettings
from server.tools import design_argv, epitope_argv

DATA = Path(__file__).resolve().parent / "data"


class _Off(IgGMSettings):
    model_config = SettingsConfigDict(
        env_prefix="IGGM_TEST_", env_file=None, extra="ignore",
    )


def _stage(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def _design_build(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    fasta = _stage(inputs["fasta"], input_dir / "input.fasta")
    antigen = _stage(inputs["antigen"], input_dir / "antigen.pdb")
    return design_argv(
        req, job_dir=job_dir, fasta_path=fasta, antigen_path=antigen,
        settings=settings, run_task=req.run_task,
    )


def _affinity_build(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    fasta = _stage(inputs["fasta"], input_dir / "input.fasta")
    antigen = _stage(inputs["antigen"], input_dir / "antigen.pdb")
    origin = _stage(inputs["fasta_origin"], input_dir / "origin.fasta")
    return design_argv(
        req, job_dir=job_dir, fasta_path=fasta, antigen_path=antigen,
        settings=settings, run_task="affinity_maturation", fasta_origin_path=origin,
    )


def _epitope_build(_req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    fasta = _stage(inputs["fasta"], input_dir / "input.fasta")
    antigen = _stage(inputs["antigen"], input_dir / "antigen.pdb")
    return epitope_argv(
        job_dir=job_dir, fasta_path=fasta, antigen_path=antigen, settings=settings,
    )


ENDPOINTS = {
    "design": CLIEndpoint(
        name="design",
        help="Antibody design",
        request_model=DesignRequest,
        build_argv=_design_build,
        inputs={"fasta": ("Antibody FASTA", True), "antigen": ("Antigen PDB", True)},
    ),
    "affinity-maturation": CLIEndpoint(
        name="affinity-maturation",
        help="Affinity maturation",
        request_model=AffinityMaturationRequest,
        build_argv=_affinity_build,
        inputs={
            "fasta": ("Antibody FASTA", True),
            "antigen": ("Antigen PDB", True),
            "fasta_origin": ("Original FASTA", True),
        },
    ),
    "epitope": CLIEndpoint(
        name="epitope",
        help="Epitope calc",
        request_model=EpitopeRequest,
        build_argv=_epitope_build,
        inputs={"fasta": ("Complex FASTA", True), "antigen": ("Complex PDB", True)},
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"design", "affinity-maturation", "epitope"}


def test_design_endpoint_inputs():
    assert set(ENDPOINTS["design"].inputs.keys()) == {"fasta", "antigen"}


def test_affinity_endpoint_inputs():
    assert set(ENDPOINTS["affinity-maturation"].inputs.keys()) == {
        "fasta", "antigen", "fasta_origin"
    }


# ---- build_argv ----


def test_design_build_argv(tmp_path):
    s = _Off(python="/bin/true", design_script="/opt/run_design.py")
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    inputs = {"fasta": DATA / "ab_CDR_H3.fasta", "antigen": DATA / "antigen.pdb"}

    argv = _design_build(
        DesignRequest(run_task="design", num_samples=2), inputs, job_dir, s
    )
    assert argv[0] == "/bin/true"
    assert "/opt/run_design.py" in argv
    assert argv[argv.index("--run_task") + 1] == "design"
    assert (job_dir / "input" / "input.fasta").is_file()
    assert (job_dir / "input" / "antigen.pdb").is_file()


def test_epitope_build_argv(tmp_path):
    s = _Off(python="/bin/true", epitope_script="/opt/epitope.py")
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    inputs = {"fasta": DATA / "complex.fasta", "antigen": DATA / "complex.pdb"}
    argv = _epitope_build(None, inputs, job_dir, s)
    assert "/opt/epitope.py" in argv
    assert (job_dir / "input" / "input.fasta").is_file()


# ---- End-to-end create_cli ----


def test_cli_design_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = IgGMAdapter(settings=s)

    with patch.object(sys, "argv", [
        "prog", "design",
        "--fasta", str(DATA / "ab_CDR_H3.fasta"),
        "--antigen", str(DATA / "antigen.pdb"),
        "--run-task", "design",
        "--num-samples", "1",
        "--output-dir", str(tmp_path / "run" / "output"),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = IgGMAdapter(settings=s)
    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
