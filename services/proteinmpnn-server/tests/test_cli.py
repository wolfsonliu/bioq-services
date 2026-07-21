"""CLI batch-mode tests for proteinmpnn-server.

Tests endpoint registration, build_argv callbacks (with prepare_inputs),
and end-to-end create_cli.
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

from server.adapter import ProteinMPNNAdapter
from server.models import DesignRequest, ProbsRequest, ScoreRequest
from server.settings import ProteinMPNNSettings
from server.tools import design_argv, prepare_inputs, probs_argv, score_argv


class _Off(ProteinMPNNSettings):
    model_config = SettingsConfigDict(env_prefix="PROTEINMPNN_TEST_", env_file=None, extra="ignore")


def _prepare_and_design(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs["pdb"], input_dir / inputs["pdb"].name)
    paths = prepare_inputs(
        job_dir,
        settings=settings,
        ca_only=(req.model_variant == "ca_only"),
        chains_to_design=req.chains_to_design,
        fixed_positions=req.fixed_positions,
        tied_positions=req.tied_positions,
        homooligomer=req.homooligomer,
        bias_AA=req.bias_AA,
        bias_by_res=req.bias_by_res,
        omit_AA_per_chain=req.omit_AA_per_chain,
    )
    return design_argv(req, job_dir=job_dir, paths=paths, settings=settings)


def _prepare_and_score(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs["pdb"], input_dir / inputs["pdb"].name)
    paths = prepare_inputs(
        job_dir,
        settings=settings,
        ca_only=(req.model_variant == "ca_only"),
        chains_to_design=req.chains_to_design,
        fixed_positions=None,
        tied_positions=None,
        homooligomer=False,
        bias_AA=None,
        bias_by_res=None,
        omit_AA_per_chain=None,
    )
    return score_argv(req, job_dir=job_dir, paths=paths, settings=settings)


def _prepare_and_probs(req, inputs, job_dir, settings):
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs["pdb"], input_dir / inputs["pdb"].name)
    paths = prepare_inputs(
        job_dir,
        settings=settings,
        ca_only=(req.model_variant == "ca_only"),
        chains_to_design=req.chains_to_design,
        fixed_positions=None,
        tied_positions=None,
        homooligomer=False,
        bias_AA=None,
        bias_by_res=None,
        omit_AA_per_chain=None,
    )
    return probs_argv(req, job_dir=job_dir, paths=paths, settings=settings)


def _make_endpoints():
    return {
        "design": CLIEndpoint(
            name="design",
            help="Sequence design (FASTA output) over the input PDB",
            request_model=DesignRequest,
            build_argv=_prepare_and_design,
            inputs={"pdb": ("Input PDB file", True)},
        ),
        "score": CLIEndpoint(
            name="score",
            help="Score a (structure, sequence) pair",
            request_model=ScoreRequest,
            build_argv=_prepare_and_score,
            inputs={"pdb": ("Input PDB file", True)},
        ),
        "probs": CLIEndpoint(
            name="probs",
            help="Per-residue AA probability output",
            request_model=ProbsRequest,
            build_argv=_prepare_and_probs,
            inputs={"pdb": ("Input PDB file", True)},
        ),
    }


ENDPOINTS = _make_endpoints()


def _make_settings_with_helpers(tmp_path):
    """Create _Off settings with stub helper scripts."""
    s = _Off(root=tmp_path / "pmpnn", weights_dir=tmp_path / "pmpnn")
    helpers = s.root / "helper_scripts"
    helpers.mkdir(parents=True)
    stub_body = (
        "import sys, pathlib;\n"
        "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
        "pathlib.Path(args['--output_path']).write_text('{}')\n"
    )
    for name in (
        "parse_multiple_chains.py",
        "assign_fixed_chains.py",
        "make_fixed_positions_dict.py",
        "make_tied_positions_dict.py",
    ):
        (helpers / name).write_text(stub_body)
    return s


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"design", "score", "probs"}


def test_design_endpoint_fields():
    ep = ENDPOINTS["design"]
    assert ep.request_model is DesignRequest
    assert ep.inputs["pdb"] == ("Input PDB file", True)


def test_score_endpoint_fields():
    ep = ENDPOINTS["score"]
    assert ep.request_model is ScoreRequest


def test_probs_endpoint_fields():
    ep = ENDPOINTS["probs"]
    assert ep.request_model is ProbsRequest


# ---- Build_argv callbacks ----


def test_design_build_argv(tmp_path):
    s = _make_settings_with_helpers(tmp_path)
    job_dir = tmp_path / "job"
    (job_dir / "input").mkdir(parents=True)
    pdb = tmp_path / "input.pdb"
    pdb.write_text("ATOM\n")

    argv = _prepare_and_design(
        DesignRequest(),
        {"pdb": pdb},
        job_dir,
        s,
    )
    assert "protein_mpnn_run.py" in argv
    assert "--jsonl_path" in argv
    assert "--num_seq_per_target" in argv
    assert argv[argv.index("--num_seq_per_target") + 1] == "8"


def test_score_build_argv(tmp_path):
    s = _make_settings_with_helpers(tmp_path)
    job_dir = tmp_path / "job"
    (job_dir / "input").mkdir(parents=True)
    pdb = tmp_path / "input.pdb"
    pdb.write_text("ATOM\n")

    argv = _prepare_and_score(
        ScoreRequest(),
        {"pdb": pdb},
        job_dir,
        s,
    )
    assert "--score_only" in argv
    assert argv[argv.index("--score_only") + 1] == "1"


def test_probs_build_argv(tmp_path):
    s = _make_settings_with_helpers(tmp_path)
    job_dir = tmp_path / "job"
    (job_dir / "input").mkdir(parents=True)
    pdb = tmp_path / "input.pdb"
    pdb.write_text("ATOM\n")

    argv = _prepare_and_probs(
        ProbsRequest(kind="conditional"),
        {"pdb": pdb},
        job_dir,
        s,
    )
    assert "--conditional_probs_only" in argv


# ---- End-to-end create_cli ----


def test_cli_design_success(tmp_path):
    s = _make_settings_with_helpers(tmp_path)
    s.jobs_base_dir = tmp_path / "jobs"
    adapter = ProteinMPNNAdapter(settings=s)

    pdb = tmp_path / "input.pdb"
    pdb.write_text("ATOM\n")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "design",
        "--pdb", str(pdb),
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(root=tmp_path / "pmpnn", weights_dir=tmp_path / "pmpnn")
    adapter = ProteinMPNNAdapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
