"""CLI batch-mode tests for rfdiffusion-server.

Tests endpoint registration, build_argv callbacks, and end-to-end create_cli.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from bioq_service.cli import CLIEndpoint, create_cli

from server.adapter import RFdiffusionAdapter
from server.models import (
    BinderRequest,
    CustomRequest,
    MotifRequest,
    SymmetryRequest,
    UnconditionalRequest,
)
from server.settings import RFdiffusionSettings
from server.tools import (
    binder_argv,
    custom_argv,
    motif_argv,
    symmetry_argv,
    unconditional_argv,
)


@pytest.fixture
def settings(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    return RFdiffusionSettings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "rfdiffusion",
        models_dir=models,
        inference_script=tmp_path / "rfdiffusion" / "scripts" / "run_inference.py",
        python=tmp_path / "rfdiffusion" / ".venv" / "bin" / "python",
    )


def _unconditional_build(req, inputs, job_dir, settings):
    return unconditional_argv(req, job_dir, settings)


def _motif_build(req, inputs, job_dir, settings):
    return motif_argv(req, inputs["input_pdb"], job_dir, settings)


def _binder_build(req, inputs, job_dir, settings):
    return binder_argv(req, inputs["input_pdb"], job_dir, settings)


def _symmetry_build(req, inputs, job_dir, settings):
    return symmetry_argv(req, job_dir, settings)


def _custom_build(req, inputs, job_dir, settings):
    return custom_argv(req, inputs.get("input_pdb"), job_dir, settings)


ENDPOINTS = {
    "unconditional": CLIEndpoint(
        name="unconditional",
        help="Unconditional monomer backbone generation",
        request_model=UnconditionalRequest,
        build_argv=_unconditional_build,
    ),
    "motif": CLIEndpoint(
        name="motif",
        help="Motif scaffolding (input PDB + contig)",
        request_model=MotifRequest,
        build_argv=_motif_build,
        inputs={"input_pdb": ("Input PDB carrying the motif", True)},
    ),
    "binder": CLIEndpoint(
        name="binder",
        help="PPI binder design against a target PDB",
        request_model=BinderRequest,
        build_argv=_binder_build,
        inputs={"input_pdb": ("Target PDB file", True)},
    ),
    "symmetry": CLIEndpoint(
        name="symmetry",
        help="Symmetric oligomer generation",
        request_model=SymmetryRequest,
        build_argv=_symmetry_build,
    ),
    "custom": CLIEndpoint(
        name="custom",
        help="Raw contig + freeform Hydra overrides",
        request_model=CustomRequest,
        build_argv=_custom_build,
        inputs={"input_pdb": ("Optional input PDB", False)},
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"unconditional", "motif", "binder", "symmetry", "custom"}


def test_unconditional_has_no_inputs():
    assert ENDPOINTS["unconditional"].inputs == {}


def test_motif_input_required():
    assert ENDPOINTS["motif"].inputs["input_pdb"][1] is True


def test_binder_input_required():
    assert ENDPOINTS["binder"].inputs["input_pdb"][1] is True


def test_custom_input_optional():
    assert ENDPOINTS["custom"].inputs["input_pdb"][1] is False


def test_symmetry_has_no_inputs():
    assert ENDPOINTS["symmetry"].inputs == {}


# ---- Build_argv callbacks ----


def test_unconditional_build_argv(settings, tmp_path):
    argv = _unconditional_build(
        UnconditionalRequest(min_length=100, max_length=150, num_designs=4),
        {},
        tmp_path / "j",
        settings,
    )
    joined = " ".join(argv)
    assert "contigmap.contigs=[100-150]" in joined
    assert "inference.num_designs=4" in joined


def test_binder_build_argv(settings, tmp_path):
    pdb = tmp_path / "target.pdb"
    pdb.write_text("ATOM")
    argv = _binder_build(
        BinderRequest(contigs="A1-150/0 70-100", hotspots="A59,A83"),
        {"input_pdb": pdb},
        tmp_path / "j",
        settings,
    )
    joined = " ".join(argv)
    assert "ppi.hotspot_res=[A59,A83]" in joined
    assert "contigmap.contigs=" in joined


def test_motif_build_argv(settings, tmp_path):
    pdb = tmp_path / "motif.pdb"
    pdb.write_text("ATOM")
    argv = _motif_build(
        MotifRequest(contigs="10-40/A163-181/10-40"),
        {"input_pdb": pdb},
        tmp_path / "j",
        settings,
    )
    joined = " ".join(argv)
    assert "contigmap.contigs=[10-40/A163-181/10-40]" in joined


def test_symmetry_build_argv(settings, tmp_path):
    argv = _symmetry_build(
        SymmetryRequest(symmetry="c6", total_length=480),
        {},
        tmp_path / "j",
        settings,
    )
    joined = " ".join(argv)
    assert "inference.symmetry=c6" in joined
    assert "contigmap.contigs=[480-480]" in joined


def test_custom_build_argv_without_input(settings, tmp_path):
    argv = _custom_build(
        CustomRequest(contigs="100-100"),
        {},
        tmp_path / "j",
        settings,
    )
    joined = " ".join(argv)
    assert "contigmap.contigs=[100-100]" in joined


def test_custom_build_argv_with_input(settings, tmp_path):
    pdb = tmp_path / "input.pdb"
    pdb.write_text("ATOM")
    argv = _custom_build(
        CustomRequest(contigs="79-79"),
        {"input_pdb": pdb},
        tmp_path / "j",
        settings,
    )
    joined = " ".join(argv)
    assert f"inference.input_pdb={pdb}" in joined


# ---- End-to-end create_cli ----


def test_cli_unconditional_success(settings, tmp_path):
    adapter = RFdiffusionAdapter(settings=settings)
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "unconditional",
        "--min-length", "100", "--max-length", "150",
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, settings, ENDPOINTS, version="0.1.0")
                assert exc_info.value.code == 0


def test_cli_binder_with_input(settings, tmp_path):
    adapter = RFdiffusionAdapter(settings=settings)
    pdb = tmp_path / "target.pdb"
    pdb.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "binder",
        "--input-pdb", str(pdb),
        "--contigs", "A1-150/0 70-100",
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, settings, ENDPOINTS, version="0.1.0")
                assert exc_info.value.code == 0


def test_cli_no_subcommand_exits_2(settings):
    adapter = RFdiffusionAdapter(settings=settings)
    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, settings, ENDPOINTS)
