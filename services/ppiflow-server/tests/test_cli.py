"""CLI batch-mode tests for ppiflow-server.

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

from server.adapter import PPIFlowAdapter
from server.models import (
    AntibodyRequest,
    BinderRequest,
    MonomerRequest,
    NanobodyRequest,
    ScaffoldingRequest,
)
from server.settings import PPIFlowSettings
from server.tools import (
    antibody_argv,
    binder_argv,
    monomer_argv,
    nanobody_argv,
    scaffolding_argv,
)


class _Off(PPIFlowSettings):
    model_config = SettingsConfigDict(
        env_prefix="PPIFLOW_TEST_", env_file=None, extra="ignore",
    )


@pytest.fixture
def settings(tmp_path):
    s = _Off(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path / "ppiflow",
        ckpt_dir=tmp_path / "ppiflow" / "checkpoint",
        config_dir=tmp_path / "ppiflow" / "configs",
    )
    s.ckpt_dir.mkdir(parents=True, exist_ok=True)
    s.config_dir.mkdir(parents=True, exist_ok=True)
    s.root.mkdir(parents=True, exist_ok=True)
    return s


def _binder_build(req, inputs, job_dir, settings):
    return binder_argv(req, inputs["target"], job_dir, settings)


def _antibody_build(req, inputs, job_dir, settings):
    return antibody_argv(req, inputs["antigen"], inputs["framework"], job_dir, settings)


def _nanobody_build(req, inputs, job_dir, settings):
    return nanobody_argv(req, inputs["antigen"], inputs["framework"], job_dir, settings)


def _monomer_build(req, inputs, job_dir, settings):
    return monomer_argv(req, job_dir, settings)


def _scaffolding_build(req, inputs, job_dir, settings):
    return scaffolding_argv(req, inputs["motif_csv"], job_dir, settings)


ENDPOINTS = {
    "binder": CLIEndpoint(
        name="binder",
        help="PPI binder design against a target PDB",
        request_model=BinderRequest,
        build_argv=_binder_build,
        inputs={"target": ("Target PDB file", True)},
    ),
    "antibody": CLIEndpoint(
        name="antibody",
        help="Antibody (heavy + light) CDR design",
        request_model=AntibodyRequest,
        build_argv=_antibody_build,
        inputs={
            "antigen": ("Antigen PDB file", True),
            "framework": ("Antibody framework PDB file", True),
        },
    ),
    "nanobody": CLIEndpoint(
        name="nanobody",
        help="VHH nanobody CDR design",
        request_model=NanobodyRequest,
        build_argv=_nanobody_build,
        inputs={
            "antigen": ("Antigen PDB file", True),
            "framework": ("Nanobody framework PDB file", True),
        },
    ),
    "monomer": CLIEndpoint(
        name="monomer",
        help="Unconditional monomer generation",
        request_model=MonomerRequest,
        build_argv=_monomer_build,
    ),
    "scaffolding": CLIEndpoint(
        name="scaffolding",
        help="Motif scaffolding from CSV + motif PDBs",
        request_model=ScaffoldingRequest,
        build_argv=_scaffolding_build,
        inputs={"motif_csv": ("Motif metadata CSV file", True)},
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"binder", "antibody", "nanobody", "monomer", "scaffolding"}


def test_binder_endpoint_fields():
    ep = ENDPOINTS["binder"]
    assert ep.request_model is BinderRequest
    assert ep.inputs["target"][1] is True


def test_antibody_endpoint_inputs():
    ep = ENDPOINTS["antibody"]
    assert ep.request_model is AntibodyRequest
    assert "antigen" in ep.inputs
    assert "framework" in ep.inputs


def test_nanobody_endpoint_inputs():
    ep = ENDPOINTS["nanobody"]
    assert ep.request_model is NanobodyRequest
    assert "antigen" in ep.inputs
    assert "framework" in ep.inputs


def test_monomer_has_no_inputs():
    assert ENDPOINTS["monomer"].inputs == {}
    assert ENDPOINTS["monomer"].request_model is MonomerRequest


def test_scaffolding_endpoint_inputs():
    ep = ENDPOINTS["scaffolding"]
    assert ep.request_model is ScaffoldingRequest
    assert ep.inputs["motif_csv"][1] is True


# ---- Build_argv callbacks ----


def test_binder_build_argv(settings, tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    target = tmp_path / "target.pdb"
    target.write_text("HEADER")

    argv = _binder_build(
        BinderRequest(
            target_chain="B",
            binder_chain="A",
            specified_hotspots="B119,B141",
            samples_per_target=8,
        ),
        {"target": target},
        job_dir,
        settings,
    )
    assert any("sample_binder.py" in str(a) for a in argv)
    assert "--input_pdb" in argv
    assert "--target_chain" in argv
    assert "--specified_hotspots" in argv
    assert "--samples_per_target" in argv


def test_antibody_build_argv(settings, tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    antigen = tmp_path / "antigen.pdb"
    framework = tmp_path / "framework.pdb"
    antigen.write_text("ATOM")
    framework.write_text("ATOM")

    argv = _antibody_build(
        AntibodyRequest(antigen_chain="C", heavy_chain="A", light_chain="B"),
        {"antigen": antigen, "framework": framework},
        job_dir,
        settings,
    )
    assert any("sample_antibody_nanobody.py" in str(a) for a in argv)
    assert "--light_chain" in argv


def test_nanobody_build_argv(settings, tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    antigen = tmp_path / "antigen.pdb"
    framework = tmp_path / "framework.pdb"
    antigen.write_text("ATOM")
    framework.write_text("ATOM")

    argv = _nanobody_build(
        NanobodyRequest(antigen_chain="C", heavy_chain="A"),
        {"antigen": antigen, "framework": framework},
        job_dir,
        settings,
    )
    assert "--light_chain" not in argv
    assert any("nanobody.ckpt" in str(a) for a in argv)


def test_monomer_build_argv(settings, tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    argv = _monomer_build(
        MonomerRequest(length_subset=[50, 100], samples_per_target=3),
        {},
        job_dir,
        settings,
    )
    assert "--length_subset" in argv
    idx = argv.index("--length_subset")
    assert json.loads(argv[idx + 1]) == [50, 100]


def test_scaffolding_build_argv(settings, tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    csv = tmp_path / "motif.csv"
    csv.write_text("target,length,contig,motif_path\n01_1LDB,125,0-100,01_1LDB.pdb\n")

    argv = _scaffolding_build(
        ScaffoldingRequest(motif_names=["01_1LDB"], samples_per_target=5),
        {"motif_csv": csv},
        job_dir,
        settings,
    )
    assert any("monomer.ckpt" in str(a) for a in argv)
    assert "--motif_names" in argv


# ---- End-to-end create_cli ----


def test_cli_binder_success(settings, tmp_path):
    adapter = PPIFlowAdapter(settings=settings)
    target = tmp_path / "target.pdb"
    target.write_text("HEADER")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "binder",
        "--target", str(target),
        "--target-chain", "B",
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, settings, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_monomer_json_output(settings, tmp_path, capsys):
    adapter = PPIFlowAdapter(settings=settings)
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "monomer",
        "--json", "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, settings, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"


def test_cli_no_subcommand_exits_2(settings):
    adapter = PPIFlowAdapter(settings=settings)
    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, settings, ENDPOINTS)
