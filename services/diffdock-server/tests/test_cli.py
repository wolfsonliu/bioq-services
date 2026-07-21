"""Offline CLI batch-mode tests for diffdock-server.

Covers ``cli_impl.dock_build`` (the argv builder) and the endpoint
registration.  ``__main__`` is intentionally NOT imported — that module
invokes ``create_cli`` at import time which would parse pytest's argv.
Real end-to-end inference is exercised in ``test_fc*.py``.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioq_service.cli import CLIEndpoint, create_cli

from server.adapter import DiffdockAdapter
from server.cli_impl import build_endpoints, dock_build
from server.models import DockRequest
from server.settings import DiffdockSettings


class _Off(DiffdockSettings):
    model_config = SettingsConfigDict(
        env_prefix="DIFFDOCK_TEST_", env_file=None, extra="ignore",
    )


# ----- Endpoint registration -----


def test_build_endpoints_returns_dock():
    endpoints = build_endpoints()
    assert set(endpoints.keys()) == {"dock"}
    ep = endpoints["dock"]
    assert isinstance(ep, CLIEndpoint)
    assert ep.name == "dock"
    assert ep.request_model is DockRequest
    # Both inputs are optional (three-way mutex handled inside build_argv)
    assert ep.inputs["protein"][1] is False
    assert ep.inputs["ligand"][1] is False


# ----- dock_build (CLI-level argv builder) -----


def test_dock_build_pdb_and_sdf(tmp_path):
    s = _Off(python="/bin/true", root=tmp_path / "opt")
    job_dir = tmp_path / "j"
    (job_dir / "input").mkdir(parents=True)
    target = tmp_path / "target.pdb"
    target.write_text("ATOM  ...")
    lig = tmp_path / "lig.sdf"
    lig.write_text("$$$$")

    argv = dock_build(
        req=DockRequest(complex_name="cli_pdb_sdf"),
        inputs={"protein": target, "ligand": lig},
        job_dir=job_dir,
        settings=s,
    )
    assert argv[0] == "/bin/true"
    assert argv[1].endswith("run_inference.py")
    assert "--protein_path" in argv
    assert str(target) in argv
    assert "--ligand" in argv
    assert str(lig) in argv
    assert "--complex_name" in argv
    assert "cli_pdb_sdf" in argv


def test_dock_build_sequence_and_smiles(tmp_path):
    s = _Off()
    req = DockRequest(
        protein_sequence="MKW" * 30,
        ligand_description="CCO",
        complex_name="from_text",
    )
    argv = dock_build(
        req=req, inputs={}, job_dir=tmp_path / "j", settings=s,
    )
    assert "--protein_sequence" in argv
    assert "MKW" * 30 in argv
    assert "--protein_path" not in argv
    assert "CCO" in argv


def test_dock_build_rejects_no_protein(tmp_path):
    """No protein file AND no protein_sequence in params → SystemExit."""
    with pytest.raises(SystemExit, match="protein"):
        dock_build(
            req=DockRequest(ligand_description="CCO"),
            inputs={},
            job_dir=tmp_path / "j",
            settings=_Off(),
        )


def test_dock_build_rejects_two_ligand_forms(tmp_path):
    with pytest.raises(SystemExit, match="ligand"):
        dock_build(
            req=DockRequest(
                protein_sequence="MKW" * 30,
                ligand_description="CCO",
            ),
            inputs={"ligand": tmp_path / "l.sdf"},
            job_dir=tmp_path / "j",
            settings=_Off(),
        )


def test_dock_build_rejects_uri_in_cli_mode(tmp_path):
    """ligand_uri is HTTP-only; CLI mode must give a concrete file/string."""
    with pytest.raises(SystemExit, match="only supported over HTTP"):
        dock_build(
            req=DockRequest(
                protein_sequence="MKW" * 30,
                ligand_uri="oss://bucket/lig.sdf",
            ),
            inputs={},
            job_dir=tmp_path / "j",
            settings=_Off(),
        )


# ----- run_inference (wrapper) unit tests -----


def test_run_inference_postprocess_writes_confidence_json(tmp_path):
    """run_inference.postprocess() scans rank<r>_confidence<c>.sdf → JSON."""
    from server.run_inference import postprocess

    complex_dir = tmp_path / "complex_x"
    complex_dir.mkdir()
    (complex_dir / "rank1.sdf").write_text("top")
    (complex_dir / "rank1_confidence1.42.sdf").write_text("1")
    (complex_dir / "rank2_confidence0.87.sdf").write_text("2")
    (complex_dir / "rank3_confidence-0.15.sdf").write_text("3")

    dst = postprocess(complex_dir)
    assert dst is not None
    import json
    entries = json.loads(dst.read_text())
    assert len(entries) == 3
    assert entries[0]["rank"] == 1
    assert entries[0]["confidence"] == pytest.approx(1.42)
    assert entries[2]["rank"] == 3
    assert entries[2]["confidence"] == pytest.approx(-0.15)


def test_run_inference_postprocess_empty_dir_returns_none(tmp_path):
    from server.run_inference import postprocess

    empty = tmp_path / "nope"
    empty.mkdir()
    assert postprocess(empty) is None


def test_run_inference_build_upstream_argv_pdb(tmp_path):
    """build_upstream_argv produces the argv shape upstream expects."""
    from argparse import Namespace

    from server.run_inference import build_upstream_argv

    args = Namespace(
        protein_path=tmp_path / "p.pdb",
        protein_sequence=None,
        ligand="/tmp/l.sdf",
        complex_name="test",
        out_dir=tmp_path / "out",
        samples_per_complex=10,
        inference_steps=20,
        actual_steps=19,
        batch_size=10,
        no_final_step_noise=True,
        save_visualisation=False,
        seed=0,
        model_dir=tmp_path / "score",
        confidence_model_dir=tmp_path / "conf",
        config=tmp_path / "cfg.yml",
        torchhub_dir=tmp_path / "esm",
    )
    argv = build_upstream_argv(args)
    assert "--config" in argv
    assert "--protein_path" in argv
    assert "--protein_sequence" not in argv
    assert "--ligand_description" in argv
    assert "--save_visualisation" not in argv


def test_run_inference_build_upstream_argv_sequence(tmp_path):
    from argparse import Namespace

    from server.run_inference import build_upstream_argv

    args = Namespace(
        protein_path=None,
        protein_sequence="MKW" * 30,
        ligand="CCO",
        complex_name="test",
        out_dir=tmp_path / "out",
        samples_per_complex=5,
        inference_steps=10,
        actual_steps=10,
        batch_size=5,
        no_final_step_noise=True,
        save_visualisation=True,
        seed=42,
        model_dir=tmp_path / "score",
        confidence_model_dir=tmp_path / "conf",
        config=tmp_path / "cfg.yml",
        torchhub_dir=tmp_path / "esm",
    )
    argv = build_upstream_argv(args)
    assert "--protein_sequence" in argv
    assert "MKW" * 30 in argv
    assert "--protein_path" not in argv
    assert "--save_visualisation" in argv


def test_run_inference_argv_parse_missing_protein_errors(monkeypatch, tmp_path):
    """--protein_path and --protein_sequence both missing → argparse error."""
    from server.run_inference import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_inference",
            "--ligand", "CCO",
            "--out_dir", str(tmp_path),
            "--model_dir", str(tmp_path),
            "--confidence_model_dir", str(tmp_path),
            "--config", str(tmp_path / "cfg.yml"),
            "--torchhub_dir", str(tmp_path / "esm"),
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


# ----- End-to-end create_cli (mocked subprocess) -----


def test_cli_dock_success_with_files(tmp_path):
    """`python -m server dock --protein X --ligand Y` runs the pipeline."""
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = DiffdockAdapter(settings=s)

    target = tmp_path / "p.pdb"
    target.write_text("ATOM  ...")
    lig = tmp_path / "l.sdf"
    lig.write_text("$$$$")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "dock",
        "--protein", str(target),
        "--ligand", str(lig),
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, build_endpoints(), version="0.0.1")
                assert exc_info.value.code == 0
