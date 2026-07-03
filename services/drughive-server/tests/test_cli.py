"""CLI batch-mode tests for drughive-server.

Covers endpoint registration, build_argv, and end-to-end create_cli
(mocked subprocess).  Real inference is exercised in test_fc*.py.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioagent_service.cli import CLIEndpoint, create_cli

from server.adapter import DrughiveAdapter
from server.configs import (
    build_generate_config,
    build_generate_spatial_config,
    build_optimize_config,
)
from server.models import GenerateRequest, GenerateSpatialRequest, OptimizeRequest
from server.settings import DrughiveSettings
from server.tools import generate_argv, optimize_argv


class _Off(DrughiveSettings):
    model_config = SettingsConfigDict(
        env_prefix="DRUGHIVE_TEST_", env_file=None, extra="ignore",
    )


def _write_cfg(cfg: dict, job_dir):
    import yaml
    (job_dir / "input").mkdir(parents=True, exist_ok=True)
    path = job_dir / "input" / "config.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _generate_build(req, inputs, job_dir, settings):
    cfg = build_generate_config(
        req=req,
        target_path=inputs["target"],
        ligand_path=inputs["ligand"],
        output_dir=job_dir / "output",
        settings=settings,
    )
    cfg_path = _write_cfg(cfg, job_dir)
    return generate_argv(cfg_path=cfg_path, settings=settings)


def _generate_spatial_build(req, inputs, job_dir, settings):
    cfg = build_generate_spatial_config(
        req=req,
        target_path=inputs["target"],
        ligand_path=inputs["ligand"],
        output_dir=job_dir / "output",
        settings=settings,
        substruct_modify_path=inputs.get("substruct_modify"),
    )
    cfg_path = _write_cfg(cfg, job_dir)
    return generate_argv(cfg_path=cfg_path, settings=settings)


def _optimize_build(req, inputs, job_dir, settings):
    cfg = build_optimize_config(
        req=req,
        target_path=inputs["target"],
        ligand_path=inputs["ligand"],
        target_pdbqt_path=inputs.get("target_pdbqt"),
        output_dir=job_dir / "output",
        settings=settings,
    )
    cfg_path = _write_cfg(cfg, job_dir)
    return optimize_argv(cfg_path=cfg_path, settings=settings)


ENDPOINTS = {
    "generate": CLIEndpoint(
        name="generate",
        help="De novo ligand generation",
        request_model=GenerateRequest,
        build_argv=_generate_build,
        inputs={
            "target": ("Pocket PDB", True),
            "ligand": ("Reference ligand SDF", True),
        },
    ),
    "generate_spatial": CLIEndpoint(
        name="generate_spatial",
        help="Scaffold hopping",
        request_model=GenerateSpatialRequest,
        build_argv=_generate_spatial_build,
        inputs={
            "target": ("Pocket PDB", True),
            "ligand": ("Reference ligand SDF", True),
            "substruct_modify": ("Preserved fragment SDF", False),
        },
    ),
    "optimize": CLIEndpoint(
        name="optimize",
        help="Multi-cycle QVina2 optimization",
        request_model=OptimizeRequest,
        build_argv=_optimize_build,
        inputs={
            "target": ("Pocket PDB", True),
            "ligand": ("Reference ligand SDF", True),
            "target_pdbqt": ("Target PDBQT", False),
        },
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"generate", "generate_spatial", "optimize"}


def test_generate_endpoint_fields():
    ep = ENDPOINTS["generate"]
    assert ep.request_model is GenerateRequest
    assert ep.inputs["target"][1] is True
    assert ep.inputs["ligand"][1] is True


def test_generate_spatial_optional_frag():
    ep = ENDPOINTS["generate_spatial"]
    assert ep.inputs["substruct_modify"][1] is False


def test_optimize_optional_pdbqt():
    ep = ENDPOINTS["optimize"]
    assert ep.inputs["target_pdbqt"][1] is False


# ---- build_argv ----


def test_generate_build_argv_shape(tmp_path):
    s = _Off(python="/bin/true", root=tmp_path / "opt")
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    target = tmp_path / "p.pdb"
    target.write_text("ATOM")
    ligand = tmp_path / "l.sdf"
    ligand.write_text("$$$$")

    argv = _generate_build(
        GenerateRequest(n_samples=5, pdb_id="test"),
        {"target": target, "ligand": ligand},
        job_dir, s,
    )
    assert argv[0] == "/bin/true"
    assert argv[1].endswith("generate_molecules.py")
    # config.yml must have been written
    cfg_path = job_dir / "input" / "config.yml"
    assert cfg_path.exists()
    assert argv[2] == str(cfg_path)


def test_optimize_build_argv_uses_optimize_script(tmp_path):
    s = _Off(python="/bin/true")
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    target = tmp_path / "p.pdb"
    target.write_text("ATOM")
    ligand = tmp_path / "l.sdf"
    ligand.write_text("$$$$")
    pdbqt = tmp_path / "p.pdbqt"
    pdbqt.write_text("REMARK")

    argv = _optimize_build(
        OptimizeRequest(n_cycles=2),
        {"target": target, "ligand": ligand, "target_pdbqt": pdbqt},
        job_dir, s,
    )
    assert argv[1].endswith("generate_optimize.py")


# ---- End-to-end create_cli ----


def test_cli_generate_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = DrughiveAdapter(settings=s)

    target = tmp_path / "p.pdb"
    target.write_text("ATOM")
    ligand = tmp_path / "l.sdf"
    ligand.write_text("$$$$")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "generate",
        "--target", str(target),
        "--ligand", str(ligand),
        "--output-dir", str(output_dir),
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_optimize_success_with_pdbqt(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = DrughiveAdapter(settings=s)

    target = tmp_path / "p.pdb"
    target.write_text("ATOM")
    ligand = tmp_path / "l.sdf"
    ligand.write_text("$$$$")
    pdbqt = tmp_path / "p.pdbqt"
    pdbqt.write_text("REMARK")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "optimize",
        "--target", str(target),
        "--ligand", str(ligand),
        "--target-pdbqt", str(pdbqt),
        "--output-dir", str(output_dir),
        "--json",
        "--params-json", '{"n_cycles": 2, "n_samples_initial": 20, "n_samples": 4, "n_best_parents": 2}',
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert result["return_code"] == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = DrughiveAdapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
