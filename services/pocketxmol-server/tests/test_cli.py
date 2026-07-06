"""CLI batch-mode tests for pocketxmol-server.

Covers endpoint registration, build_argv shape, and end-to-end create_cli
(with SubprocessRunner mocked).  Real inference is exercised in test_fc*.py.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioagent_service.cli import CLIEndpoint, create_cli

from server.adapter import PocketXMolAdapter
from server.configs import (
    build_dock_config,
    build_linking_config,
    build_model_config,
    build_optimize_config,
    build_pepdesign_config,
    build_sbdd_config,
)
from server.models import (
    DockRequest,
    LinkingRequest,
    OptimizeRequest,
    PepDesignMode,
    PepDesignRequest,
    SbddMode,
    SbddRequest,
)
from server.settings import PocketXMolSettings
from server.tools import sample_argv


class _Off(PocketXMolSettings):
    model_config = SettingsConfigDict(
        env_prefix="POCKETXMOL_TEST_", env_file=None, extra="ignore",
    )


def _write_pair(cfg: dict, job_dir, settings):
    import yaml
    (job_dir / "input").mkdir(parents=True, exist_ok=True)
    task = job_dir / "input" / "task_config.yml"
    task.write_text(yaml.safe_dump(cfg, sort_keys=False))
    model = job_dir / "input" / "model_config.yml"
    model.write_text(yaml.safe_dump(build_model_config(settings), sort_keys=False))
    return task, model


def _dock_build(req, inputs, job_dir, settings):
    output_dir = job_dir / "output"
    cfg = build_dock_config(
        req=req, protein_path=inputs["protein"],
        ligand_path=inputs.get("ligand"),
        ref_ligand_path=inputs.get("ref_ligand"),
        output_dir=output_dir,
    )
    task, model = _write_pair(cfg, job_dir, settings)
    return sample_argv(
        task_config_path=task, model_config_path=model,
        output_dir=output_dir, settings=settings, batch_size=req.batch_size,
    )


def _sbdd_build(req, inputs, job_dir, settings):
    output_dir = job_dir / "output"
    cfg = build_sbdd_config(
        req=req, protein_path=inputs["protein"], output_dir=output_dir,
    )
    task, model = _write_pair(cfg, job_dir, settings)
    return sample_argv(
        task_config_path=task, model_config_path=model,
        output_dir=output_dir, settings=settings, batch_size=req.batch_size,
    )


def _linking_build(req, inputs, job_dir, settings):
    output_dir = job_dir / "output"
    cfg = build_linking_config(
        req=req, protein_path=inputs["protein"],
        input_ligand_path=inputs["input_ligand"], output_dir=output_dir,
    )
    task, model = _write_pair(cfg, job_dir, settings)
    return sample_argv(
        task_config_path=task, model_config_path=model,
        output_dir=output_dir, settings=settings, batch_size=req.batch_size,
    )


def _optimize_build(req, inputs, job_dir, settings):
    output_dir = job_dir / "output"
    cfg = build_optimize_config(
        req=req, protein_path=inputs["protein"],
        input_ligand_path=inputs["input_ligand"], output_dir=output_dir,
    )
    task, model = _write_pair(cfg, job_dir, settings)
    return sample_argv(
        task_config_path=task, model_config_path=model,
        output_dir=output_dir, settings=settings, batch_size=req.batch_size,
    )


def _pepdesign_build(req, inputs, job_dir, settings):
    output_dir = job_dir / "output"
    cfg = build_pepdesign_config(
        req=req, protein_path=inputs["protein"],
        input_peptide_path=inputs.get("input_peptide"),
        ref_ligand_path=inputs.get("ref_ligand"),
        output_dir=output_dir,
    )
    task, model = _write_pair(cfg, job_dir, settings)
    return sample_argv(
        task_config_path=task, model_config_path=model,
        output_dir=output_dir, settings=settings, batch_size=req.batch_size,
    )


ENDPOINTS = {
    "dock": CLIEndpoint(
        name="dock",
        help="Molecular docking",
        request_model=DockRequest,
        build_argv=_dock_build,
        inputs={
            "protein": ("Protein PDB", True),
            "ligand": ("Ligand SDF/PDB", False),
            "ref_ligand": ("Reference ligand for pocket extraction", False),
        },
    ),
    "sbdd": CLIEndpoint(
        name="sbdd",
        help="De novo SBDD",
        request_model=SbddRequest,
        build_argv=_sbdd_build,
        inputs={"protein": ("Protein PDB", True)},
    ),
    "linking": CLIEndpoint(
        name="linking",
        help="Fragment linking / growing",
        request_model=LinkingRequest,
        build_argv=_linking_build,
        inputs={
            "protein": ("Protein PDB", True),
            "input_ligand": ("Input SDF with fragments", True),
        },
    ),
    "optimize": CLIEndpoint(
        name="optimize",
        help="Molecular optimization",
        request_model=OptimizeRequest,
        build_argv=_optimize_build,
        inputs={
            "protein": ("Protein PDB", True),
            "input_ligand": ("Input SDF to optimize", True),
        },
    ),
    "pepdesign": CLIEndpoint(
        name="pepdesign",
        help="Peptide design",
        request_model=PepDesignRequest,
        build_argv=_pepdesign_build,
        inputs={
            "protein": ("Protein PDB", True),
            "input_peptide": ("Input peptide PDB", False),
            "ref_ligand": ("Reference ligand for pocket extraction", False),
        },
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {
        "dock", "sbdd", "linking", "optimize", "pepdesign",
    }


def test_sbdd_endpoint_required_fields():
    ep = ENDPOINTS["sbdd"]
    assert ep.request_model is SbddRequest
    assert ep.inputs["protein"][1] is True


def test_linking_endpoint_both_required():
    ep = ENDPOINTS["linking"]
    assert ep.inputs["protein"][1] is True
    assert ep.inputs["input_ligand"][1] is True


def test_pepdesign_input_peptide_optional():
    ep = ENDPOINTS["pepdesign"]
    assert ep.inputs["input_peptide"][1] is False


# ---- build_argv shape ----


def test_dock_build_argv_shape(tmp_path):
    s = _Off(python="/bin/true", root=tmp_path / "opt")
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    protein = tmp_path / "p.pdb"
    protein.write_text("ATOM")

    argv = _dock_build(
        DockRequest(num_samples=3, smiles="c1ccccc1"),
        {"protein": protein},
        job_dir, s,
    )
    assert argv[0] == "/bin/true"
    assert argv[1].endswith("sample_use.py")
    # --config_task points to the on-disk task_config.yml
    assert "--config_task" in argv
    task_idx = argv.index("--config_task") + 1
    assert (job_dir / "input" / "task_config.yml").exists()
    assert argv[task_idx] == str(job_dir / "input" / "task_config.yml")
    # --config_model likewise
    assert "--config_model" in argv
    assert (job_dir / "input" / "model_config.yml").exists()
    # --batch_size present
    assert "--batch_size" in argv


def test_sbdd_build_argv_uses_sample_script(tmp_path):
    s = _Off(python="/bin/true")
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    protein = tmp_path / "p.pdb"
    protein.write_text("ATOM")

    argv = _sbdd_build(
        SbddRequest(pocket_coord=[1.0, 2.0, 3.0], mode=SbddMode.simple),
        {"protein": protein},
        job_dir, s,
    )
    assert argv[1].endswith("sample_use.py")


def test_linking_build_argv_with_fragments(tmp_path):
    s = _Off(python="/bin/true")
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    protein = tmp_path / "p.pdb"
    protein.write_text("ATOM")
    ligand = tmp_path / "l.sdf"
    ligand.write_text("$$$$")

    argv = _linking_build(
        LinkingRequest(fragments=[[0, 1, 2, 3, 4, 5, 6]]),
        {"protein": protein, "input_ligand": ligand},
        job_dir, s,
    )
    assert argv[1].endswith("sample_use.py")


def test_pepdesign_denovo_build_argv(tmp_path):
    s = _Off(python="/bin/true")
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    protein = tmp_path / "p.pdb"
    protein.write_text("ATOM")

    argv = _pepdesign_build(
        PepDesignRequest(mode=PepDesignMode.denovo_linear, pep_length=10),
        {"protein": protein},
        job_dir, s,
    )
    assert argv[1].endswith("sample_use.py")


# ---- End-to-end create_cli ----


def test_cli_dock_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = PocketXMolAdapter(settings=s)

    protein = tmp_path / "p.pdb"
    protein.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "dock",
        "--protein", str(protein),
        "--output-dir", str(output_dir),
        "--params-json", '{"num_samples": 3, "smiles": "c1ccccc1"}',
    ]):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_sbdd_json_output(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = PocketXMolAdapter(settings=s)

    protein = tmp_path / "p.pdb"
    protein.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "sbdd",
        "--protein", str(protein),
        "--output-dir", str(output_dir),
        "--json",
        "--params-json", '{"pocket_coord": [1.0, 2.0, 3.0], "num_samples": 3}',
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


def test_cli_linking_needs_both_inputs(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs", python="/bin/true")
    adapter = PocketXMolAdapter(settings=s)

    protein = tmp_path / "p.pdb"
    protein.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    # Missing --input-ligand → argparse error → SystemExit 2.
    with patch.object(sys, "argv", [
        "prog", "linking",
        "--protein", str(protein),
        "--output-dir", str(output_dir),
        "--params-json", '{"fragments": [[0,1,2]]}',
    ]):
        with pytest.raises(SystemExit) as exc_info:
            create_cli(adapter, s, ENDPOINTS, version="0.0.1")
        assert exc_info.value.code == 2


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = PocketXMolAdapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
