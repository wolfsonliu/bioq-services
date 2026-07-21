"""CLI batch-mode tests for lightdock-server.

Covers endpoint registration, the build_argv callback, and end-to-end
create_cli (with the SubprocessRunner + output detection mocked, so no real
lightdock binary is invoked).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioq_service.cli import CLIEndpoint, create_cli
from server.adapter import LightdockAdapter
from server.models import DockRequest
from server.settings import LightdockSettings
from server.tools import dock_argv

DATA_DIR = Path(__file__).resolve().parent / "data"
RECEPTOR = DATA_DIR / "receptor.pdb"
LIGAND = DATA_DIR / "ligand.pdb"


class _Off(LightdockSettings):
    model_config = SettingsConfigDict(
        env_prefix="LIGHTDOCK_TEST_", env_file=None, extra="ignore",
    )


# Mirror of server/__main__.py endpoints (importing __main__ would run
# create_cli at import time and parse sys.argv).
def _dock_build(req, inputs, job_dir, settings):
    return dock_argv(
        req,
        job_dir=job_dir,
        receptor_path=inputs["receptor"],
        ligand_path=inputs["ligand"],
        restraints_path=inputs.get("restraints"),
        settings=settings,
    )


ENDPOINTS = {
    "dock": CLIEndpoint(
        name="dock",
        help="Run the full LightDock GSO docking protocol",
        request_model=DockRequest,
        build_argv=_dock_build,
        inputs={
            "receptor": ("Receptor PDB file", True),
            "ligand": ("Ligand PDB file", True),
            "restraints": ("Optional LightDock restraints file", False),
        },
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"dock"}
    ep = ENDPOINTS["dock"]
    assert ep.inputs["receptor"][1] is True
    assert ep.inputs["ligand"][1] is True
    assert ep.inputs["restraints"][1] is False


# ---- build_argv callback ----


def test_dock_build_argv(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    inputs = {"receptor": RECEPTOR, "ligand": LIGAND}
    argv = ENDPOINTS["dock"].build_argv(
        DockRequest(swarms=10, steps=20), inputs, job_dir, s,
    )
    assert argv[0] == s.python
    assert s.driver_script in argv
    assert "dock" in argv
    assert argv[argv.index("--swarms") + 1] == "10"
    assert argv[argv.index("--steps") + 1] == "20"
    assert "--restraints" not in argv


def test_dock_build_argv_with_restraints(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    r = tmp_path / "restraints.list"
    r.write_text("R A.35\n")
    inputs = {"receptor": RECEPTOR, "ligand": LIGAND, "restraints": r}
    argv = ENDPOINTS["dock"].build_argv(DockRequest(), inputs, job_dir, s)
    assert argv[argv.index("--restraints") + 1] == str(r)


# ---- direct tools argv (parity with the callback) ----


def test_dock_argv_direct(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    argv = dock_argv(
        DockRequest(),
        job_dir=job_dir,
        receptor_path=RECEPTOR,
        ligand_path=LIGAND,
        settings=s,
    )
    assert argv[2] == "dock"
    assert str(RECEPTOR) in argv
    assert str(LIGAND) in argv


# ---- End-to-end create_cli ----


def test_cli_dock_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = LightdockAdapter(settings=s)
    out = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "dock",
        "--receptor", str(RECEPTOR), "--ligand", str(LIGAND),
        "--output-dir", str(out), "--swarms", "2", "--steps", "3",
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc.value.code == 0


def test_cli_json_output(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = LightdockAdapter(settings=s)
    out = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "dock",
        "--receptor", str(RECEPTOR), "--ligand", str(LIGAND),
        "--json", "--output-dir", str(out),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc.value.code == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert result["return_code"] == 0


def test_cli_dock_missing_required_input_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = LightdockAdapter(settings=s)
    out = tmp_path / "run" / "output"
    with patch.object(sys, "argv", [
        "prog", "dock", "--receptor", str(RECEPTOR), "--output-dir", str(out),
    ]):
        with pytest.raises(SystemExit) as exc:
            create_cli(adapter, s, ENDPOINTS, version="0.0.1")
        assert exc.value.code == 2


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = LightdockAdapter(settings=s)
    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
