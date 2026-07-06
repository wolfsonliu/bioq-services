"""CLI batch-mode smoke tests — mock SubprocessRunner, patch sys.argv."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioagent_service.cli import create_cli

from server.adapter import QligfepAdapter
from server.settings import QligfepSettings


class _Off(QligfepSettings):
    model_config = SettingsConfigDict(env_prefix="QLIGFEP_TEST_", env_file=None, extra="ignore")


def _endpoints():
    """Import server.__main__ with create_cli() patched to a no-op so the
    module-level call at import time doesn't parse pytest's argv or SystemExit.
    Return the registered dict."""
    import importlib
    sys.modules.pop("server.__main__", None)
    with patch("bioagent_service.cli.create_cli", lambda *a, **k: None):
        importlib.import_module("server.__main__")
    return sys.modules["server.__main__"].endpoints


# ---- registry checks ----

def test_all_9_endpoints_registered():
    eps = _endpoints()
    assert set(eps.keys()) == {
        "ligprep", "protprep", "cog",
        "setup-ligfep", "setup-resfep", "setup-lie",
        "run-fep", "analyze-fep", "analyze-lie",
    }


def test_ligprep_endpoint_fields():
    ep = _endpoints()["ligprep"]
    assert "ligand" in ep.inputs and ep.inputs["ligand"][1] is True


def test_run_fep_endpoint_fields():
    ep = _endpoints()["run-fep"]
    assert "setup_dir" in ep.inputs


# ---- end-to-end (mock SubprocessRunner) ----

def _run_cli(argv, tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = QligfepAdapter(settings=s)
    with patch.object(sys, "argv", argv):
        with patch("bioagent_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc:
                    create_cli(adapter, s, _endpoints(), version="0.0.1")
                return exc.value.code


def test_cli_ligprep_success(tmp_path):
    lig = tmp_path / "17.mol2"; lig.write_text("@<TRIPOS>")
    out = tmp_path / "out"
    code = _run_cli(
        ["prog", "ligprep", "--ligand", str(lig),
         "--params-json", '{"ligand_name": "17"}',
         "--output-dir", str(out)],
        tmp_path,
    )
    assert code == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = QligfepAdapter(settings=s)
    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit) as exc:
            create_cli(adapter, s, _endpoints())
        assert exc.value.code == 2
