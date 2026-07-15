"""Unit tests for the dock_argv builder (no lightdock binary needed)."""

from __future__ import annotations

from pathlib import Path

from server.models import DockRequest
from server.settings import LightdockSettings
from server.tools import dock_argv


def _settings(tmp_path) -> LightdockSettings:
    return LightdockSettings(
        _env_file=None,
        jobs_base_dir=tmp_path / "jobs",
        python="/opt/lightdock/.venv/bin/python",
        driver_script="/opt/lightdock/server/docking.py",
        bin_dir=Path("/opt/lightdock/.venv/bin"),
    )


def _argv(req, tmp_path, **kw):
    settings = _settings(tmp_path)
    job_dir = tmp_path / "job"
    return dock_argv(
        req,
        job_dir=job_dir,
        receptor_path=tmp_path / "r.pdb",
        ligand_path=tmp_path / "l.pdb",
        settings=settings,
        **kw,
    )


def test_dock_argv_basic(tmp_path):
    argv = _argv(DockRequest(swarms=20, glowworms=100, steps=50, top=5), tmp_path)
    assert argv[0] == "/opt/lightdock/.venv/bin/python"
    assert argv[1] == "/opt/lightdock/server/docking.py"
    assert argv[2] == "dock"
    assert "--receptor" in argv and "--ligand" in argv
    assert "--swarms" in argv and argv[argv.index("--swarms") + 1] == "20"
    assert argv[argv.index("--glowworms") + 1] == "100"
    assert argv[argv.index("--steps") + 1] == "50"
    assert argv[argv.index("--top") + 1] == "5"
    assert argv[argv.index("--scoring") + 1] == "fastdfire"


def test_dock_argv_default_cores_from_settings(tmp_path):
    argv = _argv(DockRequest(), tmp_path)
    assert argv[argv.index("--cores") + 1] == "8"


def test_dock_argv_explicit_cores(tmp_path):
    argv = _argv(DockRequest(cores=4), tmp_path)
    assert argv[argv.index("--cores") + 1] == "4"


def test_dock_argv_flags(tmp_path):
    argv = _argv(DockRequest(use_anm=True, noxt=True, noh=True, now=True), tmp_path)
    for flag in ("--anm", "--noxt", "--noh", "--now"):
        assert flag in argv


def test_dock_argv_no_flags_by_default(tmp_path):
    argv = _argv(DockRequest(), tmp_path)
    for flag in ("--anm", "--noxt", "--noh", "--now"):
        assert flag not in argv


def test_dock_argv_restraints(tmp_path):
    r = tmp_path / "restraints.list"
    r.write_text("R A.35\n")
    argv = _argv(DockRequest(), tmp_path, restraints_path=r)
    assert "--restraints" in argv
    assert argv[argv.index("--restraints") + 1] == str(r)


def test_dock_argv_no_restraints_when_none(tmp_path):
    argv = _argv(DockRequest(), tmp_path)
    assert "--restraints" not in argv


def test_dock_argv_creates_output_and_work_dirs(tmp_path):
    _argv(DockRequest(), tmp_path)
    assert (tmp_path / "job" / "output").is_dir()
    assert (tmp_path / "job" / "work").is_dir()


def test_scoring_function_pattern_rejects_spaces():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DockRequest(scoring_function="bad name")
