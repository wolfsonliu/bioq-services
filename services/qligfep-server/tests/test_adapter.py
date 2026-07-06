"""Adapter detect_outputs per-label dispatch tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from server.adapter import QligfepAdapter
from server.settings import QligfepSettings


class _Off(QligfepSettings):
    model_config = SettingsConfigDict(env_prefix="QLIGFEP_TEST_", env_file=None, extra="ignore")


@pytest.fixture
def adapter(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    return QligfepAdapter(settings=s)


def _make_job(tmp_path, label: str) -> Path:
    job = tmp_path / "jobs" / "job1"
    (job / "output").mkdir(parents=True)
    (job / "manifest.json").write_text(json.dumps({"label": label}))
    return job


def test_detect_ligprep_missing_returns_false(adapter, tmp_path):
    job = _make_job(tmp_path, "ligprep")
    assert adapter.detect_outputs(job) is False


def test_detect_ligprep_all_present(adapter, tmp_path):
    job = _make_job(tmp_path, "ligprep")
    for ext in ("lib", "prm", "pdb"):
        (job / "output" / f"17.{ext}").write_text("x")
    assert adapter.detect_outputs(job) is True


def test_detect_protprep(adapter, tmp_path):
    job = _make_job(tmp_path, "protprep")
    assert adapter.detect_outputs(job) is False
    (job / "output" / "protein.pdb").write_text("ATOM")
    (job / "output" / "water.pdb").write_text("HETATM")
    assert adapter.detect_outputs(job) is True


def test_detect_cog(adapter, tmp_path):
    job = _make_job(tmp_path, "cog")
    assert adapter.detect_outputs(job) is False
    (job / "output" / "result.json").write_text('{"cx":0,"cy":0,"cz":0}')
    assert adapter.detect_outputs(job) is True


def test_detect_setup_fep(adapter, tmp_path):
    job = _make_job(tmp_path, "setup-ligfep")
    (job / "output" / "1.protein").mkdir()
    (job / "output" / "1.protein" / "FEP_submit.sh").write_text("#!/bin/bash")
    assert adapter.detect_outputs(job) is False  # no FEP*/md_*.inp yet
    fep = job / "output" / "1.protein" / "FEP1"
    fep.mkdir()
    (fep / "md_0500_0500.inp").write_text("x")
    assert adapter.detect_outputs(job) is True


def test_detect_setup_lie(adapter, tmp_path):
    job = _make_job(tmp_path, "setup-lie")
    assert adapter.detect_outputs(job) is False
    (job / "output" / "setup.json").write_text("{}")
    (job / "output" / "md_LIE_bound").mkdir()
    (job / "output" / "md_LIE_bound" / "x.inp").write_text("x")
    assert adapter.detect_outputs(job) is True


def test_detect_run_fep(adapter, tmp_path):
    job = _make_job(tmp_path, "run-fep")
    win = job / "output" / "window_5_rep_0"
    win.mkdir(parents=True)
    assert adapter.detect_outputs(job) is False
    (win / "md_0500_0500.en").write_bytes(b"\x00\x01")
    assert adapter.detect_outputs(job) is True


def test_detect_analyze_fep(adapter, tmp_path):
    job = _make_job(tmp_path, "analyze-fep")
    assert adapter.detect_outputs(job) is False
    (job / "output" / "results.txt").write_text("DDG = -3.2 kcal/mol")
    assert adapter.detect_outputs(job) is True


def test_detect_unknown_label(adapter, tmp_path):
    job = _make_job(tmp_path, "no-such-label")
    assert adapter.detect_outputs(job) is False


def test_manifest_extras_lists_all_endpoints(adapter):
    extras = adapter.manifest_extras()
    tools = extras["tool_outputs"]
    for name in ("ligprep", "protprep", "cog", "setup-ligfep",
                 "setup-resfep", "setup-lie", "run-fep",
                 "analyze-fep", "analyze-lie"):
        assert name in tools, f"missing {name} in tool_outputs"


def test_endpoint_examples_curls_exist(adapter):
    examples = adapter.endpoint_examples()
    for path in ("/api/ligprep", "/api/protprep", "/api/setup-ligfep"):
        assert path in examples and examples[path]
