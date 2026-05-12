"""Genie3Adapter — detect_outputs (recursive pdb hunt) + manifest_extras."""

from __future__ import annotations

from pathlib import Path

import pytest

from server.adapter import Genie3Adapter
from server.settings import Genie3Settings


@pytest.fixture
def adapter(tmp_path: Path) -> Genie3Adapter:
    return Genie3Adapter(
        settings=Genie3Settings(jobs_base_dir=tmp_path, root=tmp_path / "root"),
    )


def test_detect_outputs_finds_nested_pdbs(adapter: Genie3Adapter, tmp_path: Path) -> None:
    job_dir = tmp_path / "j"
    nested = job_dir / "output" / "binder" / "pdbs"
    nested.mkdir(parents=True)
    (nested / "0.pdb").write_text("ATOM\n")
    assert adapter.detect_outputs(job_dir) is True


def test_detect_outputs_ignores_empty_pdbs(adapter: Genie3Adapter, tmp_path: Path) -> None:
    job_dir = tmp_path / "j"
    nested = job_dir / "output" / "exp" / "pdbs"
    nested.mkdir(parents=True)
    (nested / "0.pdb").touch()  # 0 bytes
    assert adapter.detect_outputs(job_dir) is False


def test_detect_outputs_false_for_non_pdb_outputs(adapter: Genie3Adapter, tmp_path: Path) -> None:
    """genie3-server's detect must NOT trip on incidental files like logs or YAML."""
    job_dir = tmp_path / "j"
    (job_dir / "output").mkdir(parents=True)
    (job_dir / "output" / "experiment.yaml").write_text("name: x\n")
    assert adapter.detect_outputs(job_dir) is False


def test_subprocess_cwd_is_project_root(adapter: Genie3Adapter, tmp_path: Path) -> None:
    assert adapter.subprocess_cwd() == (tmp_path / "root")


def test_infer_job_counts_pdbs(adapter: Genie3Adapter, tmp_path: Path) -> None:
    job_dir = tmp_path / "j-good"
    pdbs = job_dir / "output" / "unc" / "pdbs"
    pdbs.mkdir(parents=True)
    for i in range(3):
        (pdbs / f"{i}.pdb").write_text("ATOM\n")
    info = adapter.infer_job_from_dir(job_dir)
    assert info.status.value == "completed"
    assert "3 PDB" in (info.message or "")


def test_infer_job_failed_when_no_outputs(adapter: Genie3Adapter, tmp_path: Path) -> None:
    job_dir = tmp_path / "j-bad"
    (job_dir / "output").mkdir(parents=True)
    info = adapter.infer_job_from_dir(job_dir)
    assert info.status.value == "failed"


def test_manifest_extras_documents_endpoints(adapter: Genie3Adapter) -> None:
    extras = adapter.manifest_extras()
    summary = extras["endpoints_summary"]
    assert "/api/generate/unconditional" in summary
    assert "/api/generate/motif" in summary
    assert "/api/generate/binder" in summary
    assert "/api/generate" in summary
    # Output convention.
    assert "*.pdb" in extras["tool_outputs"]["all_modes"]
    # Config tips include the cond_strategy gotcha we hit during validation.
    assert "cond_strategy" in extras["config_tips"]
