"""RFdiffusion2Adapter — output detection + restart recovery + env injection."""

from __future__ import annotations

from pathlib import Path

import pytest

from server.adapter import RFdiffusion2Adapter
from server.settings import RFdiffusion2Settings
from server.tools import OUTPUT_STEM


@pytest.fixture
def adapter(tmp_path: Path) -> RFdiffusion2Adapter:
    return RFdiffusion2Adapter(
        settings=RFdiffusion2Settings(
            jobs_base_dir=tmp_path,
            root=tmp_path / "root",
            models_dir=tmp_path / "models",
            pythonpath=tmp_path / "root",
        )
    )


def _write(p: Path, content: bytes = b"PDB\n") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def test_detect_outputs_true_when_any_design_pdb_present(
    adapter: RFdiffusion2Adapter, tmp_path: Path
) -> None:
    job_dir = tmp_path / "j-ok"
    _write(job_dir / "output" / f"{OUTPUT_STEM}_0.pdb")
    assert adapter.detect_outputs(job_dir) is True


def test_detect_outputs_false_for_empty_or_unrelated(
    adapter: RFdiffusion2Adapter, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    (empty / "output").mkdir(parents=True)
    assert adapter.detect_outputs(empty) is False

    # Random non-design file in output/ doesn't satisfy the contract.
    junk = tmp_path / "junk"
    _write(junk / "output" / "other.pdb")
    assert adapter.detect_outputs(junk) is False


def test_infer_job_completed_with_count(
    adapter: RFdiffusion2Adapter, tmp_path: Path
) -> None:
    j = tmp_path / "j-multi"
    _write(j / "output" / f"{OUTPUT_STEM}_0.pdb")
    _write(j / "output" / f"{OUTPUT_STEM}_1.pdb")
    _write(j / "output" / f"{OUTPUT_STEM}_2.pdb")
    info = adapter.infer_job_from_dir(j)
    assert info.status.value == "completed"
    assert "3 PDB" in (info.message or "")


def test_infer_job_failed_when_empty(
    adapter: RFdiffusion2Adapter, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    (empty / "output").mkdir(parents=True)
    info = adapter.infer_job_from_dir(empty)
    assert info.status.value == "failed"


def test_subprocess_env_injects_pythonpath(
    adapter: RFdiffusion2Adapter, tmp_path: Path
) -> None:
    """rfdiffusion2 imports rely on PYTHONPATH being set (no editable install)."""
    env = adapter.subprocess_env()
    assert env.get("PYTHONPATH") == str(tmp_path / "root")


def test_subprocess_cwd_is_repo_root(
    adapter: RFdiffusion2Adapter, tmp_path: Path
) -> None:
    assert adapter.subprocess_cwd() == tmp_path / "root"


def test_manifest_lists_all_endpoints(adapter: RFdiffusion2Adapter) -> None:
    extras = adapter.manifest_extras()
    endpoints = extras["endpoints_summary"]
    assert "/api/generate/active_site" in endpoints
    assert "/api/generate/small_molecule_binder" in endpoints
    assert "/api/generate" in endpoints
    # Models surface
    assert "rfd_140" in extras["models"]["checkpoints"]
    assert "rfd_173" in extras["models"]["checkpoints"]
