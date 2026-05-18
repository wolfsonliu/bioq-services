"""RFdiffusionAdapter — output detection + restart recovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from server.adapter import RFdiffusionAdapter
from server.settings import RFdiffusionSettings
from server.tools import OUTPUT_STEM


@pytest.fixture
def adapter(tmp_path: Path) -> RFdiffusionAdapter:
    return RFdiffusionAdapter(
        settings=RFdiffusionSettings(
            jobs_base_dir=tmp_path,
            root=tmp_path / "root",
            models_dir=tmp_path / "models",
        )
    )


def _write(p: Path, content: bytes = b"PDB\n") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def test_detect_outputs_true_when_any_design_pdb_present(
    adapter: RFdiffusionAdapter, tmp_path: Path
) -> None:
    job_dir = tmp_path / "j-ok"
    _write(job_dir / "output" / f"{OUTPUT_STEM}_0.pdb")
    assert adapter.detect_outputs(job_dir) is True


def test_detect_outputs_false_for_empty_or_unrelated(
    adapter: RFdiffusionAdapter, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    (empty / "output").mkdir(parents=True)
    assert adapter.detect_outputs(empty) is False

    # Random non-design file in output/ doesn't satisfy the contract.
    junk = tmp_path / "junk"
    _write(junk / "output" / "other.pdb")
    assert adapter.detect_outputs(junk) is False


def test_infer_job_completed_with_count(
    adapter: RFdiffusionAdapter, tmp_path: Path
) -> None:
    j = tmp_path / "j-multi"
    _write(j / "output" / f"{OUTPUT_STEM}_0.pdb")
    _write(j / "output" / f"{OUTPUT_STEM}_1.pdb")
    _write(j / "output" / f"{OUTPUT_STEM}_2.pdb")
    info = adapter.infer_job_from_dir(j)
    assert info.status.value == "completed"
    assert "3 PDB" in (info.message or "")


def test_infer_job_failed_when_empty(
    adapter: RFdiffusionAdapter, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    (empty / "output").mkdir(parents=True)
    info = adapter.infer_job_from_dir(empty)
    assert info.status.value == "failed"


def test_manifest_lists_all_endpoints(adapter: RFdiffusionAdapter) -> None:
    extras = adapter.manifest_extras()
    endpoints = extras["endpoints_summary"]
    assert "/api/generate/unconditional" in endpoints
    assert "/api/generate/motif" in endpoints
    assert "/api/generate/binder" in endpoints
    assert "/api/generate/symmetry" in endpoints
    assert "/api/generate" in endpoints
    # Sanity: model list is published.
    assert "base" in extras["models"]["checkpoints"]
    assert "active_site" in extras["models"]["checkpoints"]
