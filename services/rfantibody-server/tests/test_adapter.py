"""RFantibodyAdapter — output detection + restart recovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from server.adapter import RFantibodyAdapter
from server.settings import RFantibodySettings
from server.tools import PROTEINMPNN_OUTPUT, RF2_OUTPUT, RFDIFFUSION_OUTPUT


@pytest.fixture
def adapter(tmp_path: Path) -> RFantibodyAdapter:
    return RFantibodyAdapter(
        settings=RFantibodySettings(
            jobs_base_dir=tmp_path,
            root=tmp_path / "root",
            weights_dir=tmp_path / "w",
            scripts_dir=tmp_path / "s",
        )
    )


def _write(p: Path, content: bytes = b"x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def test_detect_outputs_true_for_any_known_artifact(
    adapter: RFantibodyAdapter, tmp_path: Path
) -> None:
    for fname in (RFDIFFUSION_OUTPUT, PROTEINMPNN_OUTPUT, RF2_OUTPUT):
        job_dir = tmp_path / f"job-{fname}"
        _write(job_dir / "output" / fname)
        assert adapter.detect_outputs(job_dir) is True


def test_detect_outputs_false_for_empty_or_unrelated(
    adapter: RFantibodyAdapter, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    (empty / "output").mkdir(parents=True)
    assert adapter.detect_outputs(empty) is False

    junk = tmp_path / "junk"
    _write(junk / "output" / "random.txt")
    assert adapter.detect_outputs(junk) is False


def test_infer_job_promotes_to_latest_completed_step(
    adapter: RFantibodyAdapter, tmp_path: Path
) -> None:
    # Only rfdiffusion ran → recovered as completed at the rfdiffusion step.
    j = tmp_path / "j-rfd"
    _write(j / "output" / RFDIFFUSION_OUTPUT)
    info = adapter.infer_job_from_dir(j)
    assert info.status.value == "completed"
    assert info.progress == "rfdiffusion"

    # All three present → progress is the furthest step (rf2).
    j2 = tmp_path / "j-all"
    _write(j2 / "output" / RFDIFFUSION_OUTPUT)
    _write(j2 / "output" / PROTEINMPNN_OUTPUT)
    _write(j2 / "output" / RF2_OUTPUT)
    info2 = adapter.infer_job_from_dir(j2)
    assert info2.progress == "rf2"


def test_infer_job_failed_when_empty(
    adapter: RFantibodyAdapter, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    (empty / "output").mkdir(parents=True)
    info = adapter.infer_job_from_dir(empty)
    assert info.status.value == "failed"
