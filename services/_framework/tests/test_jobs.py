"""JobStore + filesystem helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from bioagent_service.jobs import (
    JobStore,
    cleanup_job,
    disk_usage_bytes,
    evict_finished_until_under_limit,
    get_job_dir,
)
from bioagent_service.models import JobStatus


def test_create_get_update_round_trip() -> None:
    store = JobStore()
    job = store.create("abc123")
    assert job.job_id == "abc123"
    assert job.status == JobStatus.PENDING
    assert store.get("abc123") == job

    updated = store.update("abc123", status=JobStatus.RUNNING, message="go")
    assert updated.status == JobStatus.RUNNING
    assert updated.message == "go"
    assert store.get("abc123") == updated


def test_create_records_service_and_endpoint() -> None:
    store = JobStore()
    job = store.create("j", service="rfantibody", endpoint="rfdiffusion")
    assert job.service == "rfantibody"
    assert job.endpoint == "rfdiffusion"
    assert store.get("j").service == "rfantibody"


def test_create_service_endpoint_default_none() -> None:
    store = JobStore()
    job = store.create("j")
    assert job.service is None
    assert job.endpoint is None


def test_update_validates_field_values() -> None:
    store = JobStore()
    store.create("j")
    with pytest.raises(Exception):  # pydantic ValidationError
        store.update("j", status="bogus")  # type: ignore[arg-type]


def test_update_unknown_job_raises() -> None:
    store = JobStore()
    with pytest.raises(KeyError):
        store.update("missing", status=JobStatus.RUNNING)


def test_create_collision_raises() -> None:
    store = JobStore()
    store.create("dup")
    with pytest.raises(ValueError):
        store.create("dup")


def test_cleanup_removes_store_and_dir(tmp_path: Path) -> None:
    store = JobStore()
    store.create("j")
    job_dir = get_job_dir(tmp_path, "j")
    job_dir.mkdir(parents=True)
    (job_dir / "f.txt").write_text("hi")

    cleanup_job(store, tmp_path, "j")
    assert store.get("j") is None
    assert not job_dir.exists()


def test_cleanup_idempotent(tmp_path: Path) -> None:
    store = JobStore()
    # Neither in store nor on disk — should not raise.
    cleanup_job(store, tmp_path, "nope")


def test_disk_usage_counts_files(tmp_path: Path) -> None:
    assert disk_usage_bytes(tmp_path / "missing") == 0
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 50)
    assert disk_usage_bytes(tmp_path) == 150


def test_evict_only_finished_jobs(tmp_path: Path) -> None:
    store = JobStore()
    # 3 jobs: one running, two completed. Limit is small so eviction runs.
    for i, status in enumerate(
        [JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.COMPLETED]
    ):
        jid = f"j{i}"
        store.create(jid)
        store.update(jid, status=status)
        job_dir = get_job_dir(tmp_path, jid)
        job_dir.mkdir(parents=True)
        (job_dir / "data").write_bytes(b"x" * (1024 * 1024))

    # disk_limit_mb=1 means we're already over → evict at least one completed
    evicted = evict_finished_until_under_limit(store, tmp_path, limit_mb=1)
    assert evicted >= 1
    assert store.get("j0") is not None, "running jobs must never be evicted"
