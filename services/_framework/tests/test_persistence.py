"""JSON sidecar persistence + reload_from_disk recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from bioagent_service import JobAdapter, ServiceSettings
from bioagent_service.jobs import (
    SIDECAR_NAME,
    JobStore,
    reload_from_disk,
)
from bioagent_service.models import FailureKind, JobInfo, JobStatus


class _Adapter(JobAdapter):
    name = "test"


@pytest.fixture
def adapter(tmp_path: Path) -> _Adapter:
    settings = ServiceSettings(jobs_base_dir=tmp_path)
    return _Adapter(settings=settings)


# ---- Sidecar write ----


def test_create_writes_sidecar(tmp_path: Path) -> None:
    store = JobStore(persist_dir=tmp_path)
    store.create("abc", input_params={"model": "aa", "steps": 25})
    sidecar = tmp_path / "abc" / SIDECAR_NAME
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert data["job_id"] == "abc"
    assert data["status"] == "pending"
    assert data["created_at"] is not None
    assert data["input_params"] == {"model": "aa", "steps": 25}


def test_update_rewrites_sidecar(tmp_path: Path) -> None:
    store = JobStore(persist_dir=tmp_path)
    store.create("abc")
    store.update("abc", status=JobStatus.RUNNING, message="go")
    data = json.loads((tmp_path / "abc" / SIDECAR_NAME).read_text())
    assert data["status"] == "running"
    assert data["message"] == "go"


def test_no_sidecar_when_persist_dir_unset(tmp_path: Path) -> None:
    """Default JobStore() doesn't touch disk — useful for unit tests."""
    store = JobStore()
    store.create("abc")
    store.update("abc", status=JobStatus.RUNNING)
    assert not any(tmp_path.iterdir())  # tmp_path is unrelated, just verifying no leak


# ---- Reload ----


def test_reload_restores_from_sidecar(tmp_path: Path, adapter: _Adapter) -> None:
    # Simulate a previous run: create + complete a job, then drop the in-memory store.
    pre = JobStore(persist_dir=tmp_path)
    pre.create("done1")
    out_dir = adapter.output_dir(adapter.job_dir("done1"))
    out_dir.mkdir(parents=True)
    (out_dir / "r.txt").write_text("x")
    pre.update("done1", status=JobStatus.COMPLETED, message="ok")

    # New process starts up with a fresh store.
    fresh = JobStore(persist_dir=tmp_path)
    n = reload_from_disk(fresh, adapter, tmp_path)
    assert n == 1
    restored = fresh.get("done1")
    assert restored is not None
    assert restored.status == JobStatus.COMPLETED
    assert restored.message == "ok"


def test_reload_running_downgrades_to_interrupted(
    tmp_path: Path, adapter: _Adapter
) -> None:
    """Same instance restarting should mark its own running jobs as interrupted."""
    from bioagent_service.models import utcnow

    instance_id = "same-instance"
    pre = JobStore(persist_dir=tmp_path, instance_id=instance_id)
    pre.create("zombie")
    started = utcnow()
    pre.update("zombie", status=JobStatus.RUNNING, message="mid-flight", started_at=started)

    fresh = JobStore(persist_dir=tmp_path, instance_id=instance_id)
    reload_from_disk(fresh, adapter, tmp_path)
    restored = fresh.get("zombie")
    assert restored is not None
    assert restored.status == JobStatus.FAILED
    assert restored.failure_kind == FailureKind.INTERRUPTED
    assert "Interrupted" in (restored.message or "")
    assert restored.completed_at is not None
    assert restored.duration_seconds is not None
    assert restored.duration_seconds >= 0

    # The corrected status must be durable across the *next* restart too.
    sidecar = json.loads((tmp_path / "zombie" / SIDECAR_NAME).read_text())
    assert sidecar["status"] == "failed"
    assert sidecar["failure_kind"] == "interrupted"


def test_reload_skips_running_job_from_other_instance(
    tmp_path: Path, adapter: _Adapter
) -> None:
    """A different instance's running job must NOT be marked interrupted."""
    from bioagent_service.models import utcnow

    owner = JobStore(persist_dir=tmp_path, instance_id="instance-A")
    owner.create("active")
    owner.update("active", status=JobStatus.RUNNING, started_at=utcnow())

    other = JobStore(persist_dir=tmp_path, instance_id="instance-B")
    reload_from_disk(other, adapter, tmp_path)
    restored = other.get("active")
    assert restored is not None
    assert restored.status == JobStatus.RUNNING

    # Sidecar on disk must NOT be rewritten.
    sidecar = json.loads((tmp_path / "active" / SIDECAR_NAME).read_text())
    assert sidecar["status"] == "running"
    assert sidecar["instance_id"] == "instance-A"


def test_reload_infers_legacy_dir_with_outputs(
    tmp_path: Path, adapter: _Adapter
) -> None:
    # A legacy job dir created before sidecar persistence existed: outputs but no job.json.
    legacy_dir = adapter.job_dir("legacy")
    out = adapter.output_dir(legacy_dir)
    out.mkdir(parents=True)
    (out / "result.bin").write_bytes(b"data")

    fresh = JobStore(persist_dir=tmp_path)
    n = reload_from_disk(fresh, adapter, tmp_path)
    assert n == 1
    restored = fresh.get("legacy")
    assert restored is not None
    assert restored.status == JobStatus.COMPLETED

    # The framework should backfill the sidecar so next restart doesn't re-infer.
    assert (legacy_dir / SIDECAR_NAME).exists()


def test_reload_infers_legacy_dir_without_outputs(
    tmp_path: Path, adapter: _Adapter
) -> None:
    legacy_dir = adapter.job_dir("legacy-fail")
    legacy_dir.mkdir(parents=True)
    # Empty job dir — likely a job that failed before producing anything.
    fresh = JobStore(persist_dir=tmp_path)
    reload_from_disk(fresh, adapter, tmp_path)
    restored = fresh.get("legacy-fail")
    assert restored is not None
    assert restored.status == JobStatus.FAILED


def test_reload_uses_custom_adapter_inference(tmp_path: Path) -> None:
    """Subclasses can override `infer_job_from_dir` to add service-specific state."""

    class _CustomAdapter(_Adapter):
        def infer_job_from_dir(self, job_dir: Path) -> JobInfo:
            return JobInfo(
                job_id=job_dir.name,
                status=JobStatus.COMPLETED,
                progress="step-3",
                message="custom",
            )

    adapter = _CustomAdapter(settings=ServiceSettings(jobs_base_dir=tmp_path))
    (tmp_path / "j").mkdir()  # empty legacy dir; output detection irrelevant

    fresh = JobStore(persist_dir=tmp_path)
    reload_from_disk(fresh, adapter, tmp_path)
    restored = fresh.get("j")
    assert restored is not None
    assert restored.progress == "step-3"
    assert restored.message == "custom"


def test_reload_skips_malformed_sidecar(
    tmp_path: Path, adapter: _Adapter
) -> None:
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / SIDECAR_NAME).write_text("{not json")

    fresh = JobStore(persist_dir=tmp_path)
    # Should not raise; just skip the malformed entry.
    n = reload_from_disk(fresh, adapter, tmp_path)
    assert n == 0
    assert fresh.get("bad") is None


def test_reload_ignores_non_dirs_at_root(
    tmp_path: Path, adapter: _Adapter
) -> None:
    (tmp_path / "stray.txt").write_text("not a job dir")
    fresh = JobStore(persist_dir=tmp_path)
    n = reload_from_disk(fresh, adapter, tmp_path)
    assert n == 0


def test_reload_no_jobs_dir_returns_zero(tmp_path: Path, adapter: _Adapter) -> None:
    n = reload_from_disk(JobStore(), adapter, tmp_path / "absent")
    assert n == 0


# ---- Read-through cache (multi-instance FC) ----


def test_get_loads_from_sidecar_when_not_cached(tmp_path: Path) -> None:
    """A new store sees jobs created by a peer instance via the sidecar."""
    instance_a = JobStore(persist_dir=tmp_path)
    instance_a.create("shared")
    instance_a.update("shared", status=JobStatus.RUNNING, message="A is working")

    # Instance B starts fresh, never called reload — read-through should still find it.
    instance_b = JobStore(persist_dir=tmp_path)
    found = instance_b.get("shared")
    assert found is not None
    assert found.status == JobStatus.RUNNING
    assert found.message == "A is working"


def test_get_refreshes_when_sidecar_mtime_advances(tmp_path: Path) -> None:
    """A later write by instance A is visible to instance B on next get."""
    a = JobStore(persist_dir=tmp_path)
    b = JobStore(persist_dir=tmp_path)
    a.create("evolving")
    a.update("evolving", status=JobStatus.RUNNING)

    # B caches the running state.
    cached = b.get("evolving")
    assert cached is not None
    assert cached.status == JobStatus.RUNNING

    # A finishes the job — bump mtime forcibly to avoid filesystem-resolution flakiness.
    a.update("evolving", status=JobStatus.COMPLETED, message="done")
    sidecar = tmp_path / "evolving" / SIDECAR_NAME
    import os
    future = sidecar.stat().st_mtime + 1.0
    os.utime(sidecar, (future, future))

    fresh = b.get("evolving")
    assert fresh is not None
    assert fresh.status == JobStatus.COMPLETED
    assert fresh.message == "done"


def test_get_evicts_cache_when_sidecar_deleted(tmp_path: Path) -> None:
    """If instance A deletes the job, instance B's next get returns None."""
    a = JobStore(persist_dir=tmp_path)
    b = JobStore(persist_dir=tmp_path)
    a.create("doomed")

    # Warm B's cache.
    assert b.get("doomed") is not None

    # Simulate instance A's cleanup_job removing the dir.
    import shutil
    shutil.rmtree(tmp_path / "doomed")

    # B's next get should observe the deletion and clear its cache.
    assert b.get("doomed") is None
    # And confirm the eviction stuck: another get is still None.
    assert b.get("doomed") is None


def test_get_returns_none_for_unknown_job_with_persist_dir(tmp_path: Path) -> None:
    store = JobStore(persist_dir=tmp_path)
    assert store.get("never-existed") is None


def test_get_serves_stale_cache_when_sidecar_corrupted(tmp_path: Path) -> None:
    """If the sidecar is corrupted mid-flight, prefer the stale cache over None."""
    store = JobStore(persist_dir=tmp_path)
    store.create("flaky")
    # Warm cache.
    assert store.get("flaky") is not None

    # Corrupt the sidecar (and bump mtime so the read-through path is taken).
    sidecar = tmp_path / "flaky" / SIDECAR_NAME
    sidecar.write_text("{ corrupted")
    import os
    future = sidecar.stat().st_mtime + 1.0
    os.utime(sidecar, (future, future))

    # Cache fall-back kicks in.
    served = store.get("flaky")
    assert served is not None
    assert served.job_id == "flaky"


def test_get_skips_cache_for_non_terminal_even_when_mtime_unchanged(tmp_path: Path) -> None:
    """Reproduces the 2026-05-12 genie3 incident: NFS attribute cache returns
    stale mtime so the read-through path looks like a hit, but the on-disk
    content has actually advanced. For non-terminal (pending/running) cached
    states the framework MUST always re-read the sidecar.
    """
    import os

    store = JobStore(persist_dir=tmp_path)
    store.create("evolving")  # in cache as PENDING

    # Capture the mtime the cache thinks is current.
    sidecar = tmp_path / "evolving" / SIDECAR_NAME
    pinned_mtime = sidecar.stat().st_mtime

    # Simulate the owning instance writing a new state, then **pin mtime back**
    # to what the cache saw — mimicking an NFS attribute cache returning stale
    # `disk_mtime` even though the file content has changed underneath.
    sidecar.write_text(JobInfo(
        job_id="evolving",
        status=JobStatus.COMPLETED,
        message="finished by peer",
    ).model_dump_json())
    os.utime(sidecar, (pinned_mtime, pinned_mtime))

    fresh = store.get("evolving")
    assert fresh is not None
    # Before the fix this would return the stale cached "pending"; with the fix
    # the framework re-reads because cached status was non-terminal.
    assert fresh.status == JobStatus.COMPLETED
    assert fresh.message == "finished by peer"


def test_get_uses_cache_for_terminal_when_mtime_unchanged(tmp_path: Path) -> None:
    """Terminal cached states are immutable; serving from cache on stat-equal
    is the optimization we want to keep. Verifies the fix didn't regress this."""
    import os

    store = JobStore(persist_dir=tmp_path)
    store.create("done")
    store.update("done", status=JobStatus.COMPLETED, message="real")

    sidecar = tmp_path / "done" / SIDECAR_NAME
    pinned_mtime = sidecar.stat().st_mtime
    # Tamper with the sidecar to a "wrong" state but keep mtime: a real
    # client should never see this because the cached terminal status takes
    # precedence as long as disk mtime hasn't advanced.
    sidecar.write_text(JobInfo(
        job_id="done",
        status=JobStatus.RUNNING,
        message="ghost",
    ).model_dump_json())
    os.utime(sidecar, (pinned_mtime, pinned_mtime))

    served = store.get("done")
    assert served is not None
    assert served.status == JobStatus.COMPLETED
    assert served.message == "real"


def test_get_in_memory_only_mode_ignores_disk(tmp_path: Path) -> None:
    """persist_dir=None disables both writes and read-through. Unchanged classic behavior."""
    persisting = JobStore(persist_dir=tmp_path)
    persisting.create("written")  # sidecar exists at tmp_path/written/job.json

    # An in-memory-only store with NO persist_dir does not consult the disk.
    in_mem = JobStore()
    assert in_mem.get("written") is None
