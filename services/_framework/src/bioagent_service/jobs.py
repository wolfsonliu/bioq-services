"""In-memory job store + per-job filesystem helpers + JSON sidecar persistence.

Each `JobStore.create` / `update` writes a `<job_dir>/job.json` sidecar so
jobs survive process or container restarts. On startup the framework calls
`reload_from_disk` to rehydrate the store from those sidecars (and to infer
minimal records for legacy job dirs without one).

**Multi-instance FC consistency.** Alibaba Cloud FC may keep several warm
instances of the same function, each mounting the same NAS at `jobs_base_dir`.
A job created by instance A may be polled on instance B because requests are
not pinned to instances. The sidecar is the source of truth: `get()` is
**read-through** — it stats the sidecar, compares mtime against an in-memory
mtime, and re-reads from disk on miss or staleness. Writes (`create` /
`update`) only happen on the instance that owns the subprocess, so writer
contention is not an issue. NAS attribute cache (default 1–30 s on NFS) means
non-owning instances may lag a few seconds behind the owner — acceptable for
clients polling at 5–120 s intervals.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bioagent_service.models import FailureKind, JobInfo, JobStatus

if TYPE_CHECKING:
    from bioagent_service.adapter import JobAdapter

logger = logging.getLogger(__name__)

# Filename used for the per-job pydantic-serialized JobInfo snapshot.
# Matches the legacy rfantibody-server layout so existing job dirs migrate cleanly.
SIDECAR_NAME = "job.json"


def new_job_id() -> str:
    """Short collision-resistant id used as the path segment and store key."""
    return uuid.uuid4().hex[:12]


def get_job_dir(jobs_base_dir: Path, job_id: str) -> Path:
    """Resolve <jobs_base_dir>/<job_id>. Does NOT create the directory."""
    return jobs_base_dir / job_id


class JobStore:
    """Thread-safe job_id → JobInfo map with sidecar persistence + read-through cache.

    Write path (`create` / `update`): mutate in-memory state, then write the
    sidecar. The instance that does this is by construction the one that owns
    the subprocess. Sidecar writes are tiny (~500 B) and happen ≤ a handful of
    times per job, so we hold the lock across the write to keep readers
    consistent.

    Read path (`get`): the sidecar on disk is treated as the source of truth.
    A successful read populates the in-memory cache; subsequent reads return
    from cache only while disk mtime is ≤ cached mtime. This lets a fresh FC
    instance answer queries for jobs created by other instances on the same NAS.

    `persist_dir=None` disables both sides (in-memory only); useful in tests.
    """

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._jobs: dict[str, JobInfo] = {}
        # mtime captured at the time of the last read or write of <persist_dir>/<id>/job.json.
        # Used by `get` to decide whether the in-memory copy is still authoritative.
        self._mtimes: dict[str, float] = {}
        self._lock = threading.Lock()
        self._persist_dir = persist_dir

    # ---- Internal helpers ----

    def _sidecar_path(self, job_id: str) -> Path | None:
        if self._persist_dir is None:
            return None
        return self._persist_dir / job_id / SIDECAR_NAME

    def _persist(self, job: JobInfo) -> None:
        """Write a JobInfo to its sidecar and capture the resulting mtime."""
        path = self._sidecar_path(job.job_id)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # `model_dump_json` is pydantic-validated by construction.
        path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        try:
            self._mtimes[job.job_id] = path.stat().st_mtime
        except OSError:
            # Stat failed — drop the mtime so a subsequent read re-validates.
            self._mtimes.pop(job.job_id, None)

    def _remember_mtime(self, job_id: str, mtime: float) -> None:
        """Capture an externally-observed mtime (e.g., during reload)."""
        if self._persist_dir is None:
            return
        with self._lock:
            self._mtimes[job_id] = mtime

    def _load_sidecar(self, job_id: str) -> tuple[JobInfo | None, float | None]:
        """Read the sidecar without touching the lock. Returns (job, mtime) or (None, None)."""
        path = self._sidecar_path(job_id)
        if path is None:
            return None, None
        try:
            mtime = path.stat().st_mtime
            data = json.loads(path.read_text(encoding="utf-8"))
            return JobInfo.model_validate(data), mtime
        except (FileNotFoundError, NotADirectoryError):
            return None, None
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.warning("failed to load sidecar for %s: %s", job_id, e)
            return None, None

    # ---- Public API ----

    def create(self, job_id: str | None = None) -> JobInfo:
        if job_id is None:
            job_id = new_job_id()
        info = JobInfo(job_id=job_id, status=JobStatus.PENDING)
        with self._lock:
            if job_id in self._jobs:
                raise ValueError(f"job {job_id!r} already exists")
            self._jobs[job_id] = info
            self._persist(info)
        return info

    def get(self, job_id: str) -> JobInfo | None:
        """Return the freshest known JobInfo for `job_id`, or None.

        With a persist_dir set, stats the sidecar to detect external writes by
        other instances; refreshes the in-memory cache transparently. Without
        a persist_dir, this is a pure in-memory lookup.
        """
        # Fast path: no persistence configured.
        if self._persist_dir is None:
            with self._lock:
                return self._jobs.get(job_id)

        path = self._sidecar_path(job_id)
        assert path is not None  # guaranteed by persist_dir check above

        # Stat probe (cheap on NAS, ~1 ms) to detect external writes / deletes.
        try:
            disk_mtime = path.stat().st_mtime
        except (FileNotFoundError, NotADirectoryError):
            # Sidecar gone (likely deleted by another instance) — evict our cache.
            with self._lock:
                self._jobs.pop(job_id, None)
                self._mtimes.pop(job_id, None)
            return None
        except OSError as e:
            # Transient NAS error — fall back to whatever we have cached.
            logger.warning("stat failed for %s: %s; serving from cache", path, e)
            with self._lock:
                return self._jobs.get(job_id)

        with self._lock:
            cached = self._jobs.get(job_id)
            cached_mtime = self._mtimes.get(job_id)
            if (
                cached is not None
                and cached_mtime is not None
                and cached_mtime >= disk_mtime
            ):
                return cached

        # Cache miss or stale — re-read sidecar without the lock (I/O off the hot path).
        loaded, mtime = self._load_sidecar(job_id)
        if loaded is None:
            # Read failure: prefer the stale cache over None to avoid flapping.
            with self._lock:
                return self._jobs.get(job_id)

        with self._lock:
            self._jobs[job_id] = loaded
            if mtime is not None:
                self._mtimes[job_id] = mtime
            return loaded

    def update(self, job_id: str, **fields: Any) -> JobInfo:
        """Replace the stored JobInfo with one whose fields are updated.

        Pydantic validates the result, so passing an invalid status/value raises.
        Only the instance that owns the subprocess should call `update` for a
        given job; concurrent updates from multiple instances are not supported.
        """
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                raise KeyError(f"job {job_id!r} not found")
            updated = current.model_copy(update=fields)
            # model_copy(update=...) skips validation; re-validate explicitly.
            JobInfo.model_validate(updated.model_dump())
            self._jobs[job_id] = updated
            self._persist(updated)
            return updated

    def insert(self, job: JobInfo) -> JobInfo:
        """Insert a fully-formed JobInfo (e.g., recovered from disk).

        Distinct from `create` because it doesn't allocate a fresh PENDING state —
        it stores exactly what's given. Used by `reload_from_disk` and tests.
        Does NOT write a sidecar (the sidecar is presumed to already exist on disk).
        Callers wanting durability should follow with `_persist(job)` and use
        `_remember_mtime` to align the read-through cache.
        """
        with self._lock:
            if job.job_id in self._jobs:
                raise ValueError(f"job {job.job_id!r} already exists")
            self._jobs[job.job_id] = job
        return job

    def all_jobs(self) -> list[JobInfo]:
        """Return jobs currently in the in-memory cache.

        Note: in a multi-instance FC deployment, this does NOT enumerate jobs
        owned by other instances. It reflects only what this instance has
        created or observed via `get`. Use `reload_from_disk` to bulk-rehydrate.
        """
        with self._lock:
            return list(self._jobs.values())

    def remove(self, job_id: str) -> None:
        """Drop the in-memory entry (and any cached mtime). Does NOT touch disk."""
        with self._lock:
            self._jobs.pop(job_id, None)
            self._mtimes.pop(job_id, None)


def cleanup_job(store: JobStore, jobs_base_dir: Path, job_id: str) -> None:
    """Remove a job from the store AND delete its filesystem directory.

    Safe to call repeatedly; missing entries / dirs are silently ignored.
    """
    store.remove(job_id)
    job_dir = get_job_dir(jobs_base_dir, job_id)
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)


def disk_usage_bytes(jobs_base_dir: Path) -> int:
    """Sum of file sizes under jobs_base_dir; 0 if dir does not exist."""
    if not jobs_base_dir.exists():
        return 0
    return sum(f.stat().st_size for f in jobs_base_dir.rglob("*") if f.is_file())


def evict_finished_until_under_limit(
    store: JobStore, jobs_base_dir: Path, limit_mb: int
) -> int:
    """If disk usage exceeds limit_mb, remove completed/failed jobs until under.

    Returns the number of jobs evicted. Pending/running jobs are never touched.
    """
    limit_bytes = limit_mb * 1024 * 1024
    if disk_usage_bytes(jobs_base_dir) <= limit_bytes:
        return 0
    evicted = 0
    for job in store.all_jobs():
        if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
            continue
        cleanup_job(store, jobs_base_dir, job.job_id)
        evicted += 1
        if disk_usage_bytes(jobs_base_dir) <= limit_bytes:
            break
    return evicted


def reload_from_disk(
    store: "JobStore",
    adapter: "JobAdapter",
    jobs_base_dir: Path,
) -> int:
    """Repopulate `store` from sidecars (and legacy dirs) under `jobs_base_dir`.

    For each `<jobs_base_dir>/<job_id>/`:
      * `job.json` exists, status ∈ {pending, completed, failed} → load as-is
        and align the read-through cache mtime with the on-disk file (no
        rewrite, so other instances' caches stay valid).
      * `job.json` exists, status == running → downgrade to FAILED with
        `failure_kind=INTERRUPTED`, then rewrite the sidecar so the correction
        is durable.
      * `job.json` missing → call `adapter.infer_job_from_dir(job_dir)` and
        write a fresh sidecar backfilling the legacy directory.

    Returns the number of jobs successfully restored. Malformed sidecars are
    logged and skipped without aborting startup.
    """
    if not jobs_base_dir.exists():
        return 0
    restored = 0
    for job_dir in sorted(jobs_base_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        job_id = job_dir.name
        sidecar = job_dir / SIDECAR_NAME

        rewrite_sidecar: bool
        inserted: JobInfo
        observed_mtime: float | None = None

        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                job = JobInfo.model_validate(data)
                observed_mtime = sidecar.stat().st_mtime
            except (OSError, json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "skipping job %s: failed to parse sidecar %s: %s",
                    job_id, sidecar, e,
                )
                continue
            if job.status == JobStatus.RUNNING:
                inserted = job.model_copy(update={
                    "status": JobStatus.FAILED,
                    "message": "Interrupted by container restart",
                    "failure_kind": FailureKind.INTERRUPTED,
                })
                rewrite_sidecar = True
            else:
                inserted = job
                rewrite_sidecar = False
        else:
            try:
                inferred = adapter.infer_job_from_dir(job_dir)
            except Exception as e:
                logger.warning(
                    "skipping job %s: adapter.infer_job_from_dir raised: %s",
                    job_id, e,
                )
                continue
            logger.info(
                "inferred job %s from disk: status=%s",
                job_id, inferred.status.value,
            )
            inserted = inferred
            rewrite_sidecar = True

        try:
            store.insert(inserted)
        except ValueError:
            # Already present (shouldn't happen on a fresh store) — skip gracefully.
            logger.warning("job %s already in store during reload; skipping", job_id)
            continue

        if rewrite_sidecar:
            # Bumps the sidecar's mtime; other instances will re-read on next get().
            with store._lock:
                store._persist(inserted)
        elif observed_mtime is not None:
            # Align the cache mtime with what we just read so `get` short-circuits.
            store._remember_mtime(job_id, observed_mtime)

        restored += 1
    if restored:
        logger.info("restored %d job(s) from %s", restored, jobs_base_dir)
    return restored


__all__ = [
    "FailureKind",   # re-export so callers can import jobs + failure kinds together
    "JobInfo",
    "JobStatus",
    "JobStore",
    "SIDECAR_NAME",
    "cleanup_job",
    "disk_usage_bytes",
    "evict_finished_until_under_limit",
    "get_job_dir",
    "new_job_id",
    "reload_from_disk",
]
