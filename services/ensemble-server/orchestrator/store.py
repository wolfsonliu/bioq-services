"""EnsembleJobStore — NAS-backed key-value for EnsembleJob state.

Each job is one file: <jobs_base>/<task_id>/job.json.  Reads are
read-through cached (file mtime gating), writes hold a lock.  Mirrors
the patterns from bioq_service.JobStore.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from .models import EnsembleJob

SIDECAR_NAME = "job.json"


class EnsembleJobStore:
    """Thread-safe NAS-backed store.  One file per ensemble job."""

    def __init__(self, jobs_base_dir: Path) -> None:
        self._dir = jobs_base_dir
        self._cache: dict[str, EnsembleJob] = {}
        self._mtimes: dict[str, float] = {}
        self._lock = threading.Lock()

    def _sidecar(self, task_id: str) -> Path:
        return self._dir / task_id / SIDECAR_NAME

    def create(self, job: EnsembleJob) -> EnsembleJob:
        path = self._sidecar(job.task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        with self._lock:
            self._cache[job.task_id] = job
            self._mtimes[job.task_id] = path.stat().st_mtime
        return job

    def update(self, job: EnsembleJob) -> EnsembleJob:
        # Full rewrite — simple, atomic enough for MVP.
        return self.create(job)

    def get(self, task_id: str) -> Optional[EnsembleJob]:
        path = self._sidecar(task_id)
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return None
        with self._lock:
            if (cached := self._cache.get(task_id)) and self._mtimes.get(task_id) == mtime:
                return cached
        data = json.loads(path.read_text(encoding="utf-8"))
        job = EnsembleJob.model_validate(data)
        with self._lock:
            self._cache[task_id] = job
            self._mtimes[task_id] = mtime
        return job
