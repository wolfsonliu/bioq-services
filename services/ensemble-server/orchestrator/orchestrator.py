"""Orchestrator — task-kind agnostic fan-out + gather.

Coordinates: API request → submit N FC sub-tasks → poll → collect outputs →
aggregate.  Does NOT know about FoldingInput / DesignInput schemas — only
sees them as opaque dicts.  Each TaskKind hooks aggregation via a callable
registered at app startup.
"""

from __future__ import annotations

import logging
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from ..adapters.registry import MethodRegistry
from ..dispatcher import DispatchHandle, TaskStatus
from ..task_kind import TaskKind
from .models import EnsembleJob, SubTaskRecord, SubTaskStatus
from .store import EnsembleJobStore

logger = logging.getLogger(__name__)

# An "aggregator" merges per-method outputs into a single AggregatedOutput dict.
AggregatorFn = Callable[[list[SubTaskRecord]], dict[str, Any]]


class Orchestrator:
    """Task-kind agnostic fan-out + lazy poll + aggregation."""

    def __init__(
        self,
        *,
        registry: MethodRegistry,
        store: EnsembleJobStore,
        aggregators: dict[TaskKind, AggregatorFn],
    ) -> None:
        self.registry = registry
        self.store = store
        self.aggregators = aggregators

    def _new_task_id(self, task_kind: TaskKind) -> str:
        kind_short = {
            "folding": "fold",
            "design": "des",
            "scoring": "score",
        }[task_kind.value]
        return f"ens_{kind_short}_{uuid.uuid4().hex[:20]}"

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    async def submit(
        self,
        *,
        task_kind: TaskKind,
        input: Any,                          # validated Pydantic input
        methods: list[str],
        method_options: dict[str, Any],
        customer_id: str,
    ) -> EnsembleJob:
        """Create EnsembleJob, fan out FC submits, persist, return."""
        task_id = self._new_task_id(task_kind)
        job = EnsembleJob(
            task_id=task_id,
            task_kind=task_kind.value,
            customer_id=customer_id,
            submitted_at=self._now(),
            input=input.model_dump(mode="json"),
            requested_methods=methods,
        )

        for m in methods:
            adapter = self.registry.get(task_kind, m)
            opts_schema = adapter.method_options_schema
            opts = opts_schema.model_validate(method_options.get(m, {}))

            sub_task_id = f"{task_id}__{m}"
            try:
                endpoint, payload, files = adapter.build_request(input, opts)
                handle = adapter.fc.submit(
                    task_id=sub_task_id,
                    endpoint=endpoint,
                    payload=payload,
                    files=files,
                )
                job.sub_tasks[m] = SubTaskRecord(
                    method=m,
                    sub_task_id=sub_task_id,
                    status=SubTaskStatus.RUNNING,
                    fc_invocation_id=handle.backend_ref.get("invocation_id"),
                    started_at=self._now(),
                )
            except Exception as exc:
                logger.exception("submit failed for %s/%s", task_id, m)
                job.sub_tasks[m] = SubTaskRecord(
                    method=m,
                    sub_task_id=sub_task_id,
                    status=SubTaskStatus.FAILED,
                    error_summary=str(exc),
                )

        self.store.create(job)
        return job

    async def refresh(self, task_id: str) -> Optional[EnsembleJob]:
        """Poll each non-terminal sub-task once and persist.

        Called by GET /v1/jobs/<task_id> to lazy-update state.  No background
        worker in Phase 1.
        """
        job = self.store.get(task_id)
        if job is None:
            return None
        if job.completed_at is not None:
            return job

        task_kind = TaskKind(job.task_kind)
        changed = False
        for m, sub in job.sub_tasks.items():
            if sub.status in (
                SubTaskStatus.SUCCEEDED,
                SubTaskStatus.FAILED,
                SubTaskStatus.CACHED,
            ):
                continue
            adapter = self.registry.get(task_kind, m)
            handle = DispatchHandle(
                backend=adapter.fc.backend_name,
                task_id=sub.sub_task_id,
                backend_ref={
                    "function": adapter.fc.function,
                    "invocation_id": sub.fc_invocation_id or sub.sub_task_id,
                },
            )
            try:
                status = adapter.fc.get_status(handle)
            except Exception as e:
                logger.warning("get_status failed %s/%s: %s", task_id, m, e)
                continue

            if status == TaskStatus.SUCCEEDED:
                dest = self.store._dir / task_id / "outputs" / m
                dest.mkdir(parents=True, exist_ok=True)
                try:
                    zip_path = adapter.fc.fetch_result(handle, dest_dir=dest)
                    self._unzip(zip_path, dest)
                    out = adapter.normalize_output(sub.sub_task_id, dest)
                    sub.output = out.model_dump(mode="json")
                    sub.status = SubTaskStatus.SUCCEEDED
                    sub.completed_at = self._now()
                    if sub.started_at:
                        sub.runtime_seconds = (sub.completed_at - sub.started_at).total_seconds()
                    changed = True
                except Exception as exc:
                    logger.exception("fetch/normalize failed %s/%s", task_id, m)
                    sub.status = SubTaskStatus.FAILED
                    sub.error_summary = f"fetch_or_normalize: {exc}"
                    changed = True
            elif status == TaskStatus.FAILED:
                sub.status = SubTaskStatus.FAILED
                sub.completed_at = self._now()
                sub.error_summary = "FC reports FAILED"
                changed = True
            # else: still PENDING / RUNNING

        # If all sub-tasks terminal, aggregate
        terminal_states = (
            SubTaskStatus.SUCCEEDED,
            SubTaskStatus.FAILED,
            SubTaskStatus.CACHED,
        )
        if all(s.status in terminal_states for s in job.sub_tasks.values()):
            if any(s.status == SubTaskStatus.SUCCEEDED for s in job.sub_tasks.values()):
                aggregator = self.aggregators.get(task_kind)
                if aggregator:
                    job.aggregated_output = aggregator(list(job.sub_tasks.values()))
            job.completed_at = self._now()
            changed = True

        if changed:
            self.store.update(job)
        return job

    def _unzip(self, zip_path: Path, dest_dir: Path) -> None:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
