"""Alibaba FC async-task-mode dispatcher.

Submit via POST /api/tasks/<endpoint> with FC async headers; status via FC
GetAsyncTask control plane (reliable, spins no function instance) with an HTTP
poll fallback when there's no function name / no AK/SK; download via HTTP.
Session affinity header routes follow-ups to the instance owning the job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from bioq_service.service_registry import ServiceRecord

from ..fc_status import FcStatusClient
from .base import encode_form, stream_download

SESSION_AFFINITY_HEADER = "X-Bioagent-Session-Id"


class FCDispatcher:
    def __init__(self, fc_status: FcStatusClient, client: httpx.Client | None = None) -> None:
        self._fc = fc_status
        self._client = client or httpx.Client(timeout=60.0)

    def submit(self, rec: ServiceRecord, endpoint: str, job_id: str,
               data: dict[str, Any], *, oss_prefix: str | None = None) -> str | None:
        headers = {
            "X-Fc-Invocation-Type": "Async",
            "X-Fc-Async-Task-Id": job_id,
            "X-Bioagent-Job-Id": job_id,
        }
        if oss_prefix:
            headers["X-Bioagent-Oss-Prefix"] = oss_prefix
        r = self._client.post(f"{rec.url}/api/tasks/{endpoint}",
                              data=encode_form(data), headers=headers)
        if r.status_code == 409:
            return None  # duplicate task id — idempotent, already submitted
        if r.status_code not in (200, 202):
            raise httpx.HTTPStatusError(
                f"submit failed: {r.status_code} {r.text!r}", request=r.request, response=r,
            )
        # FC keys the task by the X-Fc-Async-Task-Id we passed; status/download
        # use that same id, so no downstream handle to return.
        return None

    def status(self, rec: ServiceRecord, job_id: str) -> dict[str, Any]:
        # Control-plane status when we have a function name + AK/SK: reliable and
        # spins no downstream instance. Otherwise fall back to HTTP polling.
        if rec.function and self._fc.enabled:
            state = self._fc.get_status(function=rec.function, task_id=job_id, region=rec.region)
            return {"status": state}
        r = self._client.get(f"{rec.url}/api/jobs/{job_id}",
                             headers={SESSION_AFFINITY_HEADER: job_id})
        r.raise_for_status()
        return r.json()

    def download(self, rec: ServiceRecord, job_id: str, dest: Path) -> Path:
        return stream_download(self._client, f"{rec.url}/api/jobs/{job_id}/download",
                               dest, headers={SESSION_AFFINITY_HEADER: job_id})
