"""HTTP async dispatch to downstream FC services (VPC).

MVP transport: submit via POST /api/tasks/<endpoint> with FC async headers,
poll via GET /api/jobs/<id>, download via GET /api/jobs/<id>/download. Gateway
runs in VPC so downstream VPC-bypass covers auth. Session affinity header
routes follow-ups to the instance owning the job.

Hardening (later): FC GetAsyncTask (AK/SK) for reliable status at high
concurrency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

SESSION_AFFINITY_HEADER = "X-Bioagent-Session-Id"


class HttpDispatch:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=60.0)

    def submit(self, base_url: str, endpoint: str, job_id: str,
               data: dict[str, Any]) -> None:
        headers = {
            "X-Fc-Invocation-Type": "Async",
            "X-Fc-Async-Task-Id": job_id,
            "X-Bioagent-Job-Id": job_id,
        }
        form = {k: str(v) for k, v in data.items()}
        r = self._client.post(f"{base_url}/api/tasks/{endpoint}", data=form, headers=headers)
        if r.status_code == 409:
            return  # duplicate task id — idempotent, already submitted
        if r.status_code not in (200, 202):
            raise httpx.HTTPStatusError(
                f"submit failed: {r.status_code} {r.text!r}", request=r.request, response=r,
            )

    def status(self, base_url: str, job_id: str) -> dict[str, Any]:
        r = self._client.get(f"{base_url}/api/jobs/{job_id}",
                             headers={SESSION_AFFINITY_HEADER: job_id})
        r.raise_for_status()
        return r.json()

    def download(self, base_url: str, job_id: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._client.stream("GET", f"{base_url}/api/jobs/{job_id}/download",
                                 headers={SESSION_AFFINITY_HEADER: job_id}) as r:
            if r.status_code >= 400:
                # Read the streamed body so raise_for_status()'s exception
                # carries .text (otherwise callers hit httpx.ResponseNotRead).
                r.read()
                r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
        return dest
