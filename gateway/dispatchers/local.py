"""Local HTTP dispatcher — plain submit/poll against a service's own runner.

For Compose / plain K8s (no FC): submit to the neutral POST /api/<endpoint>
route (NOT /api/tasks/<endpoint>, which is FC-async-only), which returns a job
handled by the service's in-process async runner; poll GET /api/jobs/<id>;
download GET /api/jobs/<id>/download. No FC headers, no session affinity (fixed
replicas, no cross-instance routing to worry about).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from bioq_service.service_registry import ServiceRecord

from .base import encode_form, stream_download


class LocalHttpDispatcher:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=60.0)

    def submit(self, rec: ServiceRecord, endpoint: str, job_id: str,
               data: dict[str, Any], *, oss_prefix: str | None = None) -> str | None:
        headers = {"X-Bioagent-Job-Id": job_id}
        if oss_prefix:
            headers["X-Bioagent-Oss-Prefix"] = oss_prefix
        r = self._client.post(f"{rec.url}/api/{endpoint}",
                              data=encode_form(data), headers=headers)
        if r.status_code == 409:
            return None  # duplicate job id — idempotent
        if r.status_code not in (200, 202):
            raise httpx.HTTPStatusError(
                f"submit failed: {r.status_code} {r.text!r}", request=r.request, response=r,
            )
        # The service's in-process runner assigns its own job_id; the gateway must
        # track it for status/download (the worker doesn't honor our job_id).
        try:
            return r.json().get("job_id")
        except Exception:  # noqa: BLE001 — non-JSON body: fall back to our id
            return None

    def status(self, rec: ServiceRecord, job_id: str) -> dict[str, Any]:
        r = self._client.get(f"{rec.url}/api/jobs/{job_id}")
        r.raise_for_status()
        return r.json()

    def download(self, rec: ServiceRecord, job_id: str, dest: Path) -> Path:
        return stream_download(self._client, f"{rec.url}/api/jobs/{job_id}/download", dest)

    def describe_base_url(self, rec: ServiceRecord) -> str:
        return rec.url
