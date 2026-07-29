"""OpenFaaS dispatcher — async submit via the OpenFaaS gateway, poll for status.

Submit goes to the OpenFaaS **async** route (`POST /async-function/<fn>/...`),
which queues the request in NATS and invokes the function's task endpoint
(`/api/tasks/<endpoint>`, run synchronously by execute_task). Status/download
poll the OpenFaaS **sync** route (`/function/<fn>/api/jobs/<id>[/download]`),
which hits any replica — reliable because JobStore reads the job.json sidecar
fresh from disk, so a shared RWX `jobs_base_dir` lets any replica answer.

No callback subsystem: the worker's task endpoint honors X-Bioagent-Job-Id, so
the gateway already knows the job handle at submit time (returns None here).
Results must be persisted to shared storage (execute_task's output-sink mirrors
to the mount) since the function pod may scale to zero after completion.

`ServiceRecord.function` carries the OpenFaaS function name (required).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from bioq_service.service_registry import ServiceRecord

from .base import encode_form, stream_download


class OpenFaaSDispatcher:
    def __init__(self, gateway_url: str, client: httpx.Client | None = None) -> None:
        self._gw = gateway_url.rstrip("/")
        self._client = client or httpx.Client(timeout=60.0)

    @staticmethod
    def _fn(rec: ServiceRecord) -> str:
        if not rec.function:
            raise ValueError(
                "openfaas backend requires `function` (the OpenFaaS function name) "
                "in services.yaml for this service"
            )
        return rec.function

    def submit(self, rec: ServiceRecord, endpoint: str, job_id: str,
               data: dict[str, Any], *, oss_prefix: str | None = None) -> str | None:
        fn = self._fn(rec)
        headers = {"X-Bioagent-Job-Id": job_id}
        if oss_prefix:
            headers["X-Bioagent-Oss-Prefix"] = oss_prefix
        r = self._client.post(f"{self._gw}/async-function/{fn}/api/tasks/{endpoint}",
                              data=encode_form(data), headers=headers)
        if r.status_code == 409:
            return None  # duplicate — idempotent (execute_task dedups by job_id)
        if r.status_code not in (200, 202):
            raise httpx.HTTPStatusError(
                f"submit failed: {r.status_code} {r.text!r}", request=r.request, response=r,
            )
        # execute_task keys the job by the X-Bioagent-Job-Id we passed.
        return None

    def status(self, rec: ServiceRecord, job_id: str) -> dict[str, Any]:
        fn = self._fn(rec)
        r = self._client.get(f"{self._gw}/function/{fn}/api/jobs/{job_id}")
        r.raise_for_status()
        return r.json()

    def download(self, rec: ServiceRecord, job_id: str, dest: Path) -> Path:
        fn = self._fn(rec)
        return stream_download(
            self._client, f"{self._gw}/function/{fn}/api/jobs/{job_id}/download", dest
        )

    def describe_base_url(self, rec: ServiceRecord) -> str:
        # rec.url is a placeholder in openfaas mode — reach the worker through the
        # OpenFaaS gateway's sync route (any replica serves manifest/openapi).
        return f"{self._gw}/function/{self._fn(rec)}"
