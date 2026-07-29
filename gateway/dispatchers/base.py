"""Dispatcher protocol + shared HTTP helpers.

A `Dispatcher` is the seam that isolates the execution/scaling platform from the
gateway. Backends: `FCDispatcher` (Alibaba FC async task mode), `LocalHttpDispatcher`
(plain HTTP submit/poll against a service's own in-process runner — Compose/K8s),
and later OpenFaaS/KEDA. All backends take a `ServiceRecord` (not a bare URL) so
each can use whatever fields it needs (url / function / region).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx
from bioq_service.service_registry import ServiceRecord


@runtime_checkable
class Dispatcher(Protocol):
    """Submit a job, poll its status, and fetch its result archive."""

    def submit(self, rec: ServiceRecord, endpoint: str, job_id: str,
               data: dict[str, Any], *, oss_prefix: str | None = None) -> str | None:
        """Return the downstream job handle to use for subsequent status/download
        (e.g. a worker-generated id), or None to keep using the passed job_id."""
        ...

    def status(self, rec: ServiceRecord, job_id: str) -> dict[str, Any]: ...

    def download(self, rec: ServiceRecord, job_id: str, dest: Path) -> Path: ...

    def describe_base_url(self, rec: ServiceRecord) -> str:
        """Base URL `describe` fetches `/api/manifest` + `/openapi.json` from.

        Backends where the worker is reachable directly return `rec.url`; the
        OpenFaaS backend routes through its gateway (`{gw}/function/<fn>`) since
        the worker has no static URL (rec.url is a placeholder there)."""
        ...


def encode_form(data: dict[str, Any]) -> dict[str, str]:
    """Form-encode a request body: str as-is; list/dict -> JSON (so structured
    params and multi-file-under-one-field survive downstream form parsing, which
    expects JSON not Python repr); other scalars -> str; drop None.
    """
    form: dict[str, str] = {}
    for k, v in data.items():
        if v is None:
            continue
        form[k] = v if isinstance(v, str) else (
            json.dumps(v) if isinstance(v, (list, dict)) else str(v)
        )
    return form


def stream_download(client: httpx.Client, url: str, dest: Path,
                    headers: dict[str, str] | None = None) -> Path:
    """Stream a GET to `dest`, surfacing a readable body on error.

    Reading the streamed body before `raise_for_status()` ensures the raised
    `HTTPStatusError` carries `.text` (otherwise callers hit ResponseNotRead).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with client.stream("GET", url, headers=headers or {}) as r:
        if r.status_code >= 400:
            r.read()
            r.raise_for_status()
        with open(dest, "wb") as fh:
            fh.writelines(r.iter_bytes())
    return dest
