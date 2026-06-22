"""HTTPDispatcher — invokes downstream bioagent services via plain HTTP.

Used by the ensemble orchestrator instead of FC OpenAPI.  The downstream
services already expose every operation we need over HTTP:

  - submit       :  POST  {base}/api/tasks/<name>
                    + ``X-Fc-Invocation-Type: Async``
                    + ``X-Fc-Async-Task-Id`` / ``X-Bioagent-Job-Id``
                    → FC's HTTP trigger intercepts and returns 202.
  - get_status   :  GET   {base}/api/jobs/<task_id>  → read ``status`` field
                    (service framework's JobInfo).
  - fetch_result :  GET   {base}/api/jobs/<task_id>/download → streamed zip.
  - stop         :  best-effort no-op (FC will reclaim instances on its own).

Sub-task statuses returned by the service framework are mapped to the
backend-neutral :class:`TaskStatus`.

This module is a deliberate fork of pipelines.framework.dispatcher: keeping
``DispatchHandle`` / ``TaskStatus`` co-located here means ensemble-server
has zero source-level dependency on the pipelines package, which lets the
Dockerfile drop the entire ``pipelines/`` copy and the Alibaba Cloud SDK
deps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import httpx


logger = logging.getLogger(__name__)


# Server-side framework's _SessionAffinityMiddleware writes this header into
# POST 200 responses with value=job_id; FC's HeaderField affinity routes
# subsequent requests with a matching header back to the same instance.
SESSION_AFFINITY_HEADER = "X-Bioagent-Session-Id"


class TaskStatus(str, Enum):
    """Lifecycle state of a dispatched task — backend-neutral."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# JobInfo.status values used by the bioagent service framework.
_JOBINFO_STATUS_MAP: dict[str, TaskStatus] = {
    "pending":     TaskStatus.PENDING,
    "queued":      TaskStatus.PENDING,
    "running":     TaskStatus.RUNNING,
    "completed":   TaskStatus.SUCCEEDED,
    "succeeded":   TaskStatus.SUCCEEDED,
    "failed":      TaskStatus.FAILED,
    "interrupted": TaskStatus.FAILED,
    "cancelled":   TaskStatus.FAILED,
}


@dataclass(frozen=True)
class DispatchHandle:
    """Opaque-to-caller handle for tracking a submitted task."""
    backend: str
    task_id: str
    backend_ref: dict[str, Any] = field(default_factory=dict)


class HTTPDispatcher:
    """Submit / poll / fetch against a downstream service over HTTP.

    Each instance is bound to one service's VPC HTTP trigger URL
    (``http_base_url``) — typically ``https://fc-<name>-<id>.cn-hangzhou-vpc.fcapp.run``,
    but any HTTP base that speaks the bioagent service framework's
    ``/api/tasks/*`` + ``/api/jobs/*`` contract works (LocalDispatcher,
    K8s ingress, etc.).
    """

    backend_name = "http"

    def __init__(
        self,
        *,
        http_base_url: str,
        function: str = "",
        timeout_seconds: float = 60.0,
    ) -> None:
        """
        :param http_base_url: e.g. ``"https://fc-esmfold-xxx.cn-hangzhou-vpc.fcapp.run"``
        :param function: cosmetic label (recorded in DispatchHandle.backend_ref
            for diagnostic continuity with FCDispatcher).  Not used for routing.
        :param timeout_seconds: per-request httpx timeout for submit + poll.
        """
        self.http_base_url = http_base_url.rstrip("/")
        self.function = function
        self._timeout = timeout_seconds

    # ------------------------------------------------------------------
    # submit
    # ------------------------------------------------------------------

    def submit(
        self,
        *,
        task_id: str,
        endpoint: str,
        payload: dict[str, Any],
        files: dict[str, Path | list[Path]] | None = None,
    ) -> DispatchHandle:
        """Async-invoke ``POST {base}{endpoint}`` via FC's HTTP trigger."""
        file_payload: list[tuple[str, tuple[str, bytes, str]]] = []
        if files:
            for field_name, val in files.items():
                paths = val if isinstance(val, list) else [val]
                for p in paths:
                    file_payload.append(
                        (field_name, (p.name, p.read_bytes(), "application/octet-stream"))
                    )

        headers = {
            "X-Fc-Invocation-Type": "Async",
            "X-Fc-Async-Task-Id": task_id,
            "X-Bioagent-Job-Id": task_id,
        }
        url = f"{self.http_base_url}{endpoint}"
        data = {k: str(v) for k, v in payload.items()}

        with httpx.Client(timeout=self._timeout) as cli:
            try:
                resp = cli.post(
                    url,
                    data=data,
                    files=file_payload or None,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                raise RuntimeError(f"submit failed for {url!r}: {exc}") from exc

        # FC's async-task dedup returns 409 if the same X-Fc-Async-Task-Id is
        # submitted twice while the previous invocation is still tracked.
        # Treat as idempotent success so callers can safely retry on a flaky
        # client-side network.
        if resp.status_code == 409:
            logger.info(
                "FC async dedup: task %s already exists (HTTP 409); "
                "returning handle for existing task", task_id,
            )
            return DispatchHandle(
                backend=self.backend_name,
                task_id=task_id,
                backend_ref={
                    "function": self.function,
                    "invocation_id": task_id,
                    "deduped": True,
                },
            )
        if resp.status_code != 202:
            raise RuntimeError(
                f"async submit to {url!r} returned {resp.status_code}, "
                f"expected 202; body={resp.text!r}"
            )

        invocation_id = (
            resp.headers.get("x-fc-request-id")
            or resp.headers.get("X-Fc-Request-Id")
            or task_id
        )
        return DispatchHandle(
            backend=self.backend_name,
            task_id=task_id,
            backend_ref={"function": self.function, "invocation_id": invocation_id},
        )

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def get_status(self, handle: DispatchHandle) -> TaskStatus:
        """Read ``status`` from ``GET /api/jobs/<task_id>`` JobInfo.

        Unknown status strings are mapped to RUNNING (treat as still in-flight)
        rather than FAILED, so a transient schema mismatch doesn't poison a
        live job's record.
        """
        url = f"{self.http_base_url}/api/jobs/{handle.task_id}"
        with httpx.Client(timeout=self._timeout) as cli:
            resp = cli.get(url)
        if resp.status_code == 404:
            # Sub-task hasn't been registered yet (rare race between submit's
            # 202 and the in-function handler creating JobInfo) — treat as
            # still pending.
            return TaskStatus.PENDING
        resp.raise_for_status()
        body = resp.json()
        status_str = (body.get("status") or "").lower()
        return _JOBINFO_STATUS_MAP.get(status_str, TaskStatus.RUNNING)

    # ------------------------------------------------------------------
    # fetch
    # ------------------------------------------------------------------

    def fetch_result(self, handle: DispatchHandle, *, dest_dir: Path) -> Path:
        """Stream ``GET /api/jobs/<task_id>/download`` to ``dest_dir/<task_id>.zip``.

        Sends the session-affinity header so FC routes the download to the
        instance that ran the task (NAS-local read, no cold-start of a
        separate "polling instance").
        """
        url = f"{self.http_base_url}/api/jobs/{handle.task_id}/download"
        dest_dir.mkdir(parents=True, exist_ok=True)
        zip_path = dest_dir / f"{handle.task_id}.zip"
        headers = {SESSION_AFFINITY_HEADER: handle.task_id}
        with httpx.stream("GET", url, timeout=300.0, headers=headers) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return zip_path

    # ------------------------------------------------------------------
    # stop
    # ------------------------------------------------------------------

    def stop(self, handle: DispatchHandle) -> None:
        """Best-effort cancel.

        The bioagent service framework doesn't expose a public stop endpoint
        over HTTP (FC's StopAsyncTask is OpenAPI-only).  We log + no-op;
        FC will eventually reclaim the instance, and callers should rely on
        ``error_summary`` / timeouts to surface cancellation needs.
        """
        logger.info(
            "HTTPDispatcher.stop is a no-op (task_id=%s, function=%s); "
            "FC instance will be reclaimed on its own",
            handle.task_id, self.function,
        )


__all__ = [
    "DispatchHandle",
    "HTTPDispatcher",
    "TaskStatus",
    "SESSION_AFFINITY_HEADER",
]
