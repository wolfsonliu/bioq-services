"""Task endpoint helpers — single HTTP request = one atomic task.

Unlike the submit/poll style (which returns immediately and runs subprocess
in a background ThreadPoolExecutor), a task endpoint blocks the HTTP request
thread until the subprocess completes.  This keeps the FC instance occupied
for the full computation lifetime so FC's idle-recycle logic doesn't kill
the subprocess mid-run.

Designed to be invoked via FC Async Task Mode (X-Fc-Invocation-Type: Async),
where the request is enqueued and dispatched by FC, but works just as well
as a plain synchronous HTTP endpoint for local/dev use.

**Two public APIs**:

  - `execute_task(...)` — the primary helper.  Services define their own
    `@app.post(...)` handler with whatever signature they need (file
    uploads, headers, etc.) and call `execute_task` to run the pipeline.

  - `register_task_endpoint(...)` — a convenience wrapper for the simple
    no-upload case.  Internally calls `execute_task`.  Skip this if your
    endpoint needs UploadFile / custom Form fields.

NOTE: this module deliberately does NOT use `from __future__ import
annotations`.  `register_task_endpoint` creates a FastAPI handler whose
`params` annotation is a runtime-supplied class; PEP 563 string annotations
would prevent FastAPI's `get_type_hints` from resolving that class.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import Depends, FastAPI, Header, Request
from pydantic import BaseModel

from bioagent_service.errors import finalize_job
from bioagent_service.forms import model_form_depends
from bioagent_service.models import JobInfo, JobStatus, utcnow
from bioagent_service.runner import SubprocessRunner

logger = logging.getLogger(__name__)


BuildArgvForTask = Callable[[Any, str, Path], list[str]]
"""Endpoint-supplied closure: (request_model, job_id, job_dir) → argv.

Differs from `BuildArgv` in runner.py by taking the parsed request model as
its first argument, so the same closure can be shared between the
submit/poll endpoint and the task endpoint without re-parsing form data.
"""


def execute_task(
    request: Request,
    *,
    job_id: str,
    label: str,
    params: BaseModel,
    build_argv: BuildArgvForTask,
    save_inputs: Optional[Callable[[BaseModel, Path], None]] = None,
) -> JobInfo:
    """Run one pipeline synchronously inside the request thread.

    Service-side endpoints with custom signatures (file uploads, special
    headers) call this directly after parsing their own form data.

    Honors duplicate-job semantics: if the JobStore already has an entry
    for `job_id`, returns the existing JobInfo without re-running.
    """
    store = request.app.state.job_store
    adapter = request.app.state.adapter
    settings = request.app.state.settings

    # Duplicate-job semantics: if a job with this ID already exists in the
    # store (e.g. FC retried an Async invocation), return its current state
    # without re-running.  Idempotent client retries are safe.
    existing = store.get(job_id)
    if existing is not None:
        logger.info("task %s already exists with status=%s; returning existing", job_id, existing.status)
        return existing

    # Allocate the job dir + create PENDING record.
    job_dir = adapter.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "output").mkdir(exist_ok=True)
    adapter.log_path(job_dir).parent.mkdir(parents=True, exist_ok=True)
    store.create(job_id=job_id, input_params=params.model_dump(mode="json"))

    # Persist uploaded inputs before subprocess starts.
    input_dir = job_dir / "input"
    input_dir.mkdir(exist_ok=True)
    if save_inputs is not None:
        save_inputs(params, input_dir)

    try:
        argv = build_argv(params, job_id, job_dir)
        if not argv:
            raise ValueError("build_argv returned an empty argv")
    except Exception as exc:
        logger.exception("build_argv failed for task %s", job_id)
        store.update(
            job_id,
            status=JobStatus.FAILED,
            message=f"{label} setup error",
            error_summary=str(exc),
            completed_at=utcnow(),
        )
        return store.get(job_id)  # type: ignore[return-value]

    env = adapter.subprocess_env()
    cwd = adapter.subprocess_cwd()
    log_path = adapter.log_path(job_dir)

    store.update(
        job_id,
        status=JobStatus.RUNNING,
        message=f"{label} running",
        started_at=utcnow(),
    )

    rc = SubprocessRunner.run(argv, log_path, env=env, cwd=cwd)

    finalize_job(
        store,
        adapter,
        job_id,
        rc,
        label,
        error_tail_chars=settings.error_tail_chars,
    )
    return store.get(job_id)  # type: ignore[return-value]


def register_task_endpoint(
    app: FastAPI,
    *,
    path: str,
    label: str,
    request_model: type[BaseModel],
    build_argv: BuildArgvForTask,
    save_inputs: Optional[Callable[[BaseModel, Path], None]] = None,
) -> None:
    """Register a POST `path` that runs one full pipeline synchronously.

    Convenience for the no-upload case.  Behavior:
      1. Parse form/multipart body into `request_model`.
      2. Read job_id from `settings.task_job_id_header` (also accepts
         `X-Fc-Async-Task-Id`).  Generate UUID if absent.
      3. Delegate to `execute_task`.

    For endpoints that need UploadFile or custom Form fields, define the
    handler yourself and call `execute_task` directly.

    Honors `settings.task_endpoints_enabled` — when False, no route is
    registered (useful in local/test settings).
    """
    settings = app.state.settings
    if not settings.task_endpoints_enabled:
        logger.info("task_endpoints_enabled=False; skipping %s", path)
        return

    primary_header = settings.task_job_id_header

    # NOTE: `params: request_model = Depends(...)` uses a closure-captured class
    # as the annotation.  This works ONLY because this module does NOT use
    # `from __future__ import annotations` — otherwise FastAPI's get_type_hints
    # would see the string "request_model" and fail to resolve it against the
    # module's globals.  Do not add that future import here.
    def _task_handler(
        request: Request,
        params: request_model = Depends(model_form_depends(request_model)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias=primary_header),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = x_bioagent_job_id or x_fc_async_task_id or uuid.uuid4().hex[:20]
        return execute_task(
            request,
            job_id=job_id,
            label=label,
            params=params,
            build_argv=build_argv,
            save_inputs=save_inputs,
        )

    app.add_api_route(
        path,
        _task_handler,
        methods=["POST"],
        response_model=JobInfo,
    )
    logger.info("registered task endpoint %s (label=%s)", path, label)
