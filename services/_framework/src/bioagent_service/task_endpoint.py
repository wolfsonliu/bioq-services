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
    raise NotImplementedError("see Task 1.3")


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
    raise NotImplementedError("see Task 1.3")
