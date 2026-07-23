"""bioagent service framework.

Reusable HTTP / job / error-handling layer shared by all bioagent algorithm services
(rfantibody-server, genie3-server, etc.). A new service implements:

  * a `JobAdapter` subclass (service-wide policy: name, log layout, output detection)
  * a `ServiceSettings` subclass (env-driven config)
  * one or more endpoint handlers that build their own argv and call `runner.submit`

The framework supplies the FastAPI app, job store (with sidecar persistence and
read-through cache for multi-instance FC), error extraction, upload/download
endpoints, OpenAPI schema, and a `JobRunner.submit` primitive.

Minimal example:

    from pathlib import Path
    from pydantic import BaseModel
    from pydantic_settings import SettingsConfigDict
    from bioq_service import JobAdapter, ServiceSettings, create_app

    class EchoRequest(BaseModel):
        message: str

    class EchoSettings(ServiceSettings):
        model_config = SettingsConfigDict(env_prefix="ECHO_")

    class EchoAdapter(JobAdapter):
        name = "echo"

    settings = EchoSettings()
    adapter = EchoAdapter(settings=settings)
    app = create_app(adapter, settings, title="Echo")

    @app.post("/api/echo")
    def echo(req: EchoRequest):
        def _build(job_id, job_dir):
            out = job_dir / "output"
            out.mkdir(exist_ok=True)
            return ["bash", "-c", f"echo {req.message!r} > {out}/out.txt"]
        return app.state.runner.submit(build_argv=_build, label="echo")
"""

from __future__ import annotations

from pathlib import Path

from bioq_service.adapter import JobAdapter
from bioq_service.app import attach_mcp, create_app
from bioq_service.cli import CLIEndpoint, create_cli
from bioq_service.errors import FailureKind, extract_error_summary, finalize_job
from bioq_service.forms import model_form_depends
from bioq_service.manifest import EndpointExample, ServiceManifest
from bioq_service.models import JobInfo, JobStatus, UploadInfo
from bioq_service.oss_export import mirror_job_dir_to_oss
from bioq_service.settings import ServiceSettings
from bioq_service.task_endpoint import execute_task, register_task_endpoint, resolve_task_id
from bioq_service.uris import (
    maybe_resolve_input,
    resolve_input,
    resolve_uri,
    save_upload,
)


def read_version_file(caller_file: str, default: str = "0.0.0") -> str:
    """Read a service's `VERSION` file (sibling of the caller's app.py).

    Each service's Docker image is tagged from `services/<svc>/VERSION` (see
    Makefile). This helper lets `app.py` read the same source of truth so the
    HTTP version (manifest + /healthz) cannot drift from the image tag.

    Strips a single leading `v` (Makefile tags are `v0.0.3`; HTTP/semver expects
    `0.0.3`). Returns `default` if the file is missing — e.g. in editable installs
    or unit tests where the layout differs.
    """
    version_path = Path(caller_file).resolve().parent / "VERSION"
    if not version_path.is_file():
        return default
    text = version_path.read_text(encoding="utf-8").strip()
    return text[1:] if text.startswith("v") else text


__all__ = [
    "CLIEndpoint",
    "EndpointExample",
    "JobAdapter",
    "JobInfo",
    "JobStatus",
    "FailureKind",
    "ServiceManifest",
    "ServiceSettings",
    "UploadInfo",
    "attach_mcp",
    "create_app",
    "create_cli",
    "execute_task",
    "extract_error_summary",
    "finalize_job",
    "maybe_resolve_input",
    "mirror_job_dir_to_oss",
    "model_form_depends",
    "read_version_file",
    "register_task_endpoint",
    "resolve_input",
    "resolve_task_id",
    "resolve_uri",
    "save_upload",
]
