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
    from bioagent_service import JobAdapter, ServiceSettings, create_app

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

from bioagent_service.adapter import JobAdapter
from bioagent_service.app import attach_mcp, create_app
from bioagent_service.errors import FailureKind, extract_error_summary, finalize_job
from bioagent_service.forms import model_form_depends
from bioagent_service.manifest import EndpointExample, ServiceManifest
from bioagent_service.models import JobInfo, JobStatus
from bioagent_service.settings import ServiceSettings

__all__ = [
    "EndpointExample",
    "JobAdapter",
    "JobInfo",
    "JobStatus",
    "FailureKind",
    "ServiceManifest",
    "ServiceSettings",
    "attach_mcp",
    "create_app",
    "extract_error_summary",
    "finalize_job",
    "model_form_depends",
]
