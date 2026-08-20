"""FastAPI app for seqkit-server.

Exposes `/api/stats` + `/api/revcomp` (submit/poll) and their
`/api/tasks/*` async variants. Job lifecycle endpoints (/healthz, /api/jobs/*,
/api/manifest, /openapi.json) come from `bioq_service.create_app`.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from bioq_service import (
    JobInfo,
    attach_mcp,
    create_app,
    execute_task,
    model_form_depends,
    read_version_file,
    resolve_task_id,
)
from bioq_service.uris import resolve_input
from fastapi import Depends, File, Form, Header, Request, UploadFile

from .adapter import SeqkitAdapter
from .models import RevcompRequest, StatsRequest
from .settings import SeqkitSettings
from .tools import revcomp_argv, stats_argv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = SeqkitSettings()
adapter = SeqkitAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="SeqKit Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---- /healthz/detail override: report seqkit binary readiness ----

def _strip_route(router, path: str, method: str) -> None:
    router.routes = [
        r for r in router.routes
        if not (getattr(r, "path", None) == path
                and method in getattr(r, "methods", set()))
    ]
    for r in router.routes:
        inner = getattr(r, "original_router", None)
        if inner is not None:
            _strip_route(inner, path, method)


_strip_route(app.router, "/healthz/detail", "GET")


@app.get("/healthz/detail")
def healthz_detail(request: Request) -> dict:
    """Extended health. SeqKit has no model weights, so this probes the
    vendored binary: it must exist, be executable, and answer `version`.
    Missing pieces do not crash the service; they surface as ready=false so an
    agent can detect a broken image before spending a job on it.
    """
    bin_path = settings.bin
    checks = {
        "bin_exists": bin_path.exists(),
        "bin_executable": bin_path.is_file() and _runs(bin_path),
    }
    missing = {k: str(k) for k, ok in checks.items() if not ok}
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "seqkit_bin": str(bin_path),
        "checks": checks,
        "ready": all(checks.values()),
        "missing": missing,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


def _runs(bin_path: Path) -> bool:
    try:
        proc = subprocess.run(
            [str(bin_path), "version"],
            capture_output=True, timeout=10, check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


# ---- stats ----

@app.post("/api/stats", response_model=JobInfo)
def post_stats(
    params: StatsRequest = Depends(model_form_depends(StatsRequest)),
    input_fasta: UploadFile | None = File(None),
    input_fasta_uri: str | None = Form(None),
) -> JobInfo:
    """Summary statistics for one FASTA/FASTQ file."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        fasta_path = resolve_input(
            input_fasta, input_fasta_uri, job_dir / "input" / "input.fasta", settings,
            field_name="input_fasta",
        )
        return stats_argv(params, job_dir=job_dir, input_fasta=fasta_path, settings=settings)

    return app.state.runner.submit(
        build_argv=_build, label="stats",
        input_params=params.model_dump(mode="json"),
    )


# ---- revcomp ----

@app.post("/api/revcomp", response_model=JobInfo)
def post_revcomp(
    params: RevcompRequest = Depends(model_form_depends(RevcompRequest)),
    input_fasta: UploadFile | None = File(None),
    input_fasta_uri: str | None = Form(None),
) -> JobInfo:
    """Reverse-complement every record in one FASTA/FASTQ file."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        fasta_path = resolve_input(
            input_fasta, input_fasta_uri, job_dir / "input" / "input.fasta", settings,
            field_name="input_fasta",
        )
        return revcomp_argv(params, job_dir=job_dir, input_fasta=fasta_path, settings=settings)

    return app.state.runner.submit(
        build_argv=_build, label="revcomp",
        input_params=params.model_dump(mode="json"),
    )


# ---- task endpoints (FC async task mode) ----

if settings.task_endpoints_enabled:

    @app.post("/api/tasks/stats", response_model=JobInfo)
    def post_stats_task(
        request: Request,
        params: StatsRequest = Depends(model_form_depends(StatsRequest)),
        input_fasta: UploadFile | None = File(None),
        input_fasta_uri: str | None = Form(None),
        x_bioagent_job_id: str | None = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: str | None = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """stats as a single atomic task (blocks until completion)."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["fasta"] = resolve_input(
                input_fasta, input_fasta_uri, input_dir / "input.fasta", settings,
                field_name="input_fasta",
            )

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return stats_argv(req, job_dir=job_dir, input_fasta=paths["fasta"], settings=settings)

        return execute_task(
            request, job_id=job_id, label="stats", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/revcomp", response_model=JobInfo)
    def post_revcomp_task(
        request: Request,
        params: RevcompRequest = Depends(model_form_depends(RevcompRequest)),
        input_fasta: UploadFile | None = File(None),
        input_fasta_uri: str | None = Form(None),
        x_bioagent_job_id: str | None = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: str | None = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """revcomp as a single atomic task (blocks until completion)."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["fasta"] = resolve_input(
                input_fasta, input_fasta_uri, input_dir / "input.fasta", settings,
                field_name="input_fasta",
            )

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return revcomp_argv(req, job_dir=job_dir, input_fasta=paths["fasta"], settings=settings)

        return execute_task(
            request, job_id=job_id, label="revcomp", params=params,
            build_argv=_build, save_inputs=_save,
        )


# Mount MCP — after all POST routes so auto-discovery sees the full surface.
attach_mcp(app)
