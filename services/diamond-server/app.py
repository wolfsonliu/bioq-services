"""FastAPI app for diamond-server.

Exposes /api/blastp, /api/blastx, /api/cluster, /api/msa (+ /api/tasks/* async
variants). makedb is CLI/SIF-only (see __main__.py). Job lifecycle endpoints
(/healthz, /api/jobs/*, /api/manifest, /openapi.json) come from
`bioq_service.create_app`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from bioq_service import (
    JobInfo,
    attach_mcp,
    create_app,
    default_semantics,
    execute_task,
    model_form_depends,
    read_version_file,
    resolve_task_id,
)
from bioq_service.uris import resolve_input, resolve_uri
from fastapi import Depends, File, Form, Header, HTTPException, Request, UploadFile

from .adapter import DiamondAdapter
from .models import BlastpRequest, BlastxRequest, ClusterRequest, MsaRequest
from .settings import DiamondSettings
from .tools import blastp_argv, blastx_argv, cluster_argv, msa_argv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = DiamondSettings()
adapter = DiamondAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="DIAMOND Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---- /healthz/detail override: report reference DB reachability ----

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
    """Extended health: whether the NAS-mounted reference DB dir + default MSA DB
    are present. Missing DB does not crash the service (blastp/blastx can still
    build inline from an uploaded subject); it only disables the default MSA DB.
    """
    expected: dict[str, Path] = {"db_dir": settings.db_dir}
    if settings.msa_db:
        expected["msa_db"] = settings.db_dir / settings.msa_db
    missing = {k: str(p) for k, p in expected.items() if not p.exists()}
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "db_dir": str(settings.db_dir),
        "msa_db": settings.msa_db,
        "db_loaded": not missing,
        "db_missing": missing,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


# ---- shared DB resolution ----

def _resolve_search_db(
    *,
    db_uri: Optional[str],
    subject: Optional[UploadFile],
    subject_uri: Optional[str],
    job_dir: Path,
) -> tuple[Optional[Path], Optional[Path]]:
    """Return (db_path, subject_path) for a blastp/blastx call.

    Exactly one of `db_uri` or a subject (upload/uri) must be given. The driver
    builds the DB inline from a subject; a `db_uri` is resolved to a `.dmnd`.
    """
    has_subject = bool(subject and getattr(subject, "filename", None)) or bool(subject_uri)
    if db_uri and has_subject:
        raise HTTPException(422, "Provide either db_uri OR a subject FASTA, not both.")
    if db_uri:
        db_path = resolve_uri(db_uri, job_dir / "db" / "ref.dmnd", settings)
        return db_path, None
    if has_subject:
        subject_path = resolve_input(
            subject, subject_uri, job_dir / "input" / "subject.faa", settings,
            field_name="subject",
        )
        return None, subject_path
    raise HTTPException(
        422, "A reference is required: provide db_uri (.dmnd) or a subject FASTA.",
    )


def _resolve_msa_db(*, db_uri: Optional[str], job_dir: Path) -> Path:
    if db_uri:
        return resolve_uri(db_uri, job_dir / "db" / "ref.dmnd", settings)
    if settings.msa_db:
        return settings.db_dir / settings.msa_db
    raise HTTPException(
        422, "No reference DB: pass db_uri (.dmnd) or configure DIAMOND_MSA_DB.",
    )


# ---- blastp ----

@app.post("/api/blastp", response_model=JobInfo)
def post_blastp(
    params: BlastpRequest = Depends(model_form_depends(BlastpRequest)),
    query: Optional[UploadFile] = File(None),
    query_uri: Optional[str] = Form(None),
    subject: Optional[UploadFile] = File(None),
    subject_uri: Optional[str] = Form(None),
    db_uri: Optional[str] = Form(None, json_schema_extra=default_semantics("unset", "only used when explicitly provided")),
) -> JobInfo:
    """Align a protein query FASTA against a protein DB."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        query_path = resolve_input(query, query_uri, job_dir / "input" / "query.faa", settings, field_name="query")
        db_path, subject_path = _resolve_search_db(
            db_uri=db_uri, subject=subject, subject_uri=subject_uri, job_dir=job_dir,
        )
        return blastp_argv(
            params, job_dir=job_dir, query_path=query_path,
            db_path=db_path, subject_path=subject_path, settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build, label="blastp",
        input_params=params.model_dump(mode="json"),
    )


# ---- blastx ----

@app.post("/api/blastx", response_model=JobInfo)
def post_blastx(
    params: BlastxRequest = Depends(model_form_depends(BlastxRequest)),
    query: Optional[UploadFile] = File(None),
    query_uri: Optional[str] = Form(None),
    subject: Optional[UploadFile] = File(None),
    subject_uri: Optional[str] = Form(None),
    db_uri: Optional[str] = Form(None, json_schema_extra=default_semantics("unset", "only used when explicitly provided")),
) -> JobInfo:
    """Align a translated-DNA query FASTA against a protein DB."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        query_path = resolve_input(query, query_uri, job_dir / "input" / "query.fna", settings, field_name="query")
        db_path, subject_path = _resolve_search_db(
            db_uri=db_uri, subject=subject, subject_uri=subject_uri, job_dir=job_dir,
        )
        return blastx_argv(
            params, job_dir=job_dir, query_path=query_path,
            db_path=db_path, subject_path=subject_path, settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build, label="blastx",
        input_params=params.model_dump(mode="json"),
    )


# ---- cluster ----

@app.post("/api/cluster", response_model=JobInfo)
def post_cluster(
    params: ClusterRequest = Depends(model_form_depends(ClusterRequest)),
    sequences: Optional[UploadFile] = File(None),
    sequences_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Cluster a protein FASTA (cluster / deepclust / linclust)."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        seq_path = resolve_input(
            sequences, sequences_uri, job_dir / "input" / "sequences.faa", settings,
            field_name="sequences",
        )
        return cluster_argv(params, job_dir=job_dir, sequences_path=seq_path, settings=settings)

    return app.state.runner.submit(
        build_argv=_build, label="cluster",
        input_params=params.model_dump(mode="json"),
    )


# ---- msa ----

@app.post("/api/msa", response_model=JobInfo)
def post_msa(
    params: MsaRequest = Depends(model_form_depends(MsaRequest)),
    query: Optional[UploadFile] = File(None),
    query_uri: Optional[str] = Form(None),
    db_uri: Optional[str] = Form(None, json_schema_extra=default_semantics("unset", "only used when explicitly provided")),
) -> JobInfo:
    """Build a query-anchored a3m MSA via blastp against a reference DB."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        query_path = resolve_input(query, query_uri, job_dir / "input" / "query.faa", settings, field_name="query")
        db_path = _resolve_msa_db(db_uri=db_uri, job_dir=job_dir)
        return msa_argv(params, job_dir=job_dir, query_path=query_path, db_path=db_path, settings=settings)

    return app.state.runner.submit(
        build_argv=_build, label="msa",
        input_params=params.model_dump(mode="json"),
    )


# ---- task endpoints (FC async task mode) ----

if settings.task_endpoints_enabled:

    @app.post("/api/tasks/blastp", response_model=JobInfo)
    def post_blastp_task(
        request: Request,
        params: BlastpRequest = Depends(model_form_depends(BlastpRequest)),
        query: Optional[UploadFile] = File(None),
        query_uri: Optional[str] = Form(None),
        subject: Optional[UploadFile] = File(None),
        subject_uri: Optional[str] = Form(None),
        db_uri: Optional[str] = Form(None, json_schema_extra=default_semantics("unset", "only used when explicitly provided")),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """blastp as a single atomic task (blocks until completion)."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Optional[Path]] = {}

        def _save(_req, input_dir: Path) -> None:
            job_dir = input_dir.parent
            paths["query"] = resolve_input(query, query_uri, input_dir / "query.faa", settings, field_name="query")
            paths["db"], paths["subject"] = _resolve_search_db(
                db_uri=db_uri, subject=subject, subject_uri=subject_uri, job_dir=job_dir,
            )

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return blastp_argv(
                req, job_dir=job_dir, query_path=paths["query"],
                db_path=paths["db"], subject_path=paths["subject"], settings=settings,
            )

        return execute_task(
            request, job_id=job_id, label="blastp", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/blastx", response_model=JobInfo)
    def post_blastx_task(
        request: Request,
        params: BlastxRequest = Depends(model_form_depends(BlastxRequest)),
        query: Optional[UploadFile] = File(None),
        query_uri: Optional[str] = Form(None),
        subject: Optional[UploadFile] = File(None),
        subject_uri: Optional[str] = Form(None),
        db_uri: Optional[str] = Form(None, json_schema_extra=default_semantics("unset", "only used when explicitly provided")),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """blastx as a single atomic task (blocks until completion)."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Optional[Path]] = {}

        def _save(_req, input_dir: Path) -> None:
            job_dir = input_dir.parent
            paths["query"] = resolve_input(query, query_uri, input_dir / "query.fna", settings, field_name="query")
            paths["db"], paths["subject"] = _resolve_search_db(
                db_uri=db_uri, subject=subject, subject_uri=subject_uri, job_dir=job_dir,
            )

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return blastx_argv(
                req, job_dir=job_dir, query_path=paths["query"],
                db_path=paths["db"], subject_path=paths["subject"], settings=settings,
            )

        return execute_task(
            request, job_id=job_id, label="blastx", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/cluster", response_model=JobInfo)
    def post_cluster_task(
        request: Request,
        params: ClusterRequest = Depends(model_form_depends(ClusterRequest)),
        sequences: Optional[UploadFile] = File(None),
        sequences_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """cluster as a single atomic task (blocks until completion)."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["sequences"] = resolve_input(
                sequences, sequences_uri, input_dir / "sequences.faa", settings,
                field_name="sequences",
            )

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return cluster_argv(req, job_dir=job_dir, sequences_path=paths["sequences"], settings=settings)

        return execute_task(
            request, job_id=job_id, label="cluster", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/msa", response_model=JobInfo)
    def post_msa_task(
        request: Request,
        params: MsaRequest = Depends(model_form_depends(MsaRequest)),
        query: Optional[UploadFile] = File(None),
        query_uri: Optional[str] = Form(None),
        db_uri: Optional[str] = Form(None, json_schema_extra=default_semantics("unset", "only used when explicitly provided")),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """msa as a single atomic task (blocks until completion)."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            job_dir = input_dir.parent
            paths["query"] = resolve_input(query, query_uri, input_dir / "query.faa", settings, field_name="query")
            paths["db"] = _resolve_msa_db(db_uri=db_uri, job_dir=job_dir)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return msa_argv(req, job_dir=job_dir, query_path=paths["query"], db_path=paths["db"], settings=settings)

        return execute_task(
            request, job_id=job_id, label="msa", params=params,
            build_argv=_build, save_inputs=_save,
        )


# Mount MCP — after all POST routes so auto-discovery sees the full surface.
attach_mcp(app)
