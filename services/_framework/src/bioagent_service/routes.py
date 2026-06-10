"""Generic HTTP routes mounted by `create_app`.

Each route operates on `request.app.state.{adapter, settings, job_store}`, so the
same router works for every service — no per-service codegen needed.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from bioagent_service.adapter import JobAdapter
from bioagent_service.downloads import archive_dir, list_files, safe_subpath
from bioagent_service.jobs import (
    JobStore,
    cleanup_job,
    disk_usage_bytes,
)
from bioagent_service.models import FailureKind, JobInfo, JobStatus, utcnow
from bioagent_service.settings import ServiceSettings

logger = logging.getLogger(__name__)


def make_generic_router() -> APIRouter:
    """Return the router with health + job-lifecycle endpoints.

    Endpoints read their per-app dependencies from `request.app.state`:
      - adapter   : JobAdapter
      - settings  : ServiceSettings
      - job_store : JobStore
    """
    router = APIRouter()

    def _adapter(request: Request) -> JobAdapter:
        return request.app.state.adapter

    def _settings(request: Request) -> ServiceSettings:
        return request.app.state.settings

    def _store(request: Request) -> JobStore:
        return request.app.state.job_store

    # ---------- Health ----------

    @router.get("/healthz")
    def health(request: Request) -> dict[str, str]:
        return {
            "status": "ok",
            "service": _adapter(request).name,
            "version": request.app.version,
        }

    @router.get("/healthz/detail")
    def health_detail(request: Request) -> dict[str, object]:
        settings = _settings(request)
        runner = request.app.state.runner
        return {
            "status": "ok",
            "service": _adapter(request).name,
            "version": request.app.version,
            "active_jobs": runner.active_job_count,
            "max_concurrent_jobs": settings.max_concurrent_jobs,
            "jobs_base_dir": str(settings.jobs_base_dir),
            "jobs_base_dir_exists": settings.jobs_base_dir.exists(),
            "disk_usage_mb": round(disk_usage_bytes(settings.jobs_base_dir) / 1024 / 1024, 2),
            "disk_limit_mb": settings.disk_limit_mb,
            "session_header": settings.session_header_name,
        }

    # ---------- FC lifecycle ----------

    @router.get("/pre-stop")
    def pre_stop(request: Request) -> dict[str, object]:
        """FC PreStop hook — mark this instance's running jobs as interrupted."""
        store = _store(request)
        instance_id = store.instance_id
        interrupted: list[str] = []
        for job in store.all_jobs():
            if job.status != JobStatus.RUNNING:
                continue
            if job.instance_id is not None and job.instance_id != instance_id:
                continue
            now = utcnow()
            store.update(
                job.job_id,
                status=JobStatus.FAILED,
                message="FC PreStop: instance shutting down",
                failure_kind=FailureKind.INTERRUPTED,
                completed_at=now,
                duration_seconds=(
                    (now - job.started_at).total_seconds()
                    if job.started_at else None
                ),
            )
            interrupted.append(job.job_id)
        if interrupted:
            logger.warning(
                "pre-stop interrupted %d job(s): %s (instance=%s)",
                len(interrupted), interrupted, instance_id,
            )
        else:
            logger.info("pre-stop: no running jobs to interrupt (instance=%s)", instance_id)
        return {"status": "ok", "interrupted_jobs": interrupted}

    # ---------- Job lifecycle (read) ----------

    @router.get("/api/jobs/{job_id}", response_model=JobInfo)
    def get_job(request: Request, job_id: str) -> JobInfo:
        job = _store(request).get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        return job

    @router.get("/api/jobs/{job_id}/files")
    def list_job_files(request: Request, job_id: str) -> dict[str, object]:
        job = _store(request).get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        adapter = _adapter(request)
        return {
            "job_id": job_id,
            "files": list_files(adapter.output_dir(adapter.job_dir(job_id))),
        }

    @router.get("/api/jobs/{job_id}/log")
    def get_job_log(request: Request, job_id: str) -> dict[str, str]:
        job = _store(request).get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        adapter = _adapter(request)
        log_path = adapter.log_path(adapter.job_dir(job_id))
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        return {"job_id": job_id, "log": text}

    @router.get("/api/jobs/{job_id}/download")
    def download_job(request: Request, job_id: str) -> StreamingResponse:
        job = _store(request).get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"job is {job.status.value}, not completed",
            )
        adapter = _adapter(request)
        out_dir = adapter.output_dir(adapter.job_dir(job_id))
        files = list_files(out_dir)
        if not files:
            raise HTTPException(status_code=404, detail="no output files")
        buf = archive_dir(out_dir)
        filename = f"{adapter.name}_{job_id}.zip"
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    @router.get("/api/jobs/{job_id}/file/{file_path:path}")
    def download_single_file(
        request: Request, job_id: str, file_path: str
    ) -> FileResponse:
        job = _store(request).get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        adapter = _adapter(request)
        out_dir = adapter.output_dir(adapter.job_dir(job_id))
        try:
            requested = safe_subpath(out_dir, file_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if not requested.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(
            path=requested,
            media_type="application/octet-stream",
            filename=requested.name,
        )

    # ---------- Job lifecycle (delete) ----------

    @router.delete("/api/jobs/{job_id}")
    def delete_job(request: Request, job_id: str) -> dict[str, str]:
        job = _store(request).get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        cleanup_job(_store(request), _settings(request).jobs_base_dir, job_id)
        return {"status": "deleted", "job_id": job_id}

    return router


__all__ = ["make_generic_router"]
