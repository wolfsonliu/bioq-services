"""Job status + structure download routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from ..auth.deps import require_api_key

router = APIRouter()


@router.get("/v1/jobs/{task_id}")
async def get_job(
    request: Request,
    task_id: str,
    api_key=Depends(require_api_key),
) -> dict:
    """Get current state of an ensemble job, refreshing sub-task status lazily."""
    orchestrator = request.app.state.orchestrator
    job = await orchestrator.refresh(task_id)
    if job is None or job.customer_id != api_key.customer_id:
        raise HTTPException(404, "job not found")
    return job.model_dump(mode="json")


@router.get("/v1/jobs/{task_id}/structures/{method}/{filename}")
async def download_structure(
    request: Request,
    task_id: str,
    method: str,
    filename: str,
    api_key=Depends(require_api_key),
) -> FileResponse:
    """Stream one structure file from a completed sub-task.

    Path traversal protection: filename can only be a basename (no '/' or '..').
    """
    if "/" in filename or ".." in filename or filename.startswith("."):
        raise HTTPException(400, "invalid filename")

    orchestrator = request.app.state.orchestrator
    job = await orchestrator.refresh(task_id)
    if job is None or job.customer_id != api_key.customer_id:
        raise HTTPException(404, "job not found")

    settings = request.app.state.settings
    target = settings.jobs_base_dir / task_id / "outputs" / method / filename
    if not target.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(
        target,
        media_type="application/octet-stream",
        filename=filename,
    )
