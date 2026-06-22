"""Job status + structure download routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from ..auth.deps import AuthIdentity, require_auth

router = APIRouter()


@router.get("/v1/jobs/{task_id}")
async def get_job(
    request: Request,
    task_id: str,
    auth: AuthIdentity = Depends(require_auth),
) -> dict:
    """Get current state of an ensemble job, refreshing sub-task status lazily."""
    orchestrator = request.app.state.orchestrator
    job = await orchestrator.refresh(task_id)
    if job is None or job.customer_id != auth.customer_id:
        raise HTTPException(404, "job not found")
    return job.model_dump(mode="json")


@router.get("/v1/jobs/{task_id}/structures/{method}/{filename:path}")
async def download_structure(
    request: Request,
    task_id: str,
    method: str,
    filename: str,
    auth: AuthIdentity = Depends(require_auth),
) -> FileResponse:
    """Stream one structure file from a completed sub-task.

    Accepts multi-segment ``filename`` (e.g. ``predictions/input/input_model_0.cif``
    for boltz's nested output layout).  Path-traversal safety is enforced by
    resolving the target and checking it stays under
    ``<jobs_base_dir>/<task_id>/outputs/<method>/`` — symlinks pointing
    outside that root are rejected.
    """
    if not filename or filename.startswith("/") or ".." in filename.split("/"):
        raise HTTPException(400, "invalid filename")

    orchestrator = request.app.state.orchestrator
    job = await orchestrator.refresh(task_id)
    if job is None or job.customer_id != auth.customer_id:
        raise HTTPException(404, "job not found")

    settings = request.app.state.settings
    method_root = (settings.jobs_base_dir / task_id / "outputs" / method).resolve()
    target = (method_root / filename).resolve()
    try:
        target.relative_to(method_root)
    except ValueError:
        raise HTTPException(400, "invalid filename")
    if not target.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(
        target,
        media_type="application/octet-stream",
        filename=target.name,
    )
