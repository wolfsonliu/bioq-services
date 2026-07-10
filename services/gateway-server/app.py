"""gateway-server FastAPI app: /healthz (framework) + /v1/* (gateway API)."""

from __future__ import annotations

import uuid
from pathlib import Path

from bioagent_service import attach_mcp, create_app, read_version_file
from fastapi import Body, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from .adapter import GatewayAdapter
from .auth.deps import AuthIdentity, require_auth
from .db.store import GatewayDB
from .discover import Discovery
from .dispatch import HttpDispatch
from .models import JobView
from .registry import ServiceRegistry
from .settings import GatewaySettings

settings = GatewaySettings()
adapter = GatewayAdapter(settings=settings)

app = create_app(
    adapter, settings,
    title="Gateway Server",
    version=read_version_file(__file__, default="0.0.1"),
)

# --- wire gateway state ---
_db = GatewayDB(settings.db_url)
_db.create_all()
app.state.db = _db
app.state.registry = ServiceRegistry(settings.registry_path)
app.state.discover = Discovery(ttl_sec=300)
app.state.dispatch = HttpDispatch()


@app.get("/v1/services")
def list_services(request: Request, _: AuthIdentity = Depends(require_auth)) -> dict:
    return {"services": request.app.state.registry.list()}


@app.get("/v1/services/{svc}")
def describe_service(svc: str, request: Request,
                     _: AuthIdentity = Depends(require_auth)) -> dict:
    reg = request.app.state.registry
    try:
        base = reg.base_url(svc)
    except KeyError:
        raise HTTPException(404, f"unknown service {svc!r}")
    return request.app.state.discover.describe(svc, base)


@app.post("/v1/run/{svc}/{endpoint}", status_code=202)
def run(svc: str, endpoint: str, request: Request,
        body: dict = Body(default_factory=dict),
        ident: AuthIdentity = Depends(require_auth)) -> dict:
    reg = request.app.state.registry
    try:
        base = reg.base_url(svc)
    except KeyError:
        raise HTTPException(404, f"unknown service {svc!r}")
    job_id = uuid.uuid4().hex[:20]
    output_prefix = f"users/{ident.principal}/outputs/{job_id}/"
    db = request.app.state.db
    db.create_user(ident.principal)
    db.create_job(job_id=job_id, principal=ident.principal, svc=svc, endpoint=endpoint,
                  input_params=body, output_prefix=output_prefix)
    try:
        request.app.state.dispatch.submit(base, endpoint, job_id, body)
    except Exception as exc:  # noqa: BLE001
        db.update_job(job_id, status="failed")
        raise HTTPException(502, f"dispatch failed: {exc}")
    db.update_job(job_id, status="running", fc_task_id=job_id)
    return {"job_id": job_id, "status": "running"}


def _owned_job(request: Request, job_id: str, ident: AuthIdentity):
    job = request.app.state.db.get_job(job_id)
    if job is None or job.principal != ident.principal:
        raise HTTPException(404, "job not found")
    return job


@app.get("/v1/jobs/{job_id}", response_model=JobView)
def get_job(job_id: str, request: Request,
            ident: AuthIdentity = Depends(require_auth)) -> JobView:
    job = _owned_job(request, job_id, ident)
    reg = request.app.state.registry
    try:
        down = request.app.state.dispatch.status(reg.base_url(job.svc), job.fc_task_id or job_id)
        new_status = down.get("status", job.status)
        if new_status != job.status:
            request.app.state.db.update_job(job_id, status=new_status)
            job = request.app.state.db.get_job(job_id)
    except Exception:  # noqa: BLE001 — return last-known status on transient errors
        pass
    return JobView(job_id=job.job_id, principal=job.principal, svc=job.svc,
                   endpoint=job.endpoint, status=job.status, output_prefix=job.output_prefix)


@app.get("/v1/jobs/{job_id}/download")
def download_job(job_id: str, request: Request,
                 ident: AuthIdentity = Depends(require_auth)) -> FileResponse:
    job = _owned_job(request, job_id, ident)
    reg = request.app.state.registry
    dest = Path(settings.jobs_base_dir) / job_id / f"{job_id}.zip"
    request.app.state.dispatch.download(reg.base_url(job.svc), job.fc_task_id or job_id, dest)
    return FileResponse(str(dest), filename=f"{job_id}.zip", media_type="application/zip")


@app.post("/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request,
               ident: AuthIdentity = Depends(require_auth)) -> dict:
    _owned_job(request, job_id, ident)
    request.app.state.db.update_job(job_id, status="failed")  # MVP: mark cancelled locally
    return {"job_id": job_id, "status": "cancelled"}


attach_mcp(app)
