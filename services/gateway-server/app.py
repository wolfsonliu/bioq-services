"""gateway-server FastAPI app: /healthz (framework) + /v1/* (gateway API)."""

from __future__ import annotations

import uuid
from pathlib import Path

import httpx
from bioagent_service import attach_mcp, create_app, read_version_file
from fastapi import Body, Depends, Header, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse

from .adapter import GatewayAdapter
from .auth.deps import AuthIdentity, require_auth
from .db.store import GatewayDB
from .discover import Discovery
from .dispatch import HttpDispatch
from .fc_status import FcStatusClient
from .models import JobView, PresignRequest, PresignResponse
from .presign import Presigner, build_oss_client
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
app.state.fc_status = FcStatusClient(
    access_key_id=settings.ali_access_key_id,
    access_key_secret=settings.ali_access_key_secret,
    default_region=settings.oss_region,
    endpoint=settings.fc_endpoint,
)


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
        x_bioagent_job_id: str | None = Header(default=None, alias="X-Bioagent-Job-Id"),
        ident: AuthIdentity = Depends(require_auth)) -> dict:
    reg = request.app.state.registry
    try:
        base = reg.base_url(svc)
    except KeyError:
        raise HTTPException(404, f"unknown service {svc!r}")
    job_id = x_bioagent_job_id or uuid.uuid4().hex[:20]
    db = request.app.state.db
    if db.get_job(job_id) is not None:
        raise HTTPException(409, f"job {job_id!r} already exists")
    oss_prefix = f"users/{ident.principal}/{job_id}/"
    db.create_user(ident.principal)
    db.create_job(job_id=job_id, principal=ident.principal, svc=svc, endpoint=endpoint,
                  input_params=body, output_prefix=oss_prefix)
    try:
        request.app.state.dispatch.submit(base, endpoint, job_id, body, oss_prefix=oss_prefix)
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
    fc = request.app.state.fc_status
    task_id = job.fc_task_id or job_id
    detail = None
    try:
        rec = reg.record(job.svc)
    except KeyError:
        rec = None
    if rec is not None:
        try:
            if rec.function and fc.enabled:
                # Control-plane status — spins no downstream instance.
                new_status = fc.get_status(
                    function=rec.function, task_id=task_id, region=rec.region
                )
            else:
                # Fallback: HTTP poll (services without a function name / no AK/SK).
                down = request.app.state.dispatch.status(rec.url, task_id)
                new_status = down.get("status", job.status)
            if new_status != job.status:
                request.app.state.db.update_job(job_id, status=new_status)
                job = request.app.state.db.get_job(job_id)
        except httpx.HTTPStatusError as exc:
            # Surface (don't silently swallow) why the status refresh failed.
            detail = f"downstream status refresh: HTTP {exc.response.status_code}"
        except Exception as exc:  # noqa: BLE001 — transient network/SDK issues
            detail = f"status refresh error: {type(exc).__name__}: {exc}"
    return JobView(job_id=job.job_id, principal=job.principal, svc=job.svc,
                   endpoint=job.endpoint, status=job.status,
                   output_prefix=job.output_prefix, detail=detail)


@app.get("/v1/jobs/{job_id}/download")
def download_job(job_id: str, request: Request,
                 ident: AuthIdentity = Depends(require_auth)):
    job = _owned_job(request, job_id, ident)
    reg = request.app.state.registry
    try:
        url = _get_presigner(request).presign_get_if_exists(ident.principal, job_id, "results.zip")
    except Exception:  # noqa: BLE001 — OSS not configured / transient => fall back
        url = None
    if url:
        return RedirectResponse(url, status_code=302)
    dest = Path(settings.jobs_base_dir) / job_id / f"{job_id}.zip"
    try:
        request.app.state.dispatch.download(reg.base_url(job.svc), job.fc_task_id or job_id, dest)
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.text[:200]
        except Exception:  # noqa: BLE001
            detail = "<unavailable>"
        raise HTTPException(502, f"downstream download HTTP {exc.response.status_code}: {detail}")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"download failed: {type(exc).__name__}: {exc}")
    return FileResponse(str(dest), filename=f"{job_id}.zip", media_type="application/zip")


@app.post("/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request,
               ident: AuthIdentity = Depends(require_auth)) -> dict:
    _owned_job(request, job_id, ident)
    request.app.state.db.update_job(job_id, status="cancelled")  # MVP: local mark only
    return {"job_id": job_id, "status": "cancelled"}


def _get_presigner(request: Request) -> Presigner:
    p = getattr(request.app.state, "presigner", None)
    if p is None:
        s = request.app.state.settings
        p = Presigner(client=build_oss_client(s.oss_region),
                      bucket=s.oss_bucket, region=s.oss_region,
                      expiry_sec=s.presign_expiry_sec)
        request.app.state.presigner = p
    return p


@app.post("/v1/uploads/presign", response_model=PresignResponse)
def presign_upload(request: Request, body: PresignRequest,
                   ident: AuthIdentity = Depends(require_auth)) -> PresignResponse:
    return _get_presigner(request).presign_put(
        ident.principal, body.job_id, body.filename, body.sha256
    )


attach_mcp(app)
