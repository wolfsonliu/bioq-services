"""gateway-server FastAPI app: /healthz (framework) + /v1/* (gateway API)."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import anyio
import httpx
from bioagent_service import attach_mcp, create_app, read_version_file
from fastapi import Body, Depends, Header, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from starlette.background import BackgroundTask

from .adapter import GatewayAdapter
from .auth.deps import AuthIdentity, require_auth
from .db.store import GatewayDB
from .discover import Discovery
from .dispatch import HttpDispatch
from .fc_status import FcStatusClient
from .models import JobView, PresignRequest, PresignResponse
from .oss_map import map_oss_inputs_to_mount
from .presign import Presigner, build_oss_client
from .registry import ServiceRegistry
from .settings import GatewaySettings

logger = logging.getLogger(__name__)

settings = GatewaySettings()
adapter = GatewayAdapter(settings=settings)

app = create_app(
    adapter, settings,
    title="Gateway Server",
    version=read_version_file(__file__, default="0.0.1"),
)

# --- wire gateway state ---
# Schema is managed by Alembic (`alembic upgrade head`, run by the container
# entrypoint before uvicorn) — NOT create_all() here, so multi-instance startup
# can't race on DDL and schema changes are versioned. Tests bootstrap via
# GatewayDB.create_all() in their fixture.
_db = GatewayDB(settings.db_url)
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


@app.on_event("startup")
async def _raise_thread_pool_limit() -> None:
    # All /v1 handlers are sync `def` → FastAPI runs them in anyio's default
    # threadpool (default 40 tokens). Gateway requests are I/O-bound proxy work,
    # so raise the ceiling to allow more concurrent in-flight requests. Must run
    # inside the event loop (the limiter is a per-loop RunVar).
    anyio.to_thread.current_default_thread_limiter().total_tokens = settings.thread_pool_size


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


# `endpoint:path` so nested downstream task endpoints route through the gateway
# (e.g. rfdiffusion "generate/motif", ppiflow "sample/binder", genie3
# "generate/unconditional"). It is forwarded verbatim to POST /api/tasks/<endpoint>.
@app.post("/v1/run/{svc}/{endpoint:path}", status_code=202)
def run(svc: str, endpoint: str, request: Request,
        body: dict = Body(default_factory=dict),
        x_bioagent_job_id: str | None = Header(default=None, alias="X-Bioagent-Job-Id"),
        ident: AuthIdentity = Depends(require_auth)) -> dict:
    reg = request.app.state.registry
    try:
        base = reg.base_url(svc)
    except KeyError:
        raise HTTPException(404, f"unknown service {svc!r}")
    # job_id is client-supplied (CLI-generated) and only needs to be unique
    # WITHIN this principal — the DB keys jobs by (principal, job_id).
    job_id = x_bioagent_job_id or uuid.uuid4().hex[:20]
    db = request.app.state.db
    if db.get_job(ident.principal, job_id) is not None:
        raise HTTPException(409, f"job {job_id!r} already exists")
    oss_prefix = f"users/{ident.principal}/{job_id}/"
    # Downstream identity must be globally unique: FC dedups async tasks by
    # X-Fc-Async-Task-Id per function, and the downstream NAS job dir is shared
    # across all users. Derive it from principal so two users' identical job_ids
    # never collide there. See design doc §5.
    fc_task_id = f"{ident.principal}-{job_id}"
    rec = reg.record(svc)
    if rec.oss_mount:
        body = map_oss_inputs_to_mount(
            body, bucket=settings.oss_bucket, mount=settings.downstream_oss_mount
        )
    db.create_user(ident.principal)
    db.create_job(job_id=job_id, principal=ident.principal, svc=svc, endpoint=endpoint,
                  input_params=body, output_prefix=oss_prefix)
    try:
        request.app.state.dispatch.submit(base, endpoint, fc_task_id, body, oss_prefix=oss_prefix)
    except Exception as exc:  # noqa: BLE001
        db.update_job(ident.principal, job_id, status="failed")
        raise HTTPException(502, f"dispatch failed: {exc}")
    db.update_job(ident.principal, job_id, status="running", fc_task_id=fc_task_id)
    return {"job_id": job_id, "status": "running"}


def _owned_job(request: Request, job_id: str, ident: AuthIdentity):
    # Scoped by (principal, job_id): a caller can only ever see jobs under their
    # own principal, so cross-user job_id reuse is invisible + safe.
    job = request.app.state.db.get_job(ident.principal, job_id)
    if job is None:
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
                request.app.state.db.update_job(ident.principal, job_id, status=new_status)
                job = request.app.state.db.get_job(ident.principal, job_id)
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
    except Exception as exc:  # noqa: BLE001 — OSS not configured / down => fall back to proxy
        # `presign_get_if_exists` returns None for a genuinely-absent object; an
        # exception here means a real OSS problem (auth/5xx/transport). We still
        # fall back to proxying the downstream (resilience), but log it — otherwise
        # an OSS outage silently masquerades as "results just not on OSS".
        logger.warning("presign GET failed for %s/%s (OSS error, proxying downstream): %r",
                       ident.principal, job_id, exc)
        url = None
    if url:
        # Fast path: object is on OSS → 302 to a presigned GET. Returns instantly
        # and the client pulls bytes straight from OSS, so no gateway thread is
        # held streaming the payload. This is the path all migrated services take.
        return RedirectResponse(url, status_code=302)
    # Fallback (rare — un-migrated service or a job with no results.zip on OSS):
    # proxy the downstream zip. This buffers to gateway disk and holds a thread,
    # so clean the temp file up after the response is sent (persistent gateway).
    dest = Path(settings.jobs_base_dir) / job_id / f"{job_id}.zip"
    try:
        request.app.state.dispatch.download(reg.base_url(job.svc), job.fc_task_id or job_id, dest)
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.text[:200]
        except Exception:  # noqa: BLE001
            detail = "<unavailable>"
        raise HTTPException(
            502,
            f"results.zip not on OSS and downstream download returned "
            f"HTTP {exc.response.status_code}: {detail}",
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            502,
            f"results.zip not on OSS and proxy fallback errored: {type(exc).__name__}: {exc}",
        )
    return FileResponse(
        str(dest), filename=f"{job_id}.zip", media_type="application/zip",
        background=BackgroundTask(_unlink_quiet, dest),
    )


def _unlink_quiet(path: Path) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


@app.post("/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request,
               ident: AuthIdentity = Depends(require_auth)) -> dict:
    _owned_job(request, job_id, ident)
    request.app.state.db.update_job(ident.principal, job_id, status="cancelled")  # MVP: local mark only
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
