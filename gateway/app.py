"""gateway-server FastAPI app: /healthz (framework) + /v1/* (gateway API)."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import anyio
import httpx
from bioq_service import attach_mcp, create_app, read_version_file
from fastapi import Body, Depends, Header, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from starlette.background import BackgroundTask
from starlette.middleware.sessions import SessionMiddleware

from .admin.routes import make_templates, mount_admin_static
from .admin.routes import router as admin_router
from .adapter import GatewayAdapter
from .auth.deps import AuthIdentity, require_auth
from .config_validate import validate_settings
from .db.store import GatewayDB
from .discover import Discovery
from .dispatchers import make_dispatcher
from .models import JobView, PrepareUploadRequest, UploadTarget
from .oss_map import map_oss_inputs_to_mount
from .registry import ServiceRegistry
from .settings import GatewaySettings
from .storage import FileStorage, make_storage

logger = logging.getLogger(__name__)

settings = GatewaySettings()

# Fail fast on invalid config (before make_dispatcher, so our message wins).
_fatals, _warnings = validate_settings(settings)
for _w in _warnings:
    logger.warning("gateway config: %s", _w)
if _fatals:
    raise SystemExit("gateway config invalid:\n- " + "\n- ".join(_fatals))

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
app.state.discover = Discovery(
    ttl_sec=settings.discovery_ttl_sec,
    negative_ttl_sec=settings.discovery_negative_ttl_sec,
    connect_timeout_sec=settings.discovery_connect_timeout_sec,
    read_timeout_sec=settings.discovery_read_timeout_sec,
)
app.state.dispatch = make_dispatcher(settings)

# --- admin console (server-side rendered, terminal-style) ---
# Signed cookie session for browser login (API auth headers can't be carried by
# a browser navigation). Then mount the /admin pages + static assets.
app.add_middleware(SessionMiddleware, secret_key=settings.session_secret,
                   session_cookie="gw_admin", same_site="lax", https_only=False)
app.state.templates = make_templates()
app.include_router(admin_router)
mount_admin_static(app)


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
        rec = reg.record(svc)
    except KeyError:
        raise HTTPException(404, f"unknown service {svc!r}")
    # Static contract first: zero downstream calls, cold-start immune. Fall back
    # to live discovery only for services without committed manifests yet.
    manifest = reg.manifest(svc)
    if manifest is not None:
        return {
            "service": svc,
            "manifest": manifest,
            "openapi": reg.openapi(svc) or {},
            "status": "ok",
            "source": "registry",
        }
    base = request.app.state.dispatch.describe_base_url(rec)
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
        rec = reg.record(svc)
    except KeyError:
        raise HTTPException(404, f"unknown service {svc!r}")
    # job_id is client-supplied (CLI-generated) and only needs to be unique
    # WITHIN this account — the DB keys jobs by (account_id, job_id).
    job_id = x_bioagent_job_id or uuid.uuid4().hex[:20]
    db = request.app.state.db
    if db.get_job(ident.account_id, job_id) is not None:
        raise HTTPException(409, f"job {job_id!r} already exists")
    oss_prefix = f"users/{ident.account_id}/{job_id}/"
    # Downstream identity must be globally unique: FC dedups async tasks by
    # X-Fc-Async-Task-Id per function, and the downstream NAS job dir is shared
    # across all accounts. Derive it from account_id so two accounts' identical
    # job_ids never collide there. See design doc §5.
    fc_task_id = f"{ident.account_id}-{job_id}"
    if rec.oss_mount:
        body = map_oss_inputs_to_mount(
            body, bucket=settings.oss_bucket, mount=settings.downstream_oss_mount
        )
    db.create_user(ident.account_id)
    db.create_job(job_id=job_id, account_id=ident.account_id, svc=svc, endpoint=endpoint,
                  input_params=body, output_prefix=oss_prefix)
    try:
        downstream_id = request.app.state.dispatch.submit(
            rec, endpoint, fc_task_id, body, oss_prefix=oss_prefix
        )
    except Exception as exc:  # noqa: BLE001
        db.update_job(ident.account_id, job_id, status="failed")
        raise HTTPException(502, f"dispatch failed: {exc}")
    # Track whatever handle the backend uses for status/download: the FC task id
    # we passed (FCDispatcher returns None) or the worker-assigned job_id
    # (LocalHttpDispatcher).
    db.update_job(ident.account_id, job_id, status="running",
                  fc_task_id=downstream_id or fc_task_id)
    return {"job_id": job_id, "status": "running"}


def _owned_job(request: Request, job_id: str, ident: AuthIdentity):
    # Scoped by (account_id, job_id): a caller can only ever see jobs under their
    # own account, so cross-account job_id reuse is invisible + safe.
    job = request.app.state.db.get_job(ident.account_id, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.get("/v1/jobs/{job_id}", response_model=JobView)
def get_job(job_id: str, request: Request,
            ident: AuthIdentity = Depends(require_auth)) -> JobView:
    job = _owned_job(request, job_id, ident)
    reg = request.app.state.registry
    task_id = job.fc_task_id or job_id
    detail = None
    try:
        rec = reg.record(job.svc)
    except KeyError:
        rec = None
    if rec is not None:
        try:
            # Status source is backend-specific (FC GetAsyncTask vs HTTP poll),
            # resolved inside the dispatcher.
            down = request.app.state.dispatch.status(rec, task_id)
            new_status = down.get("status", job.status)
            if new_status != job.status:
                request.app.state.db.update_job(ident.account_id, job_id, status=new_status)
                job = request.app.state.db.get_job(ident.account_id, job_id)
        except httpx.HTTPStatusError as exc:
            # Surface (don't silently swallow) why the status refresh failed.
            detail = f"downstream status refresh: HTTP {exc.response.status_code}"
        except Exception as exc:  # noqa: BLE001 — transient network/SDK issues
            detail = f"status refresh error: {type(exc).__name__}: {exc}"
    return JobView(job_id=job.job_id, account_id=job.account_id, svc=job.svc,
                   endpoint=job.endpoint, status=job.status,
                   output_prefix=job.output_prefix, detail=detail)


@app.get("/v1/jobs/{job_id}/download")
def download_job(job_id: str, request: Request,
                 ident: AuthIdentity = Depends(require_auth)):
    job = _owned_job(request, job_id, ident)
    reg = request.app.state.registry
    try:
        url = _get_storage(request).result_url_if_exists(ident.account_id, job_id, "results.zip")
    except Exception as exc:  # noqa: BLE001 — OSS not configured / down => fall back to proxy
        # `result_url_if_exists` returns None for a genuinely-absent object; an
        # exception here means a real OSS problem (auth/5xx/transport). We still
        # fall back to proxying the downstream (resilience), but log it — otherwise
        # an OSS outage silently masquerades as "results just not on OSS".
        logger.warning("presign GET failed for %s/%s (OSS error, proxying downstream): %r",
                       ident.account_id, job_id, exc)
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
        rec = reg.record(job.svc)
        request.app.state.dispatch.download(rec, job.fc_task_id or job_id, dest)
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
    request.app.state.db.update_job(ident.account_id, job_id, status="cancelled")  # MVP: local mark only
    return {"job_id": job_id, "status": "cancelled"}


def _get_storage(request: Request):
    s = getattr(request.app.state, "storage", None)
    if s is None:
        s = make_storage(request.app.state.settings)
        request.app.state.storage = s
    return s


@app.post("/v1/uploads/prepare", response_model=UploadTarget)
def prepare_upload(request: Request, body: PrepareUploadRequest,
                   ident: AuthIdentity = Depends(require_auth)) -> UploadTarget:
    return _get_storage(request).prepare_upload(
        ident.account_id, body.job_id, body.filename, body.sha256
    )


def _file_storage_or_404(request: Request) -> FileStorage:
    storage = _get_storage(request)
    if not isinstance(storage, FileStorage):
        raise HTTPException(404, "file IO is only available with the 'file' storage backend")
    return storage


def _guard_key(key: str, ident: AuthIdentity) -> None:
    # Tenant isolation: a caller may only touch files under their own account.
    if not key.startswith(f"users/{ident.account_id}/"):
        raise HTTPException(403, "forbidden")


@app.put("/v1/files/{key:path}")
async def put_file(key: str, request: Request,
                   ident: AuthIdentity = Depends(require_auth)) -> dict:
    storage = _file_storage_or_404(request)
    _guard_key(key, ident)
    dest = storage.resolve(key)  # rejects traversal outside base_dir
    data = await request.body()
    # Blocking filesystem write off the event loop (local file backend only).
    await anyio.to_thread.run_sync(_write_bytes, dest, data)
    return {"ok": True, "key": key}


def _write_bytes(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


@app.get("/v1/files/{key:path}")
def get_file(key: str, request: Request,
             ident: AuthIdentity = Depends(require_auth)):
    storage = _file_storage_or_404(request)
    _guard_key(key, ident)
    path = storage.resolve(key)
    if not path.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(str(path))


attach_mcp(app)
