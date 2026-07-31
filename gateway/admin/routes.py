"""Admin console routes (server-side rendered). Login/logout/setlang are public;
page routes take `Depends(require_admin_web)` (redirect to /admin/login if not)."""

from __future__ import annotations

import secrets
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .auth import csrf_token, require_admin_web, verify_admin_key, verify_csrf
from .i18n import LANGS, lang_of, t

_HERE = Path(__file__).resolve().parent
TEMPLATES_DIR = _HERE / "templates"
STATIC_DIR = _HERE / "static"


def make_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.globals["t"] = t
    return templates


def mount_admin_static(app) -> None:
    app.mount("/admin/static", StaticFiles(directory=str(STATIC_DIR)), name="admin-static")


def _render(request: Request, name: str, nav: str, *, status_code: int = 200, **extra):
    ctx = {"lang": lang_of(request), "nav": nav, "csrf": csrf_token(request)}
    ctx.update(extra)
    return request.app.state.templates.TemplateResponse(
        request=request, name=name, context=ctx, status_code=status_code
    )


router = APIRouter(prefix="/admin")

JOB_STATUSES = ["pending", "running", "completed", "failed", "cancelled", "interrupted"]
_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}
PAGE_SIZE = 50


# --- public: login / logout / language ---
@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return _render(request, "login.html", "login", error=False)


@router.post("/login", response_class=HTMLResponse)
def login_submit(request: Request, api_key: str = Form(...)):
    acct = verify_admin_key(request, api_key)
    if acct is None:
        return _render(request, "login.html", "login", status_code=401, error=True)
    request.session["admin_account"] = acct
    return RedirectResponse("/admin", status_code=303)


@router.get("/logout")
def logout(request: Request):
    request.session.pop("admin_account", None)
    return RedirectResponse("/admin/login", status_code=303)


@router.get("/setlang")
def setlang(request: Request, code: str = "zh", next: str = "/admin"):
    resp = RedirectResponse(next, status_code=303)
    if code in LANGS:
        resp.set_cookie("lang", code, max_age=31536000, samesite="lax")
    return resp


# --- pages ---
@router.get("", response_class=HTMLResponse)
def dashboard(request: Request, admin: str = Depends(require_admin_web)):
    db = request.app.state.db
    registry = request.app.state.registry
    return _render(request, "dashboard.html", "dashboard", admin=admin,
                   total_users=db.count_users(),
                   total_jobs=db.count_jobs(),
                   total_services=len(registry.list()),
                   by_status=db.count_jobs_by_status())


def _accounts_rows(db) -> list[dict]:
    return [
        {"account_id": u.account_id, "display_name": u.display_name,
         "role": u.role, "status": u.status, "created_at": u.created_at,
         "key_count": len(db.list_api_keys(u.account_id)),
         "job_count": len(db.list_jobs(u.account_id))}
        for u in db.list_users()
    ]


@router.get("/accounts", response_class=HTMLResponse)
def accounts(request: Request, admin: str = Depends(require_admin_web)):
    return _render(request, "accounts.html", "accounts", admin=admin,
                   rows=_accounts_rows(request.app.state.db))


@router.post("/accounts", response_class=HTMLResponse)
def create_account(request: Request, admin: str = Depends(require_admin_web),
                   _c: None = Depends(verify_csrf),
                   account_id: str = Form(...), display_name: str = Form(""),
                   role: str = Form("user")):
    db = request.app.state.db
    account_id = account_id.strip()
    error = None
    if not account_id or role not in ("user", "admin"):
        error = "invalid_input"
    elif db.get_user(account_id) is not None:
        error = "exists"
    if error:
        return _render(request, "accounts.html", "accounts", status_code=400,
                       admin=admin, rows=_accounts_rows(db), error=error)
    db.create_user(account_id, display_name or None, role=role)
    return RedirectResponse("/admin/accounts", status_code=303)


@router.get("/accounts/{account_id}", response_class=HTMLResponse)
def account_detail(account_id: str, request: Request,
                   admin: str = Depends(require_admin_web)):
    db = request.app.state.db
    user = db.get_user(account_id)
    if user is None:
        raise HTTPException(404, "account not found")
    # One-time plaintext secret from a just-created key (PRG flash), shown once.
    new_key = request.session.pop("flash_key", None)
    return _render(request, "account_detail.html", "accounts", admin=admin,
                   user=user, keys=db.list_api_keys(account_id),
                   jobs=db.list_all_jobs(account_id=account_id, limit=20),
                   new_key=new_key)


@router.post("/accounts/{account_id}/keys")
def create_key(account_id: str, request: Request,
               admin: str = Depends(require_admin_web),
               _c: None = Depends(verify_csrf)):
    db = request.app.state.db
    if db.get_user(account_id) is None:
        raise HTTPException(404, "account not found")
    secret = secrets.token_urlsafe(24)
    key_id = f"gk_{uuid.uuid4().hex[:12]}"
    db.create_api_key(account_id, secret=secret, key_id=key_id)
    # Plaintext secret is unrecoverable (only its hash is stored) — flash it once.
    request.session["flash_key"] = {"key_id": key_id, "secret": secret}
    return RedirectResponse(f"/admin/accounts/{account_id}", status_code=303)


@router.post("/keys/{key_id}/revoke")
def revoke_key(key_id: str, request: Request,
               admin: str = Depends(require_admin_web),
               _c: None = Depends(verify_csrf),
               account_id: str = Form(...)):
    try:
        request.app.state.db.revoke_api_key(key_id)
    except KeyError:
        pass  # already gone — redirect back regardless
    return RedirectResponse(f"/admin/accounts/{account_id}", status_code=303)


@router.get("/jobs", response_class=HTMLResponse)
def jobs(request: Request, admin: str = Depends(require_admin_web),
         status: str = "", svc: str = "", account: str = "", page: int = 0):
    db = request.app.state.db
    page = max(page, 0)
    rows = db.list_all_jobs(status=status or None, svc=svc or None,
                            account_id=account or None,
                            limit=PAGE_SIZE, offset=page * PAGE_SIZE)
    return _render(request, "jobs.html", "jobs", admin=admin, rows=rows,
                   statuses=JOB_STATUSES, services=request.app.state.registry.list(),
                   f_status=status, f_svc=svc, f_account=account,
                   page=page, has_next=len(rows) == PAGE_SIZE)


@router.get("/jobs/{account_id}/{job_id}", response_class=HTMLResponse)
def job_detail(account_id: str, job_id: str, request: Request,
               admin: str = Depends(require_admin_web)):
    db = request.app.state.db
    job = db.get_job(account_id, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    live_status, refresh_error = None, None
    try:
        rec = request.app.state.registry.record(job.svc)
        down = request.app.state.dispatch.status(rec, job.fc_task_id or job.job_id)
        live_status = down.get("status")
    except Exception as exc:  # noqa: BLE001 — status refresh degrades gracefully
        refresh_error = f"{type(exc).__name__}: {exc}"
    return _render(request, "job_detail.html", "jobs", admin=admin, job=job,
                   live_status=live_status, refresh_error=refresh_error,
                   can_cancel=job.status not in _TERMINAL_STATUSES)


@router.post("/jobs/{account_id}/{job_id}/cancel")
def cancel_job(account_id: str, job_id: str, request: Request,
               admin: str = Depends(require_admin_web),
               _c: None = Depends(verify_csrf)):
    db = request.app.state.db
    if db.get_job(account_id, job_id) is None:
        raise HTTPException(404, "job not found")
    # MVP: local mark only (mirrors /v1/jobs/{id}/cancel).
    db.update_job(account_id, job_id, status="cancelled")
    return RedirectResponse(f"/admin/jobs/{account_id}/{job_id}", status_code=303)


@router.get("/services", response_class=HTMLResponse)
def services(request: Request, admin: str = Depends(require_admin_web),
             describe: str = "", reloaded: int = 0):
    registry = request.app.state.registry
    rows = []
    for name in registry.list():
        rec = registry.record(name)
        rows.append({"name": name, "url": rec.url, "region": rec.region,
                     "tier": rec.tier, "function": rec.function,
                     "gpu": rec.gpu, "oss_mount": rec.oss_mount})
    described = None
    if describe:
        # Describe one service on demand (cold-starting all 30+ would be costly).
        try:
            rec = registry.record(describe)
            base = request.app.state.dispatch.describe_base_url(rec)
            described = request.app.state.discover.describe(describe, base)
        except Exception as exc:  # noqa: BLE001 — describe degrades gracefully
            described = {"error": f"{type(exc).__name__}: {exc}"}
    return _render(request, "services.html", "services", admin=admin,
                   rows=rows, described=described, describe_name=describe,
                   reloaded=bool(reloaded),
                   reload_error=request.session.pop("flash_reload_error", None))


@router.post("/services/reload")
def reload_services(request: Request, admin: str = Depends(require_admin_web),
                    _c: None = Depends(verify_csrf)):
    """MVP dynamic loading: re-read services.yaml into the in-memory registry."""
    try:
        request.app.state.registry.reload()
    except Exception as exc:  # noqa: BLE001 — surface as a flash, don't 500
        request.session["flash_reload_error"] = f"{type(exc).__name__}: {exc}"
        return RedirectResponse("/admin/services", status_code=303)
    return RedirectResponse("/admin/services?reloaded=1", status_code=303)
