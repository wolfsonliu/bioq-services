"""Admin console routes (server-side rendered). Login/logout/setlang are public;
page routes take `Depends(require_admin_web)` (redirect to /admin/login if not)."""

from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from server.auth.jwt_verifier import verify_jwt

from . import sso
from .auth import csrf_token, require_admin_web, verify_csrf
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
    return _render(request, "login.html", "login",
                   sso_enabled=sso.sso_enabled(request.app.state.settings))


# --- OIDC Authorization Code (browser SSO login) ---
@router.get("/auth/login")
def sso_login(request: Request):
    settings = request.app.state.settings
    if not sso.sso_enabled(settings):
        raise HTTPException(404, "SSO not configured")
    state = secrets.token_urlsafe(16)
    request.session["oidc_state"] = state
    redirect_uri = str(request.url_for("sso_callback"))
    return RedirectResponse(sso.authorize_url(settings, redirect_uri, state),
                            status_code=303)


@router.get("/auth/callback", name="sso_callback")
def sso_callback(request: Request, code: str = "", state: str = ""):
    settings = request.app.state.settings
    if not state or state != request.session.pop("oidc_state", None):
        raise HTTPException(400, "bad state")
    redirect_uri = str(request.url_for("sso_callback"))
    try:
        tok = sso.exchange_code(settings, code, redirect_uri)
    except sso.SSOError as e:
        raise HTTPException(502, str(e)) from e
    claims = verify_jwt(tok["access_token"], jwks_url=settings.auth.jwt_jwks_url,
                        audience=settings.auth.jwt_audience or None,
                        issuer=settings.auth.jwt_issuer or None,
                        ttl_sec=settings.auth.jwt_jwks_cache_ttl_sec)
    account = claims.get("sub", "")
    groups = claims.get(settings.auth.jwt_groups_claim) or []
    if isinstance(groups, str):
        groups = [groups]
    role = "admin" if settings.auth.jwt_admin_group in groups else "user"
    display = claims.get("preferred_username") or claims.get("email")
    request.app.state.db.upsert_user(account, display_name=display, role=role)
    if role != "admin":
        raise HTTPException(403, "admin privileges required")
    request.session["admin_account"] = account
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
         "job_count": len(db.list_jobs(u.account_id))}
        for u in db.list_users()
    ]


# Accounts are read-only in the console: users are provisioned just-in-time from
# the IdP (account_id = token sub, role from groups). Manage them in Keycloak.
@router.get("/accounts", response_class=HTMLResponse)
def accounts(request: Request, admin: str = Depends(require_admin_web)):
    return _render(request, "accounts.html", "accounts", admin=admin,
                   rows=_accounts_rows(request.app.state.db))


@router.get("/accounts/{account_id}", response_class=HTMLResponse)
def account_detail(account_id: str, request: Request,
                   admin: str = Depends(require_admin_web)):
    db = request.app.state.db
    user = db.get_user(account_id)
    if user is None:
        raise HTTPException(404, "account not found")
    return _render(request, "account_detail.html", "accounts", admin=admin,
                   user=user, jobs=db.list_all_jobs(account_id=account_id, limit=20))


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
