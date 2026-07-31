"""Admin console routes (server-side rendered). Login/logout/setlang are public;
page routes take `Depends(require_admin_web)` (redirect to /admin/login if not)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .auth import require_admin_web, verify_admin_key
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
    ctx = {"lang": lang_of(request), "nav": nav}
    ctx.update(extra)
    return request.app.state.templates.TemplateResponse(
        request=request, name=name, context=ctx, status_code=status_code
    )


router = APIRouter(prefix="/admin")


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
