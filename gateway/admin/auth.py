"""Browser auth for the admin console.

A browser navigation can't carry the API Bearer/JWT header, so the console uses a
cookie session (Starlette SessionMiddleware): the admin signs in via OIDC SSO
(see routes: /admin/auth/*), we verify `role == "admin"` and stash the account in
the signed session. Internal VPC hosts are bypassed (no login needed).
"""

from __future__ import annotations

import secrets

from fastapi import Form, HTTPException, Request
from server.auth.vpc import is_vpc_host


def _vpc_admin(request: Request) -> str | None:
    s = request.app.state.settings.auth
    if s.bypass_vpc and s.vpc_is_admin and is_vpc_host(request.headers.get("host")):
        return s.vpc_account_id
    return None


def current_admin(request: Request) -> str | None:
    """Authenticated admin account (VPC bypass or session cookie), else None."""
    if acct := _vpc_admin(request):
        return acct
    acct = request.session.get("admin_account")
    if acct:
        user = request.app.state.db.get_user(acct)
        if user is not None and user.role == "admin":
            return acct
    return None


def require_admin_web(request: Request) -> str:
    """Page dependency: unauthenticated → 307 to /admin/login (browser-friendly)."""
    acct = current_admin(request)
    if acct is None:
        raise HTTPException(status_code=307, headers={"Location": "/admin/login"})
    return acct


def csrf_token(request: Request) -> str:
    """Get-or-create a per-session CSRF token (synchronizer-token pattern)."""
    tok = request.session.get("csrf")
    if not tok:
        tok = secrets.token_urlsafe(24)
        request.session["csrf"] = tok
    return tok


def verify_csrf(request: Request, csrf: str = Form("")) -> None:
    """POST dependency: reject if the form token doesn't match the session token.

    Defense-in-depth on top of the SameSite=lax session cookie.
    """
    if not csrf or csrf != request.session.get("csrf"):
        raise HTTPException(status_code=403, detail="bad csrf token")
