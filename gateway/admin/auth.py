"""Browser auth for the admin console.

The gateway's API auth (`X-API-Key` / Bearer / VPC) can't be used from a browser
navigation — only a cookie is carried automatically. So the console uses a
cookie session (Starlette SessionMiddleware): an admin logs in with their API
key, we verify `role == "admin"`, and stash the account in the signed session.
Internal VPC hosts are still bypassed (no login needed).
"""

from __future__ import annotations

from fastapi import HTTPException, Request
from server.auth.api_key import hash_secret
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


def verify_admin_key(request: Request, api_key: str) -> str | None:
    """Return the account_id if `api_key` maps to an admin user, else None."""
    row = request.app.state.db.find_api_key(hash_secret(api_key))
    if row is None:
        return None
    user = request.app.state.db.get_user(row.account_id)
    return row.account_id if (user is not None and user.role == "admin") else None
