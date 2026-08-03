"""Three-layer auth fallthrough: VPC bypass -> JWT -> API Key (DB lookup).

Returns AuthIdentity{account_id, method, raw_token_id}. Raises 401 if none.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from fastapi import HTTPException, Request
from pydantic import BaseModel

from .api_key import hash_secret
from .jwt_verifier import JWTError, verify_jwt
from .vpc import is_vpc_host

logger = logging.getLogger(__name__)


class AuthIdentity(BaseModel):
    account_id: str
    method: Literal["vpc_bypass", "jwt", "api_key"]
    raw_token_id: Optional[str] = None


def require_auth(request: Request) -> AuthIdentity:
    settings = request.app.state.settings

    # 1. VPC bypass
    if settings.auth.bypass_vpc and is_vpc_host(request.headers.get("host")):
        return AuthIdentity(account_id=settings.auth.vpc_account_id, method="vpc_bypass")

    # 2. JWT
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        try:
            claims = verify_jwt(
                token,
                jwks_url=settings.auth.jwt_jwks_url,
                audience=settings.auth.jwt_audience or None,
                issuer=settings.auth.jwt_issuer or None,
                ttl_sec=settings.auth.jwt_jwks_cache_ttl_sec,
            )
            account_id = claims.get("sub", "")
            groups = claims.get(settings.auth.jwt_groups_claim) or []
            if isinstance(groups, str):
                groups = [groups]
            role = "admin" if settings.auth.jwt_admin_group in groups else "user"
            display = claims.get("preferred_username") or claims.get("email")
            # JIT provisioning: IdP is the source of truth for OIDC users' role,
            # so upsert on every login keeps `require_admin` + accounts page in sync.
            request.app.state.db.upsert_user(account_id, display_name=display, role=role)
            return AuthIdentity(account_id=account_id, method="jwt",
                                raw_token_id=claims.get("jti"))
        except JWTError as e:
            logger.info("auth: jwt verify failed (%s); fall through to api_key", e)

    # 3. API Key (DB)
    api_key = request.headers.get("x-api-key", "")
    if api_key:
        row = request.app.state.db.find_api_key(hash_secret(api_key))
        if row is not None:
            request.app.state.db.touch_api_key(row.key_id)
            return AuthIdentity(account_id=row.account_id, method="api_key",
                                raw_token_id=row.key_id)

    raise HTTPException(401, "missing or invalid credentials "
                             "(provide Authorization: Bearer or X-API-Key)")


def require_admin(request: Request) -> AuthIdentity:
    """require_auth + role 闸：仅 role == "admin" 的账户放行（VPC bypass 按配置）。"""
    ident = require_auth(request)
    settings = request.app.state.settings
    if ident.method == "vpc_bypass" and settings.auth.vpc_is_admin:
        return ident
    user = request.app.state.db.get_user(ident.account_id)
    if user is None or user.role != "admin":
        raise HTTPException(403, "admin privileges required")
    return ident
