"""FastAPI auth dependencies.

Three-layer fallthrough chain (see
engineering/decisions/2026-06-21-ensemble-server-auth.md):

  1. VPC bypass — Host header matches `*-vpc.fcapp.run` or local-dev host
  2. JWT       — Authorization: Bearer <token>, verified against JWKS
  3. API Key   — X-API-Key header, SHA256-matched against settings.api_keys

The unified `require_auth` returns an `AuthIdentity` with customer_id +
method + raw_token_id for downstream route handlers and audit logging.

Legacy `require_api_key` is kept as a backward-compatible thin wrapper
that delegates to `require_auth` (drops VPC + JWT paths in the type
contract, but in practice still benefits from fallthrough because the
dependency chain ends up using the new logic).
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from fastapi import HTTPException, Request
from pydantic import BaseModel

from ..settings import APIKeyConfig
from .api_key import verify_api_key
from .jwt_verifier import JWTError, verify_jwt
from .vpc import is_vpc_host

logger = logging.getLogger(__name__)


class AuthIdentity(BaseModel):
    """Unified identity for an authenticated request."""

    customer_id: str
    method: Literal["vpc_bypass", "jwt", "api_key"]
    raw_token_id: Optional[str] = None    # api_key.key_id or jwt jti — for audit


def require_auth(request: Request) -> AuthIdentity:
    """Three-layer fallthrough: VPC → JWT → API Key.

    Raises HTTPException(401) if none of the three auth methods succeeds.
    """
    settings = request.app.state.settings

    # ---- 1. VPC bypass ----
    if settings.auth.bypass_vpc:
        host = request.headers.get("host")
        if is_vpc_host(host):
            logger.debug("auth: vpc bypass for host=%r", host)
            return AuthIdentity(
                customer_id=settings.auth.vpc_customer_id,
                method="vpc_bypass",
            )

    # ---- 2. JWT ----
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
            sub = claims.get("sub", "")
            if settings.auth.jwt_sub_is_customer:
                customer_id = sub
            else:
                customer_id = settings.auth.jwt_sub_to_customer.get(sub, sub)
            logger.info("auth: jwt sub=%r → customer_id=%r", sub, customer_id)
            return AuthIdentity(
                customer_id=customer_id,
                method="jwt",
                raw_token_id=claims.get("jti"),
            )
        except JWTError as e:
            logger.info("auth: jwt verify failed (%s); falling through to api_key", e)

    # ---- 3. API Key ----
    api_key = request.headers.get("x-api-key", "")
    if api_key:
        matched = verify_api_key(api_key, settings.api_keys)
        if matched is not None:
            logger.info(
                "auth: api_key key_id=%r → customer_id=%r",
                matched.key_id, matched.customer_id,
            )
            return AuthIdentity(
                customer_id=matched.customer_id,
                method="api_key",
                raw_token_id=matched.key_id,
            )

    raise HTTPException(
        401,
        "missing or invalid credentials (provide Authorization: Bearer or X-API-Key)",
    )


def require_api_key(request: Request) -> APIKeyConfig:
    """Backward-compatible alias for routes still expecting an APIKeyConfig.

    Delegates to `require_auth` and converts the result; reconstructs a
    minimal APIKeyConfig with the customer_id, sufficient for routes that
    only read `.customer_id`.  NEW routes should use `require_auth` and
    `AuthIdentity` directly.
    """
    identity = require_auth(request)
    return APIKeyConfig(
        key_id=identity.raw_token_id or f"auto:{identity.method}",
        secret_hash="",  # not used downstream
        customer_id=identity.customer_id,
    )
