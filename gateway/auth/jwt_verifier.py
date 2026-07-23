"""JWT verification using JWKS fetched from edge/jwt (or any JWKS provider).
Caches JWKS in-process; force-refresh once on kid miss (key rotation)."""

from __future__ import annotations

import time
from typing import Any

import httpx
import jwt
from jwt.algorithms import RSAAlgorithm
from jwt.exceptions import InvalidTokenError as JWTError

__all__ = ["JWTError", "verify_jwt"]

_jwks_cache: dict[str, tuple[dict, float]] = {}


def _fetch_jwks(jwks_url: str, ttl_sec: int) -> dict:
    now = time.time()
    if cached := _jwks_cache.get(jwks_url):
        jwks, expires_at = cached
        if expires_at > now:
            return jwks
    resp = httpx.get(jwks_url, timeout=10.0)
    resp.raise_for_status()
    jwks = resp.json()
    _jwks_cache[jwks_url] = (jwks, now + ttl_sec)
    return jwks


def _get_public_key(jwks: dict, kid: str):
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            return RSAAlgorithm.from_jwk(k)
    raise JWTError(f"kid {kid!r} not in JWKS")


def _clear_cache(jwks_url: str) -> None:
    _jwks_cache.pop(jwks_url, None)


def verify_jwt(token: str, *, jwks_url: str, audience: str | None,
               issuer: str | None = None, ttl_sec: int = 3600) -> dict[str, Any]:
    if not jwks_url:
        raise JWTError("JWT verification disabled: no jwks_url configured")
    header = jwt.get_unverified_header(token)
    kid = header.get("kid")
    if not kid:
        raise JWTError("token missing `kid` header")
    jwks = _fetch_jwks(jwks_url, ttl_sec)
    try:
        public_key = _get_public_key(jwks, kid)
    except JWTError:
        _clear_cache(jwks_url)
        jwks = _fetch_jwks(jwks_url, ttl_sec)
        public_key = _get_public_key(jwks, kid)
    options = {"require": ["exp", "iat", "sub"]}
    decode_kwargs: dict[str, Any] = {"algorithms": ["RS256"], "options": options}
    if audience:
        decode_kwargs["audience"] = audience
    if issuer:
        decode_kwargs["issuer"] = issuer
    return jwt.decode(token, public_key, **decode_kwargs)
