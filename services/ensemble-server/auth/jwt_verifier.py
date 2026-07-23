"""JWT verification using JWKS fetched from a remote endpoint.

Used to verify RS256-signed JWTs issued by `edge/jwt/` (or any other
JWKS-publishing identity provider).  Caches the JWKS in-process; on a kid
miss the cache is force-refreshed once (handles key rotation) before
failing.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import jwt
from jwt.algorithms import RSAAlgorithm
from jwt.exceptions import InvalidTokenError as JWTError

# Public re-export so callers can `except JWTError`.
__all__ = ["JWTError", "verify_jwt"]


_jwks_cache: dict[str, tuple[dict, float]] = {}  # jwks_url → (jwks_dict, expires_at)


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
    """Find the JWK with matching kid and return a verifying public key."""
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            return RSAAlgorithm.from_jwk(k)
    raise JWTError(f"kid {kid!r} not in JWKS")


def _clear_cache(jwks_url: str) -> None:
    """For tests + key-rotation force-refresh."""
    _jwks_cache.pop(jwks_url, None)


def verify_jwt(
    token: str,
    *,
    jwks_url: str,
    audience: str | None,
    issuer: str | None = None,
    ttl_sec: int = 3600,
) -> dict[str, Any]:
    """Verify RS256-signed JWT against the JWKS at `jwks_url`.

    Returns the validated payload dict (claims).  Raises `JWTError` on any
    failure: bad signature, expired, wrong aud / iss, missing required
    claims, kid not in JWKS (after a force-refresh).

    `audience` and `issuer`: when non-empty, the corresponding claim is
    validated.  Empty / None = skip that check.
    """
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
        # kid miss — force-refresh JWKS once (key rotation case)
        _clear_cache(jwks_url)
        jwks = _fetch_jwks(jwks_url, ttl_sec)
        public_key = _get_public_key(jwks, kid)

    options = {"require": ["exp", "iat", "sub"]}
    decode_kwargs: dict[str, Any] = {
        "algorithms": ["RS256"],
        "options": options,
    }
    if audience:
        decode_kwargs["audience"] = audience
    if issuer:
        decode_kwargs["issuer"] = issuer

    return jwt.decode(token, public_key, **decode_kwargs)
