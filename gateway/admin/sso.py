"""OIDC Authorization Code helpers for the admin console (browser SSO login).

Distinct from the API/CLI JWT path: here the gateway is an OIDC *client* — it
redirects the browser to the IdP, then exchanges the returned code for tokens.
The resulting access token is verified with the same `verify_jwt()` used by the
API path (JWKS + aud + iss), so role/groups handling stays identical.
"""
from __future__ import annotations

import time
from urllib.parse import urlencode

import httpx

_disc_cache: dict[str, tuple[dict, float]] = {}


class SSOError(Exception):
    """SSO/OIDC interaction failed (discovery or token exchange)."""


def sso_enabled(settings) -> bool:
    a = settings.auth
    return bool(a.oidc_issuer and a.oidc_client_id and a.oidc_client_secret
                and a.jwt_jwks_url)


def discover(issuer: str, ttl_sec: int = 3600) -> dict:
    now = time.time()
    cached = _disc_cache.get(issuer)
    if cached and cached[1] > now:
        return cached[0]
    try:
        r = httpx.get(issuer.rstrip("/") + "/.well-known/openid-configuration",
                      timeout=10.0)
        r.raise_for_status()
        meta = r.json()
    except httpx.HTTPError as e:
        raise SSOError(f"OIDC discovery failed: {e}") from e
    _disc_cache[issuer] = (meta, now + ttl_sec)
    return meta


def authorize_url(settings, redirect_uri: str, state: str) -> str:
    meta = discover(settings.auth.oidc_issuer)
    q = urlencode({
        "response_type": "code",
        "client_id": settings.auth.oidc_client_id,
        "redirect_uri": redirect_uri,
        "scope": "openid profile groups",
        "state": state,
    })
    return f"{meta['authorization_endpoint']}?{q}"


def exchange_code(settings, code: str, redirect_uri: str) -> dict:
    meta = discover(settings.auth.oidc_issuer)
    try:
        r = httpx.post(meta["token_endpoint"], timeout=15.0, data={
            "grant_type": "authorization_code", "code": code,
            "redirect_uri": redirect_uri,
            "client_id": settings.auth.oidc_client_id,
            "client_secret": settings.auth.oidc_client_secret})
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        raise SSOError(f"code exchange failed: {e}") from e
