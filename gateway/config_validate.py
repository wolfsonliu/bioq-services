"""Fail-fast validation of resolved GatewaySettings.

Catches the misconfig classes that silently broke deployments: a malformed
`fc_endpoint` (value contained the env key name), an unset/placeholder OSS bucket
(NoSuchBucket), and auth that would lock everyone out. FastAPI-free so `check`
can run headless. See ADR 2026-08-05-gateway-config-generation.md.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

_HOST_RE = re.compile(r"^[A-Za-z0-9.-]+(:\d+)?$")
_BUCKET_PLACEHOLDERS = ("<", ">", "change-me", "gateway_oss_bucket", "bucket-name")

_DISPATCH_BACKENDS = {"fc", "http", "openfaas"}
_STORAGE_BACKENDS = {"oss", "file"}


def _bad_url(value: str) -> bool:
    p = urlparse(value)
    return p.scheme not in ("http", "https") or not p.netloc


def validate_settings(settings) -> tuple[list[str], list[str]]:
    """Return (fatals, warnings). Fatals should abort startup."""
    fatals: list[str] = []
    warnings: list[str] = []
    a = settings.auth

    if settings.dispatch_backend not in _DISPATCH_BACKENDS:
        fatals.append(
            f"GATEWAY_DISPATCH_BACKEND={settings.dispatch_backend!r} invalid; "
            f"expected one of {sorted(_DISPATCH_BACKENDS)}")
    if settings.storage_backend not in _STORAGE_BACKENDS:
        fatals.append(
            f"GATEWAY_STORAGE_BACKEND={settings.storage_backend!r} invalid; "
            f"expected one of {sorted(_STORAGE_BACKENDS)}")

    # fc_endpoint must be a bare host (no scheme/path/'='/whitespace). This is the
    # exact shape that caught the `gateway_fc_endpoint=...` typo.
    ep = settings.fc_endpoint
    if ep and not _HOST_RE.match(ep):
        fatals.append(
            f"GATEWAY_FC_ENDPOINT={ep!r} is not a bare host "
            "(no scheme, no path, no '=', no spaces); "
            "e.g. <account-id>.cn-hangzhou-internal.fc.aliyuncs.com")

    # URL-shaped fields.
    for label, value in (
        ("GATEWAY_AUTH__JWT_JWKS_URL", a.jwt_jwks_url),
        ("GATEWAY_AUTH__JWT_ISSUER", a.jwt_issuer),
        ("GATEWAY_AUTH__OIDC_ISSUER", a.oidc_issuer),
        ("GATEWAY_OPENFAAS_GATEWAY_URL", settings.openfaas_gateway_url),
    ):
        if value and _bad_url(value):
            fatals.append(f"{label}={value!r} is not a valid http(s) URL")

    # storage=oss ⇒ a real bucket.
    if settings.storage_backend == "oss":
        bucket = (settings.oss_bucket or "").strip()
        low = bucket.lower()
        if not bucket or any(tok in low for tok in _BUCKET_PLACEHOLDERS):
            fatals.append(
                f"GATEWAY_STORAGE_BACKEND=oss but GATEWAY_OSS_BUCKET={settings.oss_bucket!r} "
                "is empty/placeholder; set it to a bucket that exists under your OSS AK")

    # dispatch=openfaas ⇒ a gateway URL (mirror make_dispatcher, earlier+friendlier).
    if settings.dispatch_backend == "openfaas" and not settings.openfaas_gateway_url:
        fatals.append(
            "GATEWAY_DISPATCH_BACKEND=openfaas but GATEWAY_OPENFAAS_GATEWAY_URL is unset")

    # bypass_vpc=false with no JWKS ⇒ nobody can authenticate.
    if not a.bypass_vpc and not a.jwt_jwks_url:
        fatals.append(
            "GATEWAY_AUTH__BYPASS_VPC=false but GATEWAY_AUTH__JWT_JWKS_URL is unset "
            "— no caller could authenticate")

    # fc without OpenAPI creds falls back to HTTP status polling (works, but noisy).
    if settings.dispatch_backend == "fc" and not (
            settings.ali_access_key_id and settings.ali_access_key_secret):
        warnings.append(
            "GATEWAY_DISPATCH_BACKEND=fc but ALI_AK/ALI_SK unset — falling back to "
            "HTTP status polling (spins downstream instances per poll)")

    return fatals, warnings
