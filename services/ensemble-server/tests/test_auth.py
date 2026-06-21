"""Auth dependency unit tests."""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from server.auth.api_key import verify_api_key
from server.auth.deps import require_api_key
from server.settings import APIKeyConfig, AuthSettings


def _allowlist(secret: str) -> list[APIKeyConfig]:
    return [APIKeyConfig(
        key_id="ek_test",
        secret_hash=hashlib.sha256(secret.encode("utf-8")).hexdigest(),
        customer_id="cust1",
    )]


# ---------------------------------------------------------------------------
# verify_api_key
# ---------------------------------------------------------------------------

def test_verify_returns_entry_on_match():
    allowlist = _allowlist("right_secret")
    matched = verify_api_key("right_secret", allowlist)
    assert matched is not None
    assert matched.customer_id == "cust1"


def test_verify_returns_none_on_wrong_secret():
    allowlist = _allowlist("right_secret")
    assert verify_api_key("wrong", allowlist) is None


def test_verify_returns_none_on_empty_secret():
    allowlist = _allowlist("right_secret")
    assert verify_api_key("", allowlist) is None


def test_verify_returns_none_on_empty_allowlist():
    assert verify_api_key("any", []) is None


# ---------------------------------------------------------------------------
# require_api_key (legacy alias — delegates to require_auth)
# ---------------------------------------------------------------------------

def _legacy_fake_request(api_keys: list[APIKeyConfig], x_api_key: str) -> MagicMock:
    """Build a fake FastAPI Request with VPC bypass disabled and the given
    X-API-Key header set, so require_api_key falls through to the API-key path.
    """
    request = MagicMock()
    request.app.state.settings.auth = AuthSettings(bypass_vpc=False)
    request.app.state.settings.api_keys = api_keys
    request.headers = {"x-api-key": x_api_key, "host": "public.example.com"}
    return request


def test_require_api_key_returns_key_config_on_match():
    allowlist = _allowlist("good")
    request = _legacy_fake_request(allowlist, "good")
    key = require_api_key(request)
    assert key.customer_id == "cust1"


def test_require_api_key_raises_401_on_wrong_secret():
    allowlist = _allowlist("good")
    request = _legacy_fake_request(allowlist, "bad")
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(request)
    assert exc_info.value.status_code == 401


def test_require_api_key_raises_401_on_empty():
    allowlist = _allowlist("good")
    request = _legacy_fake_request(allowlist, "")
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(request)
    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# VPC detection
# ---------------------------------------------------------------------------

from server.auth.vpc import is_vpc_host


def test_vpc_host_matches_vpc_fcapp_run():
    assert is_vpc_host("fc-ensemble-abc123.cn-hangzhou-vpc.fcapp.run") is True


def test_vpc_host_rejects_public_fcapp_run():
    assert is_vpc_host("fc-ensemble-abc123.cn-hangzhou.fcapp.run") is False


def test_vpc_host_matches_localhost():
    assert is_vpc_host("localhost") is True
    assert is_vpc_host("localhost:9000") is True


def test_vpc_host_matches_127_loopback():
    assert is_vpc_host("127.0.0.1") is True
    assert is_vpc_host("127.0.0.1:8080") is True


def test_vpc_host_case_insensitive():
    assert is_vpc_host("FC-Ensemble-ABC.CN-Hangzhou-VPC.fcapp.run") is True


def test_vpc_host_handles_empty_and_none():
    assert is_vpc_host("") is False
    assert is_vpc_host(None) is False


def test_vpc_host_rejects_arbitrary_external():
    assert is_vpc_host("evil.example.com") is False
    assert is_vpc_host("api.bioagent.com") is False


# ---------------------------------------------------------------------------
# AuthIdentity + require_auth (three-layer fallthrough)
# ---------------------------------------------------------------------------

import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import jwt as pyjwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from server.auth.deps import AuthIdentity, require_auth
from server.settings import AuthSettings


def _fake_request(
    *,
    settings_auth: AuthSettings | None = None,
    api_keys: list[APIKeyConfig] | None = None,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Build a fake FastAPI Request with given app.state.settings + headers."""
    if settings_auth is None:
        settings_auth = AuthSettings()
    if api_keys is None:
        api_keys = []
    request = MagicMock()
    request.app.state.settings.auth = settings_auth
    request.app.state.settings.api_keys = api_keys
    request.headers = headers or {}
    return request


# --- VPC bypass branch ---

def test_require_auth_vpc_bypass_matches_vpc_host():
    request = _fake_request(headers={"host": "fc-ensemble-x.cn-hangzhou-vpc.fcapp.run"})
    identity = require_auth(request)
    assert identity.method == "vpc_bypass"
    assert identity.customer_id == "internal_vpc"


def test_require_auth_vpc_bypass_uses_custom_customer_id():
    request = _fake_request(
        settings_auth=AuthSettings(vpc_customer_id="internal_alt"),
        headers={"host": "localhost:9000"},
    )
    identity = require_auth(request)
    assert identity.customer_id == "internal_alt"


def test_require_auth_vpc_bypass_disabled_when_setting_false():
    request = _fake_request(
        settings_auth=AuthSettings(bypass_vpc=False),
        headers={"host": "fc-ensemble-x.cn-hangzhou-vpc.fcapp.run"},
    )
    with pytest.raises(HTTPException) as exc_info:
        require_auth(request)
    assert exc_info.value.status_code == 401


# --- API key branch ---

def test_require_auth_api_key_success():
    secret = "test_secret"
    sha = hashlib.sha256(secret.encode()).hexdigest()
    api_keys = [APIKeyConfig(key_id="ek_t", secret_hash=sha, customer_id="cust_t")]
    request = _fake_request(
        settings_auth=AuthSettings(bypass_vpc=False),  # disable VPC to force API key
        api_keys=api_keys,
        headers={"x-api-key": secret},
    )
    identity = require_auth(request)
    assert identity.method == "api_key"
    assert identity.customer_id == "cust_t"
    assert identity.raw_token_id == "ek_t"


def test_require_auth_no_creds_raises_401():
    request = _fake_request(
        settings_auth=AuthSettings(bypass_vpc=False),
        headers={"host": "public.example.com"},
    )
    with pytest.raises(HTTPException) as exc_info:
        require_auth(request)
    assert exc_info.value.status_code == 401


def test_require_auth_wrong_api_key_raises_401():
    secret = "right"
    sha = hashlib.sha256(secret.encode()).hexdigest()
    api_keys = [APIKeyConfig(key_id="ek_t", secret_hash=sha, customer_id="cust_t")]
    request = _fake_request(
        settings_auth=AuthSettings(bypass_vpc=False),
        api_keys=api_keys,
        headers={"x-api-key": "wrong"},
    )
    with pytest.raises(HTTPException) as exc_info:
        require_auth(request)
    assert exc_info.value.status_code == 401


# --- JWT branch ---

def _gen_jwt_test_kit(kid: str = "test-kid"):
    """Generate an RSA keypair + matching JWKS for jwt tests."""
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub_numbers = priv.public_key().public_numbers()

    def _b64url_uint(n: int) -> str:
        import base64
        b = n.to_bytes((n.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")

    jwks = {
        "keys": [{
            "kid": kid, "kty": "RSA", "alg": "RS256", "use": "sig",
            "n": _b64url_uint(pub_numbers.n),
            "e": _b64url_uint(pub_numbers.e),
        }]
    }
    return priv, jwks


def _sign_test_jwt(priv, payload: dict, kid: str) -> str:
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pyjwt.encode(payload, pem, algorithm="RS256", headers={"kid": kid})


def test_require_auth_jwt_success():
    priv, jwks = _gen_jwt_test_kit(kid="kid-A")
    now = datetime.now(timezone.utc)
    token = _sign_test_jwt(priv, {
        "sub": "customer-acme",
        "iat": now, "exp": now + timedelta(hours=1),
        "aud": "ensemble-server",
    }, kid="kid-A")

    jwks_url = "https://fake.example/jwks-A.json"

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return jwks

    settings_auth = AuthSettings(
        bypass_vpc=False,                   # disable VPC to force JWT path
        jwt_jwks_url=jwks_url,
        jwt_audience="ensemble-server",
    )
    request = _fake_request(
        settings_auth=settings_auth,
        headers={"authorization": f"Bearer {token}", "host": "public.example.com"},
    )

    # Clear the JWT verifier cache so test is hermetic
    from server.auth import jwt_verifier as jv
    jv._clear_cache(jwks_url)
    with patch("server.auth.jwt_verifier.httpx.get", lambda url, timeout: _Resp()):
        identity = require_auth(request)

    assert identity.method == "jwt"
    assert identity.customer_id == "customer-acme"
    jv._clear_cache(jwks_url)


def test_require_auth_jwt_failed_falls_through_to_api_key():
    """Bad JWT but valid X-API-Key → API key path wins."""
    secret = "good_secret"
    sha = hashlib.sha256(secret.encode()).hexdigest()
    api_keys = [APIKeyConfig(key_id="ek_fallback", secret_hash=sha, customer_id="cust_fb")]

    settings_auth = AuthSettings(
        bypass_vpc=False,
        jwt_jwks_url="https://fake.example/jwks-fb.json",
        jwt_audience="ensemble-server",
    )
    request = _fake_request(
        settings_auth=settings_auth,
        api_keys=api_keys,
        headers={
            "authorization": "Bearer not.a.real.token",
            "x-api-key": secret,
            "host": "public.example.com",
        },
    )

    # JWT verifier will try to fetch JWKS and fail — mock httpx.get to raise.
    from server.auth import jwt_verifier as jv
    jv._clear_cache("https://fake.example/jwks-fb.json")

    def _raise_get(url, timeout):
        raise httpx.HTTPError("network error")

    import httpx
    with patch("server.auth.jwt_verifier.httpx.get", _raise_get):
        identity = require_auth(request)

    assert identity.method == "api_key"
    assert identity.customer_id == "cust_fb"


def test_require_auth_vpc_takes_priority_over_jwt():
    """When VPC host AND a Bearer token are both present, VPC wins."""
    request = _fake_request(
        headers={
            "host": "fc-ensemble-x.cn-hangzhou-vpc.fcapp.run",
            "authorization": "Bearer fake.token",
        },
    )
    identity = require_auth(request)
    assert identity.method == "vpc_bypass"
