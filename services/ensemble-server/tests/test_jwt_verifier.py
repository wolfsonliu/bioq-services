"""JWT verifier unit tests.

Generates an RSA keypair in-test, signs JWTs with the private key, builds
a JWKS dict from the public key, and verifies with the module's `verify_jwt`.
Uses httpx MockTransport (via monkeypatch on the module's `httpx.get`) to
serve the fake JWKS without a real HTTP server.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from server.auth import jwt_verifier
from server.auth.jwt_verifier import JWTError, verify_jwt


# ---------------------------------------------------------------------------
# Fixtures — RSA keypair + matching JWKS
# ---------------------------------------------------------------------------

def _gen_keypair():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return priv


def _to_jwk(priv, kid: str) -> dict:
    """Build a JWK dict from the public side of an RSA private key."""
    pub_numbers = priv.public_key().public_numbers()

    def _b64url_uint(n: int) -> str:
        import base64
        b = n.to_bytes((n.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")

    return {
        "kid": kid,
        "kty": "RSA",
        "alg": "RS256",
        "use": "sig",
        "n": _b64url_uint(pub_numbers.n),
        "e": _b64url_uint(pub_numbers.e),
    }


def _sign(priv, payload: dict, kid: str) -> str:
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pyjwt.encode(payload, pem, algorithm="RS256", headers={"kid": kid})


@pytest.fixture
def keypair():
    return _gen_keypair()


@pytest.fixture
def jwks(keypair):
    return {"keys": [_to_jwk(keypair, kid="self")]}


@pytest.fixture
def jwks_url():
    return "https://fake.example/.well-known/jwks.json"


@pytest.fixture(autouse=True)
def _clear_cache_between_tests(jwks_url):
    jwt_verifier._clear_cache(jwks_url)
    yield
    jwt_verifier._clear_cache(jwks_url)


@pytest.fixture
def mock_jwks_response(jwks, jwks_url):
    """Patch httpx.get in the verifier module to return our fake JWKS."""
    class _Resp:
        def __init__(self, json_data):
            self._json = json_data
        def raise_for_status(self):
            pass
        def json(self):
            return self._json

    def _fake_get(url, timeout):
        assert url == jwks_url
        return _Resp(jwks)

    with patch("server.auth.jwt_verifier.httpx.get", _fake_get):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_verify_jwt_happy_path(keypair, jwks_url, mock_jwks_response):
    """Valid token + matching JWKS → returns payload."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "customer-acme",
        "iat": now,
        "exp": now + timedelta(hours=1),
        "aud": "ensemble-server",
    }
    token = _sign(keypair, payload, kid="self")

    claims = verify_jwt(token, jwks_url=jwks_url, audience="ensemble-server")
    assert claims["sub"] == "customer-acme"
    assert claims["aud"] == "ensemble-server"


def test_verify_jwt_rejects_expired(keypair, jwks_url, mock_jwks_response):
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "x", "iat": now - timedelta(hours=2),
        "exp": now - timedelta(hours=1),  # already expired
        "aud": "ensemble-server",
    }
    token = _sign(keypair, payload, kid="self")
    with pytest.raises(JWTError):
        verify_jwt(token, jwks_url=jwks_url, audience="ensemble-server")


def test_verify_jwt_rejects_wrong_audience(keypair, jwks_url, mock_jwks_response):
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "x", "iat": now, "exp": now + timedelta(hours=1),
        "aud": "wrong-audience",
    }
    token = _sign(keypair, payload, kid="self")
    with pytest.raises(JWTError):
        verify_jwt(token, jwks_url=jwks_url, audience="ensemble-server")


def test_verify_jwt_skips_audience_when_not_required(keypair, jwks_url, mock_jwks_response):
    """audience=None → skip aud check; token without aud is accepted."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "x", "iat": now, "exp": now + timedelta(hours=1),
    }
    token = _sign(keypair, payload, kid="self")
    claims = verify_jwt(token, jwks_url=jwks_url, audience=None)
    assert claims["sub"] == "x"


def test_verify_jwt_rejects_missing_kid(keypair, jwks_url, mock_jwks_response):
    """Token without `kid` header is rejected."""
    now = datetime.now(timezone.utc)
    payload = {"sub": "x", "iat": now, "exp": now + timedelta(hours=1)}
    # Sign without kid in header
    pem = keypair.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    token = pyjwt.encode(payload, pem, algorithm="RS256")
    with pytest.raises(JWTError, match="kid"):
        verify_jwt(token, jwks_url=jwks_url, audience=None)


def test_verify_jwt_rejects_unknown_kid(keypair, jwks_url, mock_jwks_response):
    """Token's kid not in JWKS → rejected (after one force-refresh attempt)."""
    now = datetime.now(timezone.utc)
    payload = {"sub": "x", "iat": now, "exp": now + timedelta(hours=1)}
    token = _sign(keypair, payload, kid="other-kid")
    with pytest.raises(JWTError, match="kid"):
        verify_jwt(token, jwks_url=jwks_url, audience=None)


def test_verify_jwt_disabled_when_jwks_url_empty():
    """Empty jwks_url → JWT verification is disabled (treated as failure)."""
    with pytest.raises(JWTError, match="disabled"):
        verify_jwt("any-token", jwks_url="", audience=None)


def test_verify_jwt_uses_cached_jwks_within_ttl(keypair, jwks_url):
    """Two calls within TTL hit the cache (httpx.get called once)."""
    jwks_doc = {"keys": [_to_jwk(keypair, kid="self")]}
    call_count = {"n": 0}

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return jwks_doc

    def _fake_get(url, timeout):
        call_count["n"] += 1
        return _Resp()

    with patch("server.auth.jwt_verifier.httpx.get", _fake_get):
        now = datetime.now(timezone.utc)
        payload = {"sub": "x", "iat": now, "exp": now + timedelta(hours=1)}
        token = _sign(keypair, payload, kid="self")
        verify_jwt(token, jwks_url=jwks_url, audience=None, ttl_sec=3600)
        verify_jwt(token, jwks_url=jwks_url, audience=None, ttl_sec=3600)

    assert call_count["n"] == 1, f"expected 1 fetch within TTL, got {call_count['n']}"


def test_verify_jwt_force_refreshes_on_kid_miss(keypair):
    """When the kid isn't in cached JWKS, the cache is force-refreshed once.

    Simulates key rotation: server issues a token with a new kid; verifier's
    cached JWKS is stale; refetch finds the new kid.
    """
    new_kid = "rotated-kid"
    jwks_url = "https://fake.example/rotate-jwks.json"
    jwt_verifier._clear_cache(jwks_url)

    # Two JWKS versions: stale (no new_kid) and fresh (with new_kid).
    stale_jwks = {"keys": [_to_jwk(keypair, kid="other-kid")]}
    fresh_jwks = {"keys": [_to_jwk(keypair, kid=new_kid)]}
    serve_idx = {"n": 0}

    class _Resp:
        def __init__(self, doc): self._doc = doc
        def raise_for_status(self): pass
        def json(self): return self._doc

    def _fake_get(url, timeout):
        if serve_idx["n"] == 0:
            serve_idx["n"] = 1
            return _Resp(stale_jwks)
        return _Resp(fresh_jwks)

    with patch("server.auth.jwt_verifier.httpx.get", _fake_get):
        now = datetime.now(timezone.utc)
        payload = {"sub": "x", "iat": now, "exp": now + timedelta(hours=1)}
        token = _sign(keypair, payload, kid=new_kid)
        claims = verify_jwt(token, jwks_url=jwks_url, audience=None)
        assert claims["sub"] == "x"
    jwt_verifier._clear_cache(jwks_url)
