from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from server.auth import jwt_verifier
from server.auth.jwt_verifier import JWTError, verify_jwt


def _b64url_uint(n: int) -> str:
    b = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


@pytest.fixture()
def keypair():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture()
def jwks(keypair):
    pn = keypair.public_key().public_numbers()
    return {"keys": [{"kid": "self", "kty": "RSA", "alg": "RS256", "use": "sig",
                      "n": _b64url_uint(pn.n), "e": _b64url_uint(pn.e)}]}


@pytest.fixture()
def jwks_url():
    return "https://fake.example/.well-known/jwks.json"


@pytest.fixture(autouse=True)
def _clear(jwks_url):
    jwt_verifier._clear_cache(jwks_url)
    yield
    jwt_verifier._clear_cache(jwks_url)


def _sign(keypair, payload, kid="self"):
    pem = keypair.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pyjwt.encode(payload, pem, algorithm="RS256", headers={"kid": kid})


@pytest.fixture()
def mock_jwks(jwks, jwks_url):
    class _Resp:
        def raise_for_status(self): pass
        def json(self): return jwks
    with patch("server.auth.jwt_verifier.httpx.get", lambda url, timeout: _Resp()):
        yield


def test_happy_path(keypair, jwks_url, mock_jwks):
    now = datetime.now(timezone.utc)
    tok = _sign(keypair, {"sub": "acme", "iat": now, "exp": now + timedelta(hours=1),
                          "aud": "gateway-server"})
    claims = verify_jwt(tok, jwks_url=jwks_url, audience="gateway-server")
    assert claims["sub"] == "acme"


def test_rejects_expired(keypair, jwks_url, mock_jwks):
    now = datetime.now(timezone.utc)
    tok = _sign(keypair, {"sub": "x", "iat": now - timedelta(hours=2),
                          "exp": now - timedelta(hours=1), "aud": "gateway-server"})
    with pytest.raises(JWTError):
        verify_jwt(tok, jwks_url=jwks_url, audience="gateway-server")


def test_disabled_when_empty_url():
    with pytest.raises(JWTError, match="disabled"):
        verify_jwt("x", jwks_url="", audience=None)


def test_issuer_enforced(keypair, jwks_url, mock_jwks):
    now = datetime.now(timezone.utc)
    tok = _sign(keypair, {"sub": "x", "iat": now, "exp": now + timedelta(hours=1),
                          "aud": "gateway-server", "iss": "https://idp.a/realms/bioq"})
    # matching issuer accepted
    claims = verify_jwt(tok, jwks_url=jwks_url, audience="gateway-server",
                        issuer="https://idp.a/realms/bioq")
    assert claims["sub"] == "x"
    # wrong issuer rejected (prod must set GATEWAY_AUTH__JWT_ISSUER)
    with pytest.raises(JWTError):
        verify_jwt(tok, jwks_url=jwks_url, audience="gateway-server",
                   issuer="https://idp.b/realms/bioq")
