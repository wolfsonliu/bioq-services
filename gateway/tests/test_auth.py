from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from server.auth import jwt_verifier as jv
from server.auth.api_key import hash_secret
from server.auth.deps import require_auth
from server.settings import AuthSettings


def _b64url_uint(n: int) -> str:
    b = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _keypair_and_jwks(kid: str = "self"):
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pn = priv.public_key().public_numbers()
    jwks = {
        "keys": [
            {
                "kid": kid,
                "kty": "RSA",
                "alg": "RS256",
                "use": "sig",
                "n": _b64url_uint(pn.n),
                "e": _b64url_uint(pn.e),
            }
        ]
    }
    return priv, jwks


def _sign(priv, payload: dict, kid: str = "self") -> str:
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pyjwt.encode(payload, pem, algorithm="RS256", headers={"kid": kid})


def _jwt_auth(url: str) -> AuthSettings:
    return AuthSettings(
        bypass_vpc=False, jwt_jwks_url=url, jwt_audience="gateway-server"
    )


class _JwksResp:
    def __init__(self, jwks):
        self._jwks = jwks

    def raise_for_status(self):
        pass

    def json(self):
        return self._jwks


def _req(*, auth=None, headers=None, db=None):
    r = MagicMock()
    r.app.state.settings.auth = auth or AuthSettings()
    r.app.state.db = db
    r.headers = headers or {}
    return r


def test_vpc_bypass():
    r = _req(headers={"host": "fc-gateway-x.cn-hangzhou-vpc.fcapp.run"})
    ident = require_auth(r)
    assert ident.method == "vpc_bypass"
    assert ident.account_id == "internal_vpc"


def test_api_key_success():
    db = MagicMock()
    key_row = MagicMock()
    key_row.key_id = "gk_1"
    key_row.account_id = "alice"
    db.find_api_key.return_value = key_row
    r = _req(auth=AuthSettings(bypass_vpc=False), db=db,
             headers={"x-api-key": "s3cr3t", "host": "public.example.com"})
    ident = require_auth(r)
    assert ident.method == "api_key"
    assert ident.account_id == "alice"
    db.find_api_key.assert_called_once_with(hash_secret("s3cr3t"))


def test_no_creds_401():
    db = MagicMock()
    db.find_api_key.return_value = None
    r = _req(auth=AuthSettings(bypass_vpc=False), db=db,
             headers={"host": "public.example.com"})
    with pytest.raises(HTTPException) as e:
        require_auth(r)
    assert e.value.status_code == 401


def test_jwt_success():
    priv, jwks = _keypair_and_jwks()
    url = "https://fake.example/deps-jwks-1.json"
    jv._clear_cache(url)
    now = datetime.now(timezone.utc)
    tok = _sign(
        priv,
        {
            "sub": "acme",
            "iat": now,
            "exp": now + timedelta(hours=1),
            "aud": "gateway-server",
            "jti": "jti-1",
        },
    )
    db = MagicMock()
    r = _req(
        auth=_jwt_auth(url),
        db=db,
        headers={"authorization": f"Bearer {tok}", "host": "public.example.com"},
    )
    with patch("server.auth.jwt_verifier.httpx.get", lambda u, timeout: _JwksResp(jwks)):
        ident = require_auth(r)
    jv._clear_cache(url)
    assert ident.method == "jwt"
    assert ident.account_id == "acme"
    assert ident.raw_token_id == "jti-1"
    db.find_api_key.assert_not_called()


def test_jwt_failure_falls_through_to_api_key():
    url = "https://fake.example/deps-jwks-2.json"
    jv._clear_cache(url)
    db = MagicMock()
    key_row = MagicMock()
    key_row.key_id = "gk_fb"
    key_row.account_id = "fallback"
    db.find_api_key.return_value = key_row
    r = _req(
        auth=_jwt_auth(url),
        db=db,
        headers={
            "authorization": "Bearer not.a.real.token",
            "x-api-key": "sekret",
            "host": "public.example.com",
        },
    )
    ident = require_auth(r)
    jv._clear_cache(url)
    assert ident.method == "api_key"
    assert ident.account_id == "fallback"


def test_vpc_beats_jwt():
    r = _req(
        headers={
            "host": "fc-gateway-x.cn-hangzhou-vpc.fcapp.run",
            "authorization": "Bearer whatever",
        }
    )
    ident = require_auth(r)
    assert ident.method == "vpc_bypass"
