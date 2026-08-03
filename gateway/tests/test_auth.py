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
from server.auth.deps import require_admin, require_auth
from server.db.store import GatewayDB
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


def test_no_creds_401():
    r = _req(auth=AuthSettings(bypass_vpc=False), db=MagicMock(),
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


def test_jwt_failure_is_401():
    # An invalid Bearer with no VPC bypass → 401 (no api-key fallback anymore).
    r = _req(
        auth=_jwt_auth("https://fake.example/deps-jwks-2.json"),
        db=MagicMock(),
        headers={"authorization": "Bearer not.a.real.token", "host": "public.example.com"},
    )
    with pytest.raises(HTTPException) as e:
        require_auth(r)
    assert e.value.status_code == 401


def test_vpc_beats_jwt():
    r = _req(
        headers={
            "host": "fc-gateway-x.cn-hangzhou-vpc.fcapp.run",
            "authorization": "Bearer whatever",
        }
    )
    ident = require_auth(r)
    assert ident.method == "vpc_bypass"


def _admin_db(role: str | None):
    db = MagicMock()
    if role is None:
        db.get_user.return_value = None
    else:
        u = MagicMock()
        u.role = role
        db.get_user.return_value = u
    return db


def test_require_admin_vpc_bypass_is_admin_by_default():
    r = _req(headers={"host": "fc-gateway-x.cn-hangzhou-vpc.fcapp.run"})
    ident = require_admin(r)
    assert ident.method == "vpc_bypass"


def test_require_admin_vpc_not_admin_when_disabled():
    r = _req(auth=AuthSettings(vpc_is_admin=False), db=_admin_db(None),
             headers={"host": "fc-gateway-x.cn-hangzhou-vpc.fcapp.run"})
    with pytest.raises(HTTPException) as e:
        require_admin(r)
    assert e.value.status_code == 403


# --- OIDC: groups->role mapping + JIT provisioning on JWT auth ---
def _jwt_req(url, tok, db, auth=None):
    return _req(auth=auth or _jwt_auth(url), db=db,
                headers={"authorization": f"Bearer {tok}", "host": "public.example.com"})


def _oidc_token(priv, *, sub, groups, extra=None):
    now = datetime.now(timezone.utc)
    payload = {"sub": sub, "iat": now, "exp": now + timedelta(hours=1),
               "aud": "gateway-server", "jti": "j", "groups": groups}
    if extra:
        payload.update(extra)
    return _sign(priv, payload)


def test_jwt_admin_group_provisions_admin(tmp_path):
    priv, jwks = _keypair_and_jwks()
    url = "https://fake.example/oidc-a.json"; jv._clear_cache(url)
    db = GatewayDB(f"sqlite:///{tmp_path/'gw.db'}"); db.create_all()
    tok = _oidc_token(priv, sub="u1", groups=["bioq-admins"],
                      extra={"preferred_username": "u1"})
    with patch("server.auth.jwt_verifier.httpx.get", lambda u, timeout: _JwksResp(jwks)):
        ident = require_auth(_jwt_req(url, tok, db))
    jv._clear_cache(url)
    assert ident.account_id == "u1" and ident.method == "jwt"
    assert db.get_user("u1").role == "admin"
    assert db.get_user("u1").display_name == "u1"


def test_jwt_non_admin_group_is_user(tmp_path):
    priv, jwks = _keypair_and_jwks()
    url = "https://fake.example/oidc-b.json"; jv._clear_cache(url)
    db = GatewayDB(f"sqlite:///{tmp_path/'gw.db'}"); db.create_all()
    tok = _oidc_token(priv, sub="u2", groups=["some-other-group"])
    with patch("server.auth.jwt_verifier.httpx.get", lambda u, timeout: _JwksResp(jwks)):
        require_auth(_jwt_req(url, tok, db))
    jv._clear_cache(url)
    assert db.get_user("u2").role == "user"


def test_jwt_role_syncs_on_relogin(tmp_path):
    priv, jwks = _keypair_and_jwks()
    url = "https://fake.example/oidc-c.json"; jv._clear_cache(url)
    db = GatewayDB(f"sqlite:///{tmp_path/'gw.db'}"); db.create_all()
    with patch("server.auth.jwt_verifier.httpx.get", lambda u, timeout: _JwksResp(jwks)):
        require_auth(_jwt_req(url, _oidc_token(priv, sub="u3", groups=["bioq-admins"]), db))
        assert db.get_user("u3").role == "admin"
        require_auth(_jwt_req(url, _oidc_token(priv, sub="u3", groups=[]), db))
    jv._clear_cache(url)
    assert db.get_user("u3").role == "user"     # IdP 组变化被同步


def test_jwt_custom_admin_group(tmp_path):
    priv, jwks = _keypair_and_jwks()
    url = "https://fake.example/oidc-d.json"; jv._clear_cache(url)
    db = GatewayDB(f"sqlite:///{tmp_path/'gw.db'}"); db.create_all()
    auth = AuthSettings(bypass_vpc=False, jwt_jwks_url=url, jwt_audience="gateway-server",
                        jwt_admin_group="platform-admins")
    tok = _oidc_token(priv, sub="u4", groups=["platform-admins"])
    with patch("server.auth.jwt_verifier.httpx.get", lambda u, timeout: _JwksResp(jwks)):
        require_auth(_jwt_req(url, tok, db, auth=auth))
    jv._clear_cache(url)
    assert db.get_user("u4").role == "admin"


def test_require_admin_via_jwt(tmp_path):
    priv, jwks = _keypair_and_jwks()
    url = "https://fake.example/oidc-e.json"; jv._clear_cache(url)
    db = GatewayDB(f"sqlite:///{tmp_path/'gw.db'}"); db.create_all()
    tok = _oidc_token(priv, sub="u5", groups=["bioq-admins"])
    with patch("server.auth.jwt_verifier.httpx.get", lambda u, timeout: _JwksResp(jwks)):
        ident = require_admin(_jwt_req(url, tok, db))
    jv._clear_cache(url)
    assert ident.account_id == "u5" and ident.method == "jwt"
