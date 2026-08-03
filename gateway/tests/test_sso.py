from types import SimpleNamespace

import httpx
import pytest

from server.admin import sso


def _settings(**kw):
    auth = SimpleNamespace(oidc_issuer="https://idp/realms/bioq", oidc_client_id="cid",
                           oidc_client_secret="sec", jwt_jwks_url="https://idp/certs")
    for k, v in kw.items():
        setattr(auth, k, v)
    return SimpleNamespace(auth=auth)


class _Resp:
    def __init__(self, body):
        self._b = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._b


_META = {"authorization_endpoint": "https://idp/auth", "token_endpoint": "https://idp/token"}


@pytest.fixture(autouse=True)
def _clear_cache():
    sso._disc_cache.clear()
    yield
    sso._disc_cache.clear()


def test_sso_enabled():
    assert sso.sso_enabled(_settings())
    assert not sso.sso_enabled(_settings(oidc_client_secret=""))
    assert not sso.sso_enabled(_settings(jwt_jwks_url=""))


def test_discover_caches(monkeypatch):
    calls = {"n": 0}

    def get(url, **kw):
        calls["n"] += 1
        return _Resp(_META)

    monkeypatch.setattr(sso.httpx, "get", get)
    a = sso.discover("https://idp/realms/bioq")
    b = sso.discover("https://idp/realms/bioq")
    assert a == b == _META
    assert calls["n"] == 1  # second call served from cache


def test_authorize_url(monkeypatch):
    monkeypatch.setattr(sso.httpx, "get", lambda url, **kw: _Resp(_META))
    url = sso.authorize_url(_settings(), "https://gw/admin/auth/callback", "STATE123")
    assert url.startswith("https://idp/auth?")
    assert "client_id=cid" in url
    assert "state=STATE123" in url
    assert "redirect_uri=https%3A%2F%2Fgw%2Fadmin%2Fauth%2Fcallback" in url


def test_exchange_code(monkeypatch):
    monkeypatch.setattr(sso.httpx, "get", lambda url, **kw: _Resp(_META))
    captured = {}

    def post(url, **kw):
        captured["url"] = url
        captured["data"] = kw["data"]
        return _Resp({"access_token": "AT", "id_token": "IT"})

    monkeypatch.setattr(sso.httpx, "post", post)
    out = sso.exchange_code(_settings(), "the-code", "https://gw/admin/auth/callback")
    assert out["access_token"] == "AT"
    assert captured["url"] == "https://idp/token"
    assert captured["data"]["grant_type"] == "authorization_code"
    assert captured["data"]["code"] == "the-code"


def test_discover_error(monkeypatch):
    def get(url, **kw):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(sso.httpx, "get", get)
    with pytest.raises(sso.SSOError):
        sso.discover("https://idp/realms/bioq")
