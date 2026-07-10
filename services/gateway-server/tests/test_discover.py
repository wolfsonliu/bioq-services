from __future__ import annotations

import httpx

from server.discover import Discovery


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_describe_merges_manifest_and_openapi():
    def handler(request):
        if request.url.path == "/api/manifest":
            return httpx.Response(200, json={"service": "openbpmd", "endpoints": ["score"]})
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json={"paths": {"/api/score": {}}})
        return httpx.Response(404)

    disc = Discovery(client=_client(handler), ttl_sec=60)
    info = disc.describe("openbpmd-server", "https://svc.local")
    assert info["manifest"]["service"] == "openbpmd"
    assert "/api/score" in info["openapi"]["paths"]


def test_describe_cached():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if request.url.path == "/api/manifest":
            return httpx.Response(200, json={"service": "x"})
        return httpx.Response(200, json={"paths": {}})

    disc = Discovery(client=_client(handler), ttl_sec=60)
    disc.describe("s", "https://svc.local")
    disc.describe("s", "https://svc.local")
    assert calls["n"] == 2  # one manifest + one openapi; second describe served from cache


def test_describe_errors_degrade():
    disc = Discovery(client=_client(lambda req: httpx.Response(500)), ttl_sec=60)
    info = disc.describe("s", "https://svc.local")
    assert info["service"] == "s"
    assert info["manifest"] == {}
    assert info["openapi"] == {}


def test_describe_ttl_expiry_refetches():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={})

    disc = Discovery(client=_client(handler), ttl_sec=0)  # 0 => always expired
    disc.describe("s", "https://svc.local")
    disc.describe("s", "https://svc.local")
    assert calls["n"] == 4  # 2 describes x (manifest + openapi), no caching at ttl=0
