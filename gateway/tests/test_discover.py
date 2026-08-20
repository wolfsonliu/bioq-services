from __future__ import annotations

import threading
import time

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
    assert info["status"] == "ok"
    assert info["source"] == "live"


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
    assert calls["n"] == 2  # one manifest + one openapi; second describe from cache


def test_describe_ttl_expiry_refetches():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if request.url.path == "/api/manifest":
            return httpx.Response(200, json={"service": "x"})
        return httpx.Response(200, json={"paths": {}})

    disc = Discovery(client=_client(handler), ttl_sec=0)
    disc.describe("s", "https://svc.local")
    disc.describe("s", "https://svc.local")
    assert calls["n"] == 4  # ttl=0 => always expired, refetch manifest + openapi


def test_describe_errors_degrade():
    disc = Discovery(client=_client(lambda req: httpx.Response(500)), ttl_sec=60)
    info = disc.describe("s", "https://svc.local")
    assert info["service"] == "s"
    assert info["manifest"] == {}
    assert info["openapi"] == {}
    assert info["status"] == "error"


def test_timeout_is_warming_and_short_circuits():
    state = {"openapi": 0}

    def handler(request):
        if request.url.path == "/api/manifest":
            raise httpx.ReadTimeout("cold")
        if request.url.path == "/openapi.json":
            state["openapi"] += 1
            return httpx.Response(200, json={"paths": {}})
        return httpx.Response(404)

    disc = Discovery(client=_client(handler), ttl_sec=60)
    info = disc.describe("s", "https://svc.local")
    assert info["status"] == "warming"
    assert info["manifest"] == {} and info["openapi"] == {}
    assert "detail" in info
    assert state["openapi"] == 0  # short-circuited: no openapi fetch after manifest fail


def test_404_is_no_manifest():
    disc = Discovery(client=_client(lambda req: httpx.Response(404)), ttl_sec=60)
    info = disc.describe("s", "https://svc.local")
    assert info["status"] == "no_manifest"
    assert "detail" in info


def test_negative_cache_avoids_refetch():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500)

    disc = Discovery(client=_client(handler), ttl_sec=60, negative_ttl_sec=60)
    disc.describe("s", "https://svc.local")
    disc.describe("s", "https://svc.local")
    assert calls["n"] == 1  # second describe served from negative cache


def test_describe_does_not_cache_partial_failure():
    state = {"openapi_fail": True}

    def handler(request):
        if request.url.path == "/api/manifest":
            return httpx.Response(200, json={"service": "x"})
        if request.url.path == "/openapi.json":
            return httpx.Response(500) if state["openapi_fail"] \
                else httpx.Response(200, json={"paths": {"/api/s": {}}})
        return httpx.Response(404)

    disc = Discovery(client=_client(handler), ttl_sec=300)
    first = disc.describe("s", "https://svc.local")
    assert first["status"] == "partial"
    assert first["manifest"] == {"service": "x"} and first["openapi"] == {}
    state["openapi_fail"] = False
    second = disc.describe("s", "https://svc.local")  # partial was NOT cached
    assert second["status"] == "ok"
    assert second["openapi"] == {"paths": {"/api/s": {}}}


def test_describe_does_not_cache_total_failure():
    state = {"fail": True}

    def handler(request):
        if state["fail"]:
            return httpx.Response(500)
        return httpx.Response(200, json={"service": "x"} if request.url.path == "/api/manifest" else {"paths": {}})

    disc = Discovery(client=_client(handler), ttl_sec=300, negative_ttl_sec=0)
    first = disc.describe("s", "https://svc.local")
    assert first["status"] == "error"
    state["fail"] = False
    second = disc.describe("s", "https://svc.local")  # negative ttl 0 => refetch
    assert second["manifest"] == {"service": "x"}


def test_single_flight_coalesces_concurrent():
    state = {"manifest": 0, "openapi": 0}
    guard = threading.Lock()
    gate = threading.Event()

    def handler(request):
        if request.url.path == "/api/manifest":
            with guard:
                state["manifest"] += 1
            gate.wait(timeout=2.0)
            return httpx.Response(200, json={"service": "x"})
        if request.url.path == "/openapi.json":
            with guard:
                state["openapi"] += 1
            return httpx.Response(200, json={"paths": {}})
        return httpx.Response(404)

    disc = Discovery(client=_client(handler), ttl_sec=60)
    results = []

    def worker():
        results.append(disc.describe("s", "https://svc.local"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    gate.set()
    for t in threads:
        t.join()

    assert state["manifest"] == 1
    assert state["openapi"] == 1
    assert all(r["status"] == "ok" for r in results)
