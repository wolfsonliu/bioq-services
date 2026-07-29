from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import pytest
from bioq_service.service_registry import ServiceRecord
from server.dispatchers import Dispatcher, OpenFaaSDispatcher, make_dispatcher

GW = "http://gw.openfaas:8080"


def _rec(function: str | None = "dockq-server", url: str = "http://placeholder") -> ServiceRecord:
    return ServiceRecord(url=url, function=function)


def _ofn(handler) -> OpenFaaSDispatcher:
    return OpenFaaSDispatcher(GW, httpx.Client(transport=httpx.MockTransport(handler)))


def test_satisfies_protocol():
    assert isinstance(OpenFaaSDispatcher(GW), Dispatcher)


def test_submit_hits_async_route_with_job_header_no_fc():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["job"] = request.headers.get("x-bioagent-job-id")
        seen["inv"] = request.headers.get("x-fc-invocation-type")
        return httpx.Response(202, json={"status": "accepted"})

    handle = _ofn(handler).submit(_rec(), "score", "job-1", {"nreps": 1})
    assert seen["url"] == f"{GW}/async-function/dockq-server/api/tasks/score"
    assert seen["job"] == "job-1"
    assert seen["inv"] is None
    assert handle is None  # execute_task keys by the id we passed


def test_submit_treats_409_as_ok():
    _ofn(lambda r: httpx.Response(409, json={"detail": "dup"})).submit(_rec(), "score", "j", {})


def test_submit_oss_prefix_and_form_encoding():
    seen = {}

    def handler(request):
        seen["prefix"] = request.headers.get("x-bioagent-oss-prefix")
        seen["form"] = parse_qs(request.content.decode())
        return httpx.Response(202, json={})

    _ofn(handler).submit(_rec(), "design", "j1",
                         {"chains": ["A", "B"], "n": 2, "skip": None}, oss_prefix="p/")
    assert seen["prefix"] == "p/"
    assert json.loads(seen["form"]["chains"][0]) == ["A", "B"]
    assert seen["form"]["n"] == ["2"]
    assert "skip" not in seen["form"]


def test_submit_raises_on_error_status():
    d = _ofn(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(httpx.HTTPStatusError):
        d.submit(_rec(), "score", "j", {})


def test_status_hits_sync_route():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"status": "completed"})

    out = _ofn(handler).status(_rec(), "job-1")
    assert seen["url"] == f"{GW}/function/dockq-server/api/jobs/job-1"
    assert out["status"] == "completed"


def test_download_writes_file(tmp_path):
    def handler(request):
        assert str(request.url) == f"{GW}/function/dockq-server/api/jobs/job-3/download"
        return httpx.Response(200, content=b"ZIPBYTES")

    dest = tmp_path / "sub" / "job-3.zip"
    out = _ofn(handler).download(_rec(), "job-3", dest)
    assert out == dest
    assert dest.read_bytes() == b"ZIPBYTES"


def test_missing_function_raises():
    d = _ofn(lambda r: httpx.Response(202, json={}))
    with pytest.raises(ValueError):
        d.submit(_rec(function=None), "score", "j", {})
    with pytest.raises(ValueError):
        d.status(_rec(function=None), "j")


# --- factory ---

def _settings(backend: str, gw: str = GW):
    return SimpleNamespace(dispatch_backend=backend, dispatch_timeout_sec=60.0,
                           openfaas_gateway_url=gw)


def test_make_dispatcher_openfaas():
    assert isinstance(make_dispatcher(_settings("openfaas")), OpenFaaSDispatcher)
    with pytest.raises(ValueError):
        make_dispatcher(_settings("openfaas", gw=""))


def test_describe_base_url_routes_through_gateway():
    # rec.url is a placeholder in openfaas mode; discovery must go via the gateway.
    assert _ofn(lambda r: httpx.Response(200)).describe_base_url(_rec()) == \
        f"{GW}/function/dockq-server"


def test_describe_base_url_requires_function():
    with pytest.raises(ValueError):
        _ofn(lambda r: httpx.Response(200)).describe_base_url(_rec(function=None))
