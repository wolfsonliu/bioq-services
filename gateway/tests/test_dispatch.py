from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import pytest
from bioq_service.service_registry import ServiceRecord

from server.dispatchers import (
    Dispatcher,
    FCDispatcher,
    LocalHttpDispatcher,
    make_dispatcher,
)
from server.fc_status import FcStatusClient


def _rec(url: str = "https://svc.local", **kw) -> ServiceRecord:
    return ServiceRecord(url=url, **kw)


def _fc(handler) -> FCDispatcher:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://svc.local")
    return FCDispatcher(FcStatusClient(), client)  # no AK/SK -> HTTP status path


# --- protocol ---

def test_dispatchers_satisfy_protocol():
    assert isinstance(FCDispatcher(FcStatusClient()), Dispatcher)
    assert isinstance(LocalHttpDispatcher(), Dispatcher)


# --- FCDispatcher: submit ---

def test_submit_sends_async_headers():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["inv"] = request.headers.get("x-fc-invocation-type")
        seen["task"] = request.headers.get("x-fc-async-task-id")
        return httpx.Response(202, json={"status": "accepted"})

    _fc(handler).submit(_rec(), "score", "job-1", {"nreps": "1"})
    assert seen["path"] == "/api/tasks/score"
    assert seen["inv"] == "Async"
    assert seen["task"] == "job-1"


def test_submit_treats_409_as_ok():
    _fc(lambda req: httpx.Response(409, json={"detail": "dup"})).submit(_rec(), "score", "job-1", {})


def test_submit_sets_job_id_header_and_str_forms():
    seen = {}

    def handler(request):
        seen["job"] = request.headers.get("x-bioagent-job-id")
        seen["body"] = request.content
        return httpx.Response(202, json={})

    _fc(handler).submit(_rec(), "score", "job-9", {"nreps": 1})
    assert seen["job"] == "job-9"
    assert b"nreps=1" in seen["body"]


def test_submit_sends_oss_prefix_header():
    seen = {}

    def handler(request):
        seen["prefix"] = request.headers.get("x-bioagent-oss-prefix")
        return httpx.Response(202, json={})

    _fc(handler).submit(_rec(), "score", "job-1", {}, oss_prefix="users/alice/job-1/")
    assert seen["prefix"] == "users/alice/job-1/"


def test_submit_json_encodes_structured_fields_and_drops_none():
    seen = {}

    def handler(request):
        seen["form"] = parse_qs(request.content.decode())
        return httpx.Response(202, json={})

    _fc(handler).submit(_rec(), "design", "j1",
                        {"chains": ["A", "B"], "bias": {"A": 0.1}, "n": 2, "name": "x", "skip": None})
    f = seen["form"]
    assert json.loads(f["chains"][0]) == ["A", "B"]
    assert json.loads(f["bias"][0]) == {"A": 0.1}
    assert f["n"] == ["2"] and f["name"] == ["x"]
    assert "skip" not in f


# --- FCDispatcher: status ---

def test_status_http_fallback_normalizes():
    d = _fc(lambda req: httpx.Response(200, json={"status": "completed"}))
    assert d.status(_rec(), "job-1")["status"] == "completed"


def test_status_sends_affinity_header_on_http_path():
    seen = {}

    def handler(request):
        seen["aff"] = request.headers.get("x-bioagent-session-id")
        return httpx.Response(200, json={"status": "running"})

    _fc(handler).status(_rec(), "job-7")
    assert seen["aff"] == "job-7"


def test_status_uses_fc_control_plane_when_function_and_enabled():
    # Injected FC client -> FcStatusClient.enabled is True; rec.function set ->
    # control-plane path (no HTTP call). GetAsyncTask "Succeeded" -> "completed".
    class _FakeFcClient:
        def get_async_task(self, function_name, task_id, request):
            return SimpleNamespace(body=SimpleNamespace(status="Succeeded"))

    def handler(request):  # must NOT be hit on the control-plane path
        raise AssertionError("HTTP status must not be called when FC control plane is used")

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://svc.local")
    d = FCDispatcher(FcStatusClient(client=_FakeFcClient()), http)
    assert d.status(_rec(function="fc-score"), "t1")["status"] == "completed"


# --- FCDispatcher: download ---

def test_download_error_has_readable_body(tmp_path):
    d = _fc(lambda req: httpx.Response(404, text="job not found"))
    try:
        d.download(_rec(), "job-x", tmp_path / "out.zip")
        raise AssertionError("expected HTTPStatusError")
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 404
        assert "job not found" in exc.response.text


def test_download_writes_file_with_affinity(tmp_path):
    seen = {}

    def handler(request):
        seen["aff"] = request.headers.get("x-bioagent-session-id")
        return httpx.Response(200, content=b"ZIPBYTES")

    dest = tmp_path / "sub" / "job-3.zip"
    out = _fc(handler).download(_rec(), "job-3", dest)
    assert out == dest
    assert dest.read_bytes() == b"ZIPBYTES"
    assert seen["aff"] == "job-3"


# --- factory ---

def _settings(backend: str):
    return SimpleNamespace(
        dispatch_backend=backend, dispatch_timeout_sec=60.0,
        ali_access_key_id="", ali_access_key_secret="",
        oss_region="cn-hangzhou", fc_endpoint="",
    )


def test_make_dispatcher_selects_backend():
    assert isinstance(make_dispatcher(_settings("http")), LocalHttpDispatcher)
    assert isinstance(make_dispatcher(_settings("fc")), FCDispatcher)
    with pytest.raises(ValueError):
        make_dispatcher(_settings("nope"))
