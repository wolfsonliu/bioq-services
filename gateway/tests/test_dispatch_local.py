from __future__ import annotations

import json
from urllib.parse import parse_qs

import httpx
from bioq_service.service_registry import ServiceRecord

from server.dispatchers import LocalHttpDispatcher


def _rec(url: str = "https://svc.local", **kw) -> ServiceRecord:
    return ServiceRecord(url=url, **kw)


def _local(handler) -> LocalHttpDispatcher:
    return LocalHttpDispatcher(
        httpx.Client(transport=httpx.MockTransport(handler), base_url="https://svc.local")
    )


def test_submit_hits_plain_endpoint_without_fc_headers():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["inv"] = request.headers.get("x-fc-invocation-type")
        seen["task"] = request.headers.get("x-fc-async-task-id")
        seen["job"] = request.headers.get("x-bioagent-job-id")
        return httpx.Response(202, json={"job_id": "job-1"})

    _local(handler).submit(_rec(), "score", "job-1", {"nreps": 1})
    assert seen["path"] == "/api/score"          # NOT /api/tasks/score
    assert seen["inv"] is None                    # no FC headers
    assert seen["task"] is None
    assert seen["job"] == "job-1"


def test_submit_treats_409_as_ok():
    _local(lambda req: httpx.Response(409, json={"detail": "dup"})).submit(_rec(), "score", "j", {})


def test_submit_oss_prefix_and_form_encoding():
    seen = {}

    def handler(request):
        seen["prefix"] = request.headers.get("x-bioagent-oss-prefix")
        seen["form"] = parse_qs(request.content.decode())
        return httpx.Response(200, json={})

    _local(handler).submit(_rec(), "design", "j1",
                           {"chains": ["A", "B"], "n": 2, "skip": None}, oss_prefix="p/")
    assert seen["prefix"] == "p/"
    assert json.loads(seen["form"]["chains"][0]) == ["A", "B"]
    assert seen["form"]["n"] == ["2"]
    assert "skip" not in seen["form"]


def test_status_polls_jobs_endpoint_without_affinity():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["aff"] = request.headers.get("x-bioagent-session-id")
        return httpx.Response(200, json={"status": "completed"})

    out = _local(handler).status(_rec(), "job-1")
    assert seen["path"] == "/api/jobs/job-1"
    assert seen["aff"] is None
    assert out["status"] == "completed"


def test_download_writes_file(tmp_path):
    def handler(request):
        assert request.url.path == "/api/jobs/job-3/download"
        return httpx.Response(200, content=b"ZIPBYTES")

    dest = tmp_path / "sub" / "job-3.zip"
    out = _local(handler).download(_rec(), "job-3", dest)
    assert out == dest
    assert dest.read_bytes() == b"ZIPBYTES"


def test_download_error_has_readable_body(tmp_path):
    d = _local(lambda req: httpx.Response(404, text="nope"))
    try:
        d.download(_rec(), "jx", tmp_path / "o.zip")
        raise AssertionError("expected HTTPStatusError")
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 404
        assert "nope" in exc.response.text
