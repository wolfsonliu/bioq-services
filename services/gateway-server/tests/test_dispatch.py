from __future__ import annotations

import httpx

from server.dispatch import HttpDispatch


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://svc.local")


def test_submit_sends_async_headers():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["inv"] = request.headers.get("x-fc-invocation-type")
        seen["task"] = request.headers.get("x-fc-async-task-id")
        return httpx.Response(202, json={"status": "accepted"})

    d = HttpDispatch(_client(handler))
    d.submit("https://svc.local", "score", "job-1", {"nreps": "1"})
    assert seen["path"] == "/api/tasks/score"
    assert seen["inv"] == "Async"
    assert seen["task"] == "job-1"


def test_submit_treats_409_as_ok():
    d = HttpDispatch(_client(lambda req: httpx.Response(409, json={"detail": "dup"})))
    d.submit("https://svc.local", "score", "job-1", {})  # must not raise


def test_status_normalizes():
    d = HttpDispatch(_client(lambda req: httpx.Response(200, json={"status": "completed"})))
    assert d.status("https://svc.local", "job-1")["status"] == "completed"


def test_submit_sets_job_id_header_and_str_forms():
    seen = {}

    def handler(request):
        seen["job"] = request.headers.get("x-bioagent-job-id")
        seen["body"] = request.content
        return httpx.Response(202, json={})

    d = HttpDispatch(_client(handler))
    d.submit("https://svc.local", "score", "job-9", {"nreps": 1})
    assert seen["job"] == "job-9"
    assert b"nreps=1" in seen["body"]


def test_status_sends_affinity_header():
    seen = {}

    def handler(request):
        seen["aff"] = request.headers.get("x-bioagent-session-id")
        return httpx.Response(200, json={"status": "running"})

    d = HttpDispatch(_client(handler))
    d.status("https://svc.local", "job-7")
    assert seen["aff"] == "job-7"


def test_download_writes_file_with_affinity(tmp_path):
    seen = {}

    def handler(request):
        seen["aff"] = request.headers.get("x-bioagent-session-id")
        return httpx.Response(200, content=b"ZIPBYTES")

    d = HttpDispatch(_client(handler))
    dest = tmp_path / "sub" / "job-3.zip"
    out = d.download("https://svc.local", "job-3", dest)
    assert out == dest
    assert dest.read_bytes() == b"ZIPBYTES"
    assert seen["aff"] == "job-3"
