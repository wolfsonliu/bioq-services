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
