"""Tests for the MCP Streamable-HTTP transport mounted by ``attach_mcp``.

The MCP SDK enables DNS-rebinding protection on its StreamableHTTP transport
by default, which rejects any Host header outside 127.0.0.1/localhost. That
breaks any production deployment behind FC / k8s ingress / a public hostname
(returning HTTP 421 "Invalid Host header"). The framework disables that
protection in `make_mcp_server`; this test guards against accidental
re-enabling.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict

from bioagent_service import JobAdapter, JobInfo, ServiceSettings, attach_mcp, create_app


class _PingRequest(BaseModel):
    payload: str = "pong"


class _PingSettings(ServiceSettings):
    model_config = SettingsConfigDict(env_file=None, env_prefix="MCP_TEST_", extra="ignore")


class _PingAdapter(JobAdapter):
    name = "ping"


@pytest.fixture
def mcp_client(tmp_path: Path) -> TestClient:
    settings = _PingSettings(jobs_base_dir=tmp_path / "jobs", max_concurrent_jobs=1)
    adapter = _PingAdapter(settings=settings)
    app = create_app(adapter, settings, title="Ping Test")

    @app.post("/api/ping", response_model=JobInfo)
    def ping(_req: _PingRequest):
        return app.state.runner.submit(
            build_argv=lambda _job_id, job_dir: ["true"],
            label="ping",
        )

    attach_mcp(app)
    # `with TestClient(...)` triggers the FastAPI startup events that start the
    # MCP session manager. Without that, /mcp/mcp returns 503.
    with TestClient(app) as client:
        yield client


def _init_payload() -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "probe", "version": "0.1"},
        },
    }


def test_initialize_with_localhost_host_header(mcp_client: TestClient):
    """Sanity: localhost Host header (the historical default) keeps working."""
    r = mcp_client.post(
        "/mcp/mcp",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        json=_init_payload(),
    )
    assert r.status_code == 200, r.text
    assert r.headers.get("mcp-session-id")
    assert '"result"' in r.text


def test_initialize_with_public_host_header(mcp_client: TestClient):
    """Production fix: a Host header from FC / ingress / public DNS must NOT be
    rejected as ``Invalid Host header`` (HTTP 421)."""
    r = mcp_client.post(
        "/mcp/mcp",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Host": "fc-something-pahlhbttzb.cn-hangzhou.fcapp.run",
        },
        json=_init_payload(),
    )
    assert r.status_code == 200, (
        f"FC hostname rejected — DNS rebinding protection likely re-enabled. "
        f"status={r.status_code} body={r.text[:200]}"
    )
    assert r.headers.get("mcp-session-id")


def test_initialize_with_arbitrary_host_header(mcp_client: TestClient):
    """Any host header should pass — the protection is fully disabled."""
    r = mcp_client.post(
        "/mcp/mcp",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Host": "example.com",
        },
        json=_init_payload(),
    )
    assert r.status_code == 200, r.text
