"""Gateway functional test against a LOCAL OpenFaaS-mode deployment.

Adapted from test_fc.py (which targets Alibaba FC + OSS) for a self-hosted
gateway running with GATEWAY_DISPATCH_BACKEND=openfaas + GATEWAY_STORAGE_BACKEND=file.
Differences from the FC/OSS test:

  * uploads PUT through the gateway's own /v1/files route (relative URL + API key),
    not a full presigned OSS URL;
  * download follows the 302 to the same-origin /v1/files route (API key carried),
    not an external OSS GET.

Run against the deployed gateway (see tmp/ kind + OpenFaaS setup):

    GATEWAY_BASE_URL=http://127.0.0.1:9000 \
    GATEWAY_API_KEY=<seeded secret> \
    RUN_LOCAL_TESTS=1 \
    uv run --with pytest --with pytest-asyncio python -m pytest tests/test_local_openfaas.py -v
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest

BASE_URL = os.environ.get("GATEWAY_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("GATEWAY_API_KEY", "")
TIMEOUT = 60.0
POLL_TIMEOUT_S = int(os.environ.get("LOCAL_POLL_TIMEOUT_S", "300"))
POLL_INTERVAL_S = int(os.environ.get("LOCAL_POLL_INTERVAL_S", "5"))

_DATA = Path(__file__).resolve().parents[2] / "services" / "dockq-server" / "tests" / "data"

_needs = pytest.mark.skipif(
    not (os.environ.get("RUN_LOCAL_TESTS") and BASE_URL and API_KEY),
    reason="set RUN_LOCAL_TESTS=1 + GATEWAY_BASE_URL + GATEWAY_API_KEY",
)


@pytest.fixture()
def client():
    return httpx.Client(
        base_url=BASE_URL,
        headers={"X-API-Key": API_KEY, "Host": "public.example.com"},
        timeout=TIMEOUT,
        follow_redirects=True,  # download 302 -> same-origin /v1/files (API key carried)
    )


def _upload(client: httpx.Client, job_id: str, filename: str, data: bytes) -> str:
    """Presign + PUT through the gateway (file backend). Returns the file:// uri."""
    sha = hashlib.sha256(data).hexdigest()
    pre = client.post(
        "/v1/uploads/presign",
        json={"job_id": job_id, "filename": filename, "sha256": sha},
    ).json()
    if not pre["exists"]:
        put = client.put(pre["url"], content=data)  # relative URL -> base_url, API key carried
        assert put.status_code in (200, 201), f"gateway PUT failed: {put.status_code} {put.text!r}"
    return pre["uri"]


# --- control plane ---

@_needs
def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@_needs
def test_auth_required():
    r = httpx.get(f"{BASE_URL}/v1/services", headers={"Host": "public.example.com"}, timeout=TIMEOUT)
    assert r.status_code == 401


@_needs
def test_services_list_has_dockq(client):
    r = client.get("/v1/services")
    assert r.status_code == 200
    assert "dockq-server" in r.json()["services"]


@_needs
def test_describe_dockq(client):
    r = client.get("/v1/services/dockq-server")
    assert r.status_code == 200


# --- dockq-server /api/score end-to-end (openfaas async) ---

@_needs
def test_score_end_to_end(client):
    job_id = uuid.uuid4().hex[:16]
    model_uri = _upload(client, job_id, "model.pdb", (_DATA / "model.pdb").read_bytes())
    native_uri = _upload(client, job_id, "native.pdb", (_DATA / "native.pdb").read_bytes())

    r = client.post(
        "/v1/run/dockq-server/score",
        headers={"X-Bioagent-Job-Id": job_id},
        json={"model_uri": model_uri, "native_uri": native_uri},
    )
    assert r.status_code == 202, r.text
    assert r.json()["job_id"] == job_id

    status: dict = {}
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        status = client.get(f"/v1/jobs/{job_id}").json()
        if status["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(POLL_INTERVAL_S)
    assert status.get("status") == "completed", f"job {job_id} ended: {status}"

    dl = client.get(f"/v1/jobs/{job_id}/download")
    assert dl.status_code == 200, dl.text
    names = zipfile.ZipFile(io.BytesIO(dl.content)).namelist()
    assert any(n.endswith("run.json") for n in names), f"no run.json in results.zip: {names}"

    # Assert real DockQ output (not just a completed status).
    run_name = next(n for n in names if n.endswith("run.json"))
    result = json.loads(zipfile.ZipFile(io.BytesIO(dl.content)).read(run_name))
    assert "best_dockq" in result and "GlobalDockQ" in result, f"unexpected DockQ output: {list(result)}"
    assert isinstance(result["best_dockq"], (int, float))
