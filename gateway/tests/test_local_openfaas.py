"""Gateway functional test against a LOCAL OpenFaaS-mode deployment.

Adapted from each service's test_fc.py / test_fc_task.py (which target Alibaba
FC + OSS) for a self-hosted gateway running with
GATEWAY_DISPATCH_BACKEND=openfaas + GATEWAY_STORAGE_BACKEND=file. Differences
from the FC/OSS tests:

  * uploads PUT through the gateway's own /v1/files route (relative URL + API key),
    not a full presigned OSS URL;
  * download follows the 302 to the same-origin /v1/files route (API key carried),
    not an external OSS GET;
  * inputs are always passed by ``*_uri`` form field (the gateway dispatches form
    fields only — it can't multipart-upload files), mirroring the FC task tests'
    ``file://`` bootstrap pattern.

The end-to-end / describe tests are parametrized over a spec per CPU service
(see ``SPECS``); each is auto-skipped unless that service is actually deployed
(queried from ``/v1/services``). So ``make local-up LOCAL_SERVICES="dockq-server
plip-server"`` then ``make local-test`` exercises exactly what's up.

Run against the deployed gateway:

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import httpx
import pytest

BASE_URL = os.environ.get("GATEWAY_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("GATEWAY_API_KEY", "")
TIMEOUT = 60.0
POLL_INTERVAL_S = int(os.environ.get("LOCAL_POLL_INTERVAL_S", "5"))
# Global cap; per-service overrides via ServiceSpec.poll_timeout_s (clamped to this).
POLL_TIMEOUT_S = int(os.environ.get("LOCAL_POLL_TIMEOUT_S", "900"))

_REPO = Path(__file__).resolve().parents[2]

_needs = pytest.mark.skipif(
    not (os.environ.get("RUN_LOCAL_TESTS") and BASE_URL and API_KEY),
    reason="set RUN_LOCAL_TESTS=1 + GATEWAY_BASE_URL + GATEWAY_API_KEY",
)


# --- per-service specs ------------------------------------------------------

@dataclass(frozen=True)
class ServiceSpec:
    service: str                      # registry name, e.g. "plip-server"
    endpoint: str                     # task endpoint, e.g. "profile" -> /v1/run/<svc>/profile
    task_paths: tuple[str, ...]       # expected /api/tasks/... surfaced by describe
    inputs: tuple[tuple[str, str], ...]   # (uri_field, data_filename) uploaded via /v1/files
    result_suffix: str                # a filename that must appear in results.zip
    params: dict[str, str] = field(default_factory=dict)  # extra form fields
    assert_content: Optional[Callable[[bytes], None]] = None  # optional deeper check
    poll_timeout_s: int = 300

    @property
    def data_dir(self) -> Path:
        return _REPO / "services" / self.service / "tests" / "data"


def _assert_dockq(raw: bytes) -> None:
    result = json.loads(raw)
    assert "best_dockq" in result and "GlobalDockQ" in result, list(result)
    assert isinstance(result["best_dockq"], (int, float))


SPECS: tuple[ServiceSpec, ...] = (
    ServiceSpec(
        service="dockq-server",
        endpoint="score",
        task_paths=("/api/tasks/score", "/api/tasks/score_batch"),
        inputs=(("model_uri", "model.pdb"), ("native_uri", "native.pdb")),
        params={"name": "run"},
        result_suffix="run.json",
        assert_content=_assert_dockq,
    ),
    ServiceSpec(
        service="plip-server",
        endpoint="profile",
        task_paths=("/api/tasks/profile",),
        inputs=(("input_pdb_uri", "1vsn.pdb"),),
        params={"name": "run"},
        result_suffix="run.xml",
    ),
    ServiceSpec(
        service="diamond-server",
        endpoint="blastp",
        task_paths=("/api/tasks/blastp",),
        inputs=(("query_uri", "query.faa"), ("subject_uri", "subject.faa")),
        params={"name": "run"},
        result_suffix="run.tsv",
    ),
    ServiceSpec(
        service="lightdock-server",
        endpoint="dock",
        task_paths=("/api/tasks/dock",),
        inputs=(("receptor_uri", "receptor.pdb"), ("ligand_uri", "ligand.pdb")),
        # Tiny run: keep the CPU docking short (mirrors test_fc_task.py's TINY).
        params={"swarms": "2", "glowworms": "5", "steps": "3", "top": "3"},
        result_suffix="top_1.pdb",
        poll_timeout_s=600,
    ),
)

_IDS = [s.service for s in SPECS]


# --- fixtures + helpers -----------------------------------------------------

@pytest.fixture()
def client():
    return httpx.Client(
        base_url=BASE_URL,
        headers={"X-API-Key": API_KEY, "Host": "public.example.com"},
        timeout=TIMEOUT,
        follow_redirects=True,  # download 302 -> same-origin /v1/files (API key carried)
    )


_deployed_cache: Optional[set[str]] = None


def _deployed() -> set[str]:
    global _deployed_cache
    if _deployed_cache is None:
        with httpx.Client(
            base_url=BASE_URL,
            headers={"X-API-Key": API_KEY, "Host": "public.example.com"},
            timeout=TIMEOUT,
        ) as c:
            _deployed_cache = set(c.get("/v1/services").json().get("services", []))
    return _deployed_cache


def _skip_if_absent(spec: ServiceSpec) -> None:
    if spec.service not in _deployed():
        pytest.skip(f"{spec.service} not deployed")


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


# --- control plane ----------------------------------------------------------

@_needs
def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@_needs
def test_auth_required():
    r = httpx.get(f"{BASE_URL}/v1/services", headers={"Host": "public.example.com"}, timeout=TIMEOUT)
    assert r.status_code == 401


@_needs
def test_services_list_nonempty(client):
    r = client.get("/v1/services")
    assert r.status_code == 200
    assert r.json()["services"], "no services registered"


# --- per-service describe (task endpoints surface through the gateway) -------

@_needs
@pytest.mark.parametrize("spec", SPECS, ids=_IDS)
def test_describe(client, spec: ServiceSpec):
    _skip_if_absent(spec)
    r = client.get(f"/v1/services/{spec.service}")
    assert r.status_code == 200
    body = r.json()
    # describe must route through the OpenFaaS gateway to reach the worker (rec.url
    # is a placeholder in openfaas mode). An empty manifest/openapi means it fell
    # back to the placeholder URL — the CLI then reports "no runnable task
    # endpoints found". Assert the runnable task endpoints are actually surfaced.
    endpoints = (body.get("manifest") or {}).get("endpoints") or []
    paths = {e.get("path") for e in endpoints}
    assert set(spec.task_paths) <= paths, body
    assert (body.get("openapi") or {}).get("paths"), "openapi.json not discovered"


# --- per-service end-to-end (openfaas async submit -> poll -> download) ------

@_needs
@pytest.mark.parametrize("spec", SPECS, ids=_IDS)
def test_end_to_end(client, spec: ServiceSpec):
    _skip_if_absent(spec)
    job_id = uuid.uuid4().hex[:16]

    body: dict[str, str] = dict(spec.params)
    for uri_field, filename in spec.inputs:
        body[uri_field] = _upload(
            client, job_id, filename, (spec.data_dir / filename).read_bytes()
        )

    r = client.post(
        f"/v1/run/{spec.service}/{spec.endpoint}",
        headers={"X-Bioagent-Job-Id": job_id},
        json=body,
    )
    assert r.status_code == 202, r.text
    assert r.json()["job_id"] == job_id

    status: dict = {}
    deadline = time.time() + min(spec.poll_timeout_s, POLL_TIMEOUT_S)
    while time.time() < deadline:
        status = client.get(f"/v1/jobs/{job_id}").json()
        if status["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(POLL_INTERVAL_S)
    assert status.get("status") == "completed", f"job {job_id} ({spec.service}) ended: {status}"

    dl = client.get(f"/v1/jobs/{job_id}/download")
    assert dl.status_code == 200, dl.text
    zf = zipfile.ZipFile(io.BytesIO(dl.content))
    names = zf.namelist()
    match = next((n for n in names if n.endswith(spec.result_suffix)), None)
    assert match, f"no {spec.result_suffix!r} in results.zip for {spec.service}: {names}"

    if spec.assert_content is not None:
        spec.assert_content(zf.read(match))
