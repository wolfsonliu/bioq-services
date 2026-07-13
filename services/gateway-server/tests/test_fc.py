"""Live gateway-server smoke/integration tests (opt-in).

Marked ``@pytest.mark.fc``; skipped by default. Run with::

    RUN_FC_TESTS=1 uv run python -m pytest \\
        services/gateway-server/tests/test_fc.py -v

Credentials + base URL come from ``services/gateway-server/tests/.env``
(gitignored), or from the environment:

    GATEWAY_BASE_URL   e.g. http://172.27.167.158:9000
    GATEWAY_API_KEY    the seeded X-API-Key secret
    GATEWAY_PRINCIPAL  the principal that key maps to (for uri assertions)

These hit the REAL gateway (and, for discovery/presign, the real downstream
services + OSS). They do NOT launch GPU compute — `/v1/run` is intentionally
not exercised here.
"""

from __future__ import annotations

import hashlib
import io
import os
import time
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest


def _load_env(path: Path) -> None:
    """Minimal .env loader (stdlib) — sets vars not already in the environment."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


_load_env(Path(__file__).resolve().parent / ".env")

BASE_URL = os.environ.get("GATEWAY_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("GATEWAY_API_KEY", "")
PRINCIPAL = os.environ.get("GATEWAY_PRINCIPAL", "")

TIMEOUT = httpx.Timeout(connect=10, read=60, write=60, pool=10)

_needs = pytest.mark.skipif(
    not (BASE_URL and API_KEY),
    reason="set GATEWAY_BASE_URL + GATEWAY_API_KEY (services/gateway-server/tests/.env)",
)


@pytest.fixture(scope="module")
def client():
    with httpx.Client(
        base_url=BASE_URL, timeout=TIMEOUT, headers={"X-API-Key": API_KEY}
    ) as c:
        yield c


# ===================================================================
# Smoke (no auth needed)
# ===================================================================


@pytest.mark.fc
@_needs
class TestSmoke:
    def test_healthz(self, client):
        body = client.get("/healthz").json()
        assert body["status"] == "ok"
        assert body["service"] == "gateway"

    def test_openapi_registers_v1(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        for p in ("/v1/services", "/v1/run/{svc}/{endpoint}", "/v1/uploads/presign"):
            assert p in paths, f"missing {p}"


# ===================================================================
# Auth
# ===================================================================


@pytest.mark.fc
@_needs
class TestAuth:
    def test_no_key_401(self):
        # No X-API-Key, non-VPC host → must be rejected.
        r = httpx.get(f"{BASE_URL}/v1/services", timeout=TIMEOUT)
        assert r.status_code == 401

    def test_bad_key_401(self):
        r = httpx.get(
            f"{BASE_URL}/v1/services",
            headers={"X-API-Key": "definitely-wrong"},
            timeout=TIMEOUT,
        )
        assert r.status_code == 401

    def test_valid_key_200(self, client):
        assert client.get("/v1/services").status_code == 200


# ===================================================================
# Discovery
# ===================================================================


@pytest.mark.fc
@_needs
class TestDiscovery:
    def test_services_list(self, client):
        services = client.get("/v1/services").json()["services"]
        assert isinstance(services, list)
        assert "openbpmd-server" in services

    def test_describe_downstream(self, client):
        info = client.get("/v1/services/openbpmd-server").json()
        assert info["service"] == "openbpmd-server"
        # gateway -> downstream over VPC: OpenAPI must resolve
        assert "/api/score" in (info.get("openapi") or {}).get("paths", {})
        # manifest should be populated (discovery robustness fix)
        assert info.get("manifest"), "manifest empty — downstream /api/manifest not fetched"

    def test_describe_unknown_404(self, client):
        assert client.get("/v1/services/nope-server").status_code == 404


# ===================================================================
# Presign (OSS) — cheap, no GPU
# ===================================================================


@pytest.mark.fc
@_needs
class TestPresign:
    def test_presign_mint(self, client):
        job_id = uuid.uuid4().hex[:20]
        r = client.post(
            "/v1/uploads/presign",
            json={"job_id": job_id, "filename": "smoke.txt", "sha256": uuid.uuid4().hex},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["exists"] is False
        assert body["url"], "no presigned URL returned"
        if PRINCIPAL:
            assert f"users/{PRINCIPAL}/{job_id}/input/smoke.txt" in body["uri"]

    def test_presign_upload_and_dedup(self, client):
        """presign -> PUT to OSS -> re-presign sees the object (exists=True)."""
        job_id = uuid.uuid4().hex[:20]
        payload = {"job_id": job_id, "filename": "smoke.txt", "sha256": uuid.uuid4().hex}
        first = client.post("/v1/uploads/presign", json=payload).json()
        assert first["exists"] is False and first["url"]

        put = httpx.put(first["url"], content=b"hello-gateway", timeout=TIMEOUT)
        assert put.status_code in (200, 201), f"OSS PUT failed: {put.status_code} {put.text!r}"

        second = client.post("/v1/uploads/presign", json=payload).json()
        assert second["exists"] is True, "dedup: object should be found after upload"
        assert second["url"] is None
        assert second["uri"] == first["uri"]


# ===================================================================
# End-to-end: real job through the gateway (proteinmpnn — fast)
# ===================================================================

_PDB = (
    Path(__file__).resolve().parents[2]
    / "proteinmpnn-server" / "tests" / "data" / "5L33.pdb"
)

RUN_POLL_TIMEOUT_S = 900
RUN_POLL_INTERVAL_S = 10


def _upload_via_presign(client, job_id: str, filename: str, data: bytes) -> str:
    sha = hashlib.sha256(data).hexdigest()
    pre = client.post(
        "/v1/uploads/presign",
        json={"job_id": job_id, "filename": filename, "sha256": sha},
    ).json()
    if not pre["exists"]:
        put = httpx.put(pre["url"], content=data, timeout=TIMEOUT)
        assert put.status_code in (200, 201), f"OSS PUT failed: {put.status_code} {put.text!r}"
    return pre["uri"]


@pytest.mark.fc
@_needs
@pytest.mark.skipif(not _PDB.exists(), reason=f"fixture missing: {_PDB}")
class TestEndToEndProteinMPNN:
    """Full path: client-generated job_id -> job-centric upload -> run -> poll -> 302 download."""

    def test_design_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        pdb_uri = _upload_via_presign(client, job_id, "5L33.pdb", _PDB.read_bytes())

        r = client.post(
            "/v1/run/proteinmpnn-server/design",
            headers={"X-Bioagent-Job-Id": job_id},
            json={
                "pdb_uri": pdb_uri, "name": "gwtest", "num_seq_per_target": 2,
                "model_variant": "vanilla", "model_name": "v_48_020",
                "sampling_temp": "0.1", "seed": 37,
            },
        )
        assert r.status_code == 202, r.text
        assert r.json()["job_id"] == job_id

        body = {}
        deadline = time.time() + RUN_POLL_TIMEOUT_S
        while time.time() < deadline:
            body = client.get(f"/v1/jobs/{job_id}").json()
            if body["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(RUN_POLL_INTERVAL_S)
        assert body.get("status") == "completed", f"job {job_id} ended: {body}"

        # Assert the download is served via OSS (302 presigned GET), i.e. the
        # output-sink mirrored results.zip — NOT the downstream proxy fallback.
        red = client.get(f"/v1/jobs/{job_id}/download", follow_redirects=False)
        assert red.status_code == 302, (
            f"expected OSS 302 (output-sink mirror), got {red.status_code}: {red.text}"
        )
        assert "results.zip" in red.headers["location"]
        dl = httpx.get(red.headers["location"], timeout=TIMEOUT)  # pull straight from OSS
        assert dl.status_code == 200, dl.text
        names = zipfile.ZipFile(io.BytesIO(dl.content)).namelist()
        assert any(n.endswith((".fa", ".fasta")) for n in names), f"no FASTA in {names}"
