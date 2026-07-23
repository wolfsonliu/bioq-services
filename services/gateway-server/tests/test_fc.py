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

import csv
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
import yaml


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


def _fixture(svc: str, name: str) -> bytes:
    """Read a downstream service's test fixture (services/<svc>/tests/data/<name>)."""
    return (Path(__file__).resolve().parents[2] / svc / "tests" / "data" / name).read_bytes()


# ---------------------------------------------------------------------------
# Per-session download persistence — mirror each downloaded results.zip (plus
# the terminal JobInfo and its extracted tree) under tests/fc_outputs/ so humans
# can inspect real gateway outputs. Gitignored (see .gitignore). Grouped by
# <svc>/<endpoint>-<job_id> since the gateway spans many downstream services.
# ---------------------------------------------------------------------------

_FC_OUTPUTS_ROOT = Path(__file__).resolve().parent / "fc_outputs"
_FC_RUN_STAMP = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
_fc_run_dir: Path | None = None


def _fc_outputs_dir() -> Path:
    """Lazily create (once per session) tests/fc_outputs/run-<UTC stamp>/."""
    global _fc_run_dir
    if _fc_run_dir is None:
        _fc_run_dir = _FC_OUTPUTS_ROOT / f"run-{_FC_RUN_STAMP}"
        _fc_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[fc_outputs] saving downloaded job outputs under {_fc_run_dir}")
    return _fc_run_dir


def _save_gateway_outputs(
    svc: str, endpoint: str, job_id: str, job_info: dict, zip_bytes: bytes,
) -> None:
    """Persist a completed job's JobInfo + results.zip + extracted tree."""
    dst = _fc_outputs_dir() / svc / f"{endpoint}-{job_id}"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "jobinfo.json").write_text(json.dumps(job_info, indent=2))
    if zip_bytes:
        (dst / f"{job_id}.zip").write_bytes(zip_bytes)
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                zf.extractall(dst / "extracted")
        except zipfile.BadZipFile as exc:  # noqa: BLE001
            print(f"[fc_outputs] extract failed for {job_id}: {exc!r}")
    print(f"[fc_outputs] saved {job_id} → {dst}")


def _run_poll_zip(
    client,
    *,
    svc: str,
    endpoint: str,
    job_id: str,
    body: dict,
    poll_timeout_s: int,
    poll_interval_s: int = 30,
) -> bytes:
    """Shared run -> poll -> 302-OSS-download for gateway e2e tests.

    Asserts: 202 submit, terminal status=completed, and the download is an OSS
    302 (output-sink mirror, NOT the downstream proxy fallback). Returns the raw
    results.zip bytes so callers can assert on / extract from its contents.
    """
    r = client.post(
        f"/v1/run/{svc}/{endpoint}",
        headers={"X-Bioagent-Job-Id": job_id},
        json=body,
    )
    assert r.status_code == 202, r.text
    assert r.json()["job_id"] == job_id

    status: dict = {}
    deadline = time.time() + poll_timeout_s
    while time.time() < deadline:
        try:
            status = client.get(f"/v1/jobs/{job_id}").json()
        except httpx.TransportError as exc:  # transient network blip mid-poll
            print(f"[poll] transient error for {job_id}, retrying: {exc!r}")
            time.sleep(poll_interval_s)
            continue
        if status["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(poll_interval_s)
    assert status.get("status") == "completed", f"job {job_id} ended: {status}"

    red = client.get(f"/v1/jobs/{job_id}/download", follow_redirects=False)
    assert red.status_code == 302, (
        f"expected OSS 302 (output-sink mirror), got {red.status_code}: {red.text}"
    )
    assert "results.zip" in red.headers["location"]
    dl = httpx.get(red.headers["location"], timeout=TIMEOUT)  # pull straight from OSS
    assert dl.status_code == 200, dl.text
    _save_gateway_outputs(svc, endpoint, job_id, status, dl.content)
    return dl.content


def _run_poll_download(
    client,
    *,
    svc: str,
    endpoint: str,
    job_id: str,
    body: dict,
    output_suffixes: tuple[str, ...],
    poll_timeout_s: int,
    poll_interval_s: int = 30,
) -> None:
    """Run -> poll -> 302-OSS-download, asserting results.zip carries a file
    whose name ends with one of `output_suffixes`."""
    content = _run_poll_zip(
        client, svc=svc, endpoint=endpoint, job_id=job_id, body=body,
        poll_timeout_s=poll_timeout_s, poll_interval_s=poll_interval_s,
    )
    names = zipfile.ZipFile(io.BytesIO(content)).namelist()
    assert any(n.endswith(output_suffixes) for n in names), (
        f"no {output_suffixes} in results.zip: {names}"
    )


def _zip_member(content: bytes, needle: str) -> bytes:
    """Extract the first results.zip member whose name contains `needle`."""
    zf = zipfile.ZipFile(io.BytesIO(content))
    name = next((n for n in zf.namelist() if needle in n), None)
    assert name is not None, f"no member matching {needle!r} in {zf.namelist()}"
    return zf.read(name)


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
        _save_gateway_outputs("proteinmpnn-server", "design", job_id, body, dl.content)
        names = zipfile.ZipFile(io.BytesIO(dl.content)).namelist()
        assert any(n.endswith((".fa", ".fasta")) for n in names), f"no FASTA in {names}"

    def test_score_end_to_end(self, client):
        """Score-only path: writes per-position scores as score_only .npz."""
        job_id = uuid.uuid4().hex[:20]
        pdb_uri = _upload_via_presign(client, job_id, "5L33.pdb", _PDB.read_bytes())
        content = _run_poll_zip(
            client, svc="proteinmpnn-server", endpoint="score", job_id=job_id,
            body={"pdb_uri": pdb_uri, "name": "gw_score", "num_seq_per_target": 2},
            poll_timeout_s=RUN_POLL_TIMEOUT_S,
        )
        names = zipfile.ZipFile(io.BytesIO(content)).namelist()
        assert any("score_only" in n for n in names), f"no score_only in {names}"

    def test_probs_end_to_end(self, client):
        """Conditional probability path: writes conditional_probs_only .npz."""
        job_id = uuid.uuid4().hex[:20]
        pdb_uri = _upload_via_presign(client, job_id, "5L33.pdb", _PDB.read_bytes())
        content = _run_poll_zip(
            client, svc="proteinmpnn-server", endpoint="probs", job_id=job_id,
            body={"pdb_uri": pdb_uri, "name": "gw_probs", "kind": "conditional"},
            poll_timeout_s=RUN_POLL_TIMEOUT_S,
        )
        names = zipfile.ZipFile(io.BytesIO(content)).namelist()
        assert any("conditional_probs_only" in n for n in names), (
            f"no conditional_probs_only in {names}"
        )


# ===================================================================
# End-to-end: real job through the gateway (alphafold — slow)
# ===================================================================

# 76-residue ubiquitin — same sequence alphafold-server's own tests use; small
# enough that reduced_dbs MSA + monomer inference fits FC's instance lifetime.
_AF_FASTA = (
    b">test_ubiquitin\n"
    b"MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG\n"
)

# Small heterodimer (ubiquitin + a short helical peptide), ~130 residues total —
# same pair alphafold-server's own multimer test uses. Multimer runs an extra
# Jackhmmer pass against uniprot.fasta (must be staged on NAS) for paired MSAs.
_AF_MULTIMER_FASTA = (
    b">chainA\n"
    b"MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG\n"
    b">chainB\n"
    b"GSHMGSAEYELDPKQIDDLEKEIATLEKERLALEKERLALEKERLALEKERLALEK\n"
)

# AlphaFold (MSA search + inference) is far slower than proteinmpnn; give it up
# to an hour before declaring the job stuck.
AF_RUN_POLL_TIMEOUT_S = 3600
AF_RUN_POLL_INTERVAL_S = 30

# Multimer adds a UniProt Jackhmmer pass + paired MSA; allow up to 2h.
AF_MULTIMER_POLL_TIMEOUT_S = 7200


@pytest.mark.fc
@_needs
class TestEndToEndAlphaFold:
    """Full path for alphafold-server: upload FASTA -> run -> poll -> 302 download.

    Mirrors TestEndToEndProteinMPNN but for a slow GPU service. Uses monomer_ptm
    + reduced_dbs + models_to_relax=none to keep runtime bounded (still tens of
    minutes). Asserts the download is an OSS 302 (output-sink mirror), not the
    downstream proxy fallback.
    """

    def test_fold_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        fasta_uri = _upload_via_presign(client, job_id, "input.fasta", _AF_FASTA)

        r = client.post(
            "/v1/run/alphafold-server/fold",
            headers={"X-Bioagent-Job-Id": job_id},
            json={
                "input_fasta_uri": fasta_uri,
                "model_preset": "monomer_ptm",
                "db_preset": "reduced_dbs",
                "models_to_relax": "none",
                "random_seed": 37,
            },
        )
        assert r.status_code == 202, r.text
        assert r.json()["job_id"] == job_id

        body = {}
        deadline = time.time() + AF_RUN_POLL_TIMEOUT_S
        while time.time() < deadline:
            body = client.get(f"/v1/jobs/{job_id}").json()
            if body["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(AF_RUN_POLL_INTERVAL_S)
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
        _save_gateway_outputs("alphafold-server", "fold", job_id, body, dl.content)
        names = zipfile.ZipFile(io.BytesIO(dl.content)).namelist()
        assert any(n.endswith(".pdb") for n in names), f"no predicted PDB in {names}"

    def test_fold_multimer_end_to_end(self, client):
        """Multimer preset over a small heterodimer. Requires the uniprot.fasta
        Jackhmmer DB on NAS (extra paired-MSA pass monomer presets skip)."""
        job_id = uuid.uuid4().hex[:20]
        fasta_uri = _upload_via_presign(client, job_id, "input.fasta", _AF_MULTIMER_FASTA)
        _run_poll_download(
            client, svc="alphafold-server", endpoint="fold", job_id=job_id,
            body={
                "input_fasta_uri": fasta_uri,
                "model_preset": "multimer",
                "db_preset": "reduced_dbs",
                "models_to_relax": "none",
                "num_multimer_predictions_per_model": 1,
                "random_seed": 37,
            },
            output_suffixes=(".pdb",), poll_timeout_s=AF_MULTIMER_POLL_TIMEOUT_S,
        )


# ===================================================================
# End-to-end: real job through the gateway (mmseqs2 — inline FASTA, slow MSA)
# ===================================================================

# Unlike proteinmpnn/alphafold, mmseqs2 uses the ColabFold protocol: the FASTA
# is passed inline in the `q` form field (no file upload / presign). Same 52-aa
# monomer + mode="all" (UniRef30-only) that mmseqs2-server's own async task test
# uses, so the run does NOT require the colabfold_envdb to be staged on NAS.
_MMSEQS_MONOMER = "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVN"
_MMSEQS_Q = f">probe1\n{_MMSEQS_MONOMER}\n"
# Two-chain heterodimer for the paired pipeline (mode="pairgreedy"), same shape
# as mmseqs2-server's own /api/tasks/pair test.
_MMSEQS_PAIRED_Q = (
    f">chainA\n{_MMSEQS_MONOMER}\n"
    ">chainB\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDY\n"
)

# mmseqs MSA cold-start ~60s; short-seq search runs 3-10 min on the GPU subset
# DB. Allow 40 min (matches the downstream's own test) to absorb cold start.
MMSEQS_RUN_POLL_TIMEOUT_S = 2400
MMSEQS_RUN_POLL_INTERVAL_S = 30


@pytest.mark.fc
@_needs
class TestEndToEndMMseqs2:
    """Full path for mmseqs2-server: inline-FASTA run -> poll -> 302 download.

    Mirrors TestEndToEndProteinMPNN but the input is the ColabFold `q` form
    field carried in the JSON run body (the gateway form-encodes str values
    as-is), so there is no presign upload step. Asserts the download is an OSS
    302 (output-sink mirror), not the downstream proxy fallback, and that the
    mirrored results.zip carries the MSA `.a3m`.
    """

    def test_msa_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]

        r = client.post(
            "/v1/run/mmseqs2-server/msa",
            headers={"X-Bioagent-Job-Id": job_id},
            json={"q": _MMSEQS_Q, "mode": "all"},
        )
        assert r.status_code == 202, r.text
        assert r.json()["job_id"] == job_id

        body = {}
        deadline = time.time() + MMSEQS_RUN_POLL_TIMEOUT_S
        while time.time() < deadline:
            body = client.get(f"/v1/jobs/{job_id}").json()
            if body["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(MMSEQS_RUN_POLL_INTERVAL_S)
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
        _save_gateway_outputs("mmseqs2-server", "msa", job_id, body, dl.content)
        names = zipfile.ZipFile(io.BytesIO(dl.content)).namelist()
        assert any(n.endswith(".a3m") for n in names), f"no MSA a3m in {names}"

    def test_pair_end_to_end(self, client):
        """Paired multimer MSA (mode=pairgreedy) over a 2-chain query."""
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="mmseqs2-server", endpoint="pair", job_id=job_id,
            body={"q": _MMSEQS_PAIRED_Q, "mode": "pairgreedy"},
            output_suffixes=(".a3m",), poll_timeout_s=MMSEQS_RUN_POLL_TIMEOUT_S,
        )


# ===================================================================
# End-to-end: remaining services (one class each, shared via helpers).
#
# Each test uploads any file inputs via presign (-> oss:// URI passed in the
# `<name>_uri` body field) then delegates to _run_poll_download, which asserts
# the OSS 302 output-sink download. Params are the minimal working sets lifted
# from each service's own FC tests. These are opt-in (@pytest.mark.fc) and are
# NOT run here — the services must be redeployed with the OSS-mount migration
# (and the gateway with the {endpoint:path} routing fix) first.
# ===================================================================


@pytest.mark.fc
@_needs
class TestEndToEndRFdiffusion:
    """RFdiffusion exposes 5 generation modes, each with a ``tasks/`` variant
    (motif / binder / generate-custom / unconditional / symmetry), so all 5 are
    dispatchable through the gateway. unconditional and symmetry carry all inputs
    inline in the run body (no file upload)."""

    def test_generate_motif_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        input_uri = _upload_via_presign(
            client, job_id, "5TPN.pdb", _fixture("rfdiffusion-server", "5TPN.pdb")
        )
        _run_poll_download(
            client, svc="rfdiffusion-server", endpoint="generate/motif", job_id=job_id,
            body={
                "input_uri": input_uri,
                "contigs": "10-40/A163-181/10-40",
                "num_designs": 1, "diffuser_t": 25,
            },
            output_suffixes=(".pdb",), poll_timeout_s=1800,
        )

    def test_generate_binder_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        input_uri = _upload_via_presign(
            client, job_id, "insulin_target.pdb",
            _fixture("rfdiffusion-server", "insulin_target.pdb"),
        )
        _run_poll_download(
            client, svc="rfdiffusion-server", endpoint="generate/binder", job_id=job_id,
            body={
                "input_uri": input_uri,
                "contigs": "A1-150/0 70-70", "hotspots": "A59,A83,A91",
                "num_designs": 1, "diffuser_t": 25,
            },
            output_suffixes=(".pdb",), poll_timeout_s=1800,
        )

    def test_generate_custom_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="rfdiffusion-server", endpoint="generate", job_id=job_id,
            body={
                "contigs": "60-60", "config_name": "base",
                "num_designs": 1, "diffuser_t": 25,
            },
            output_suffixes=(".pdb",), poll_timeout_s=1800,
        )

    def test_generate_unconditional_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="rfdiffusion-server", endpoint="generate/unconditional", job_id=job_id,
            body={
                "num_designs": 1, "diffuser_t": 25,
                "min_length": 60, "max_length": 60,
            },
            output_suffixes=(".pdb",), poll_timeout_s=1800,
        )

    def test_generate_symmetry_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="rfdiffusion-server", endpoint="generate/symmetry", job_id=job_id,
            body={
                "symmetry": "c2", "total_length": 80,
                "num_designs": 1, "diffuser_t": 25,
            },
            output_suffixes=(".pdb",), poll_timeout_s=1800,
        )


@pytest.mark.fc
@_needs
class TestEndToEndRFdiffusion2:
    def test_generate_active_site_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        input_uri = _upload_via_presign(
            client, job_id, "M0584_1ldm.pdb",
            _fixture("rfdiffusion2-server", "M0584_1ldm.pdb"),
        )
        _run_poll_download(
            client, svc="rfdiffusion2-server", endpoint="generate/active_site", job_id=job_id,
            body={
                "input_uri": input_uri,
                "contigs": "46,A106-106,59,A166-166,2,A169-169,23,A193-193,46",
                "contig_atoms": {
                    "A106": "NE,CD,CZ", "A166": "OD1,CG",
                    "A169": "NH2,CZ", "A193": "NE2,CD2,CE1",
                },
                "ligand": "NAD,OXM", "contig_as_guidepost": True,
                "num_designs": 1, "diffuser_t": 10,
            },
            output_suffixes=(".pdb",), poll_timeout_s=1800,
        )

    def test_generate_small_molecule_binder_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        input_uri = _upload_via_presign(
            client, job_id, "trimmed_ec2_M0151_NO_ORI_zero_com0.pdb",
            _fixture("rfdiffusion2-server", "trimmed_ec2_M0151_NO_ORI_zero_com0.pdb"),
        )
        _run_poll_download(
            client, svc="rfdiffusion2-server",
            endpoint="generate/small_molecule_binder", job_id=job_id,
            body={
                "input_uri": input_uri,
                "contigs": "50", "length": "50-50", "ligand": "PH2",
                "rasa_active": True, "rasa_target": 0,
                "num_designs": 1, "diffuser_t": 10,
            },
            output_suffixes=(".pdb",), poll_timeout_s=1800,
        )

    def test_generate_custom_end_to_end(self, client):
        """Freeform endpoint — active-site via raw contigs + extra_overrides."""
        job_id = uuid.uuid4().hex[:20]
        input_uri = _upload_via_presign(
            client, job_id, "M0584_1ldm.pdb",
            _fixture("rfdiffusion2-server", "M0584_1ldm.pdb"),
        )
        _run_poll_download(
            client, svc="rfdiffusion2-server", endpoint="generate", job_id=job_id,
            body={
                "input_uri": input_uri,
                "contigs": "46,A106-106,59,A166-166,2,A169-169,23,A193-193,46",
                "config_name": "aa", "input_pdb_required": True,
                "ligand": "NAD,OXM", "num_designs": 1, "diffuser_t": 10,
                "extra_overrides": {
                    "inference.contig_as_guidepost": True,
                    "contigmap.contig_atoms": {
                        "A106": "NE,CD,CZ", "A166": "OD1,CG",
                        "A169": "NH2,CZ", "A193": "NE2,CD2,CE1",
                    },
                },
            },
            output_suffixes=(".pdb",), poll_timeout_s=1800,
        )


@pytest.mark.fc
@_needs
class TestEndToEndRFantibody:
    """RFantibody is a 3-stage pipeline through the gateway:

      rfdiffusion (target + framework PDB) -> 1_rfdiffusion.qv
      proteinmpnn (1_rfdiffusion.qv)       -> 2_proteinmpnn.qv
      rf2         (2_proteinmpnn.qv)        -> 3_rf2.qv

    The service's own tests chain the stages via ``job://`` URIs. Here each
    stage's ``.qv`` is pulled from the gateway's OSS results.zip and re-uploaded
    via presign as the next stage's ``oss://`` input, so proteinmpnn and rf2 are
    exercised through the gateway (upload -> run -> poll -> 302 OSS download)
    without job:// cross-job coupling.
    """

    def _rfdiffusion_qv(self, client) -> bytes:
        """Run stage 1 and return the 1_rfdiffusion.qv bytes."""
        job_id = uuid.uuid4().hex[:20]
        target_uri = _upload_via_presign(
            client, job_id, "rsv_site3.pdb", _fixture("rfantibody-server", "rsv_site3.pdb")
        )
        framework_uri = _upload_via_presign(
            client, job_id, "hu-4D5-8_Fv.pdb", _fixture("rfantibody-server", "hu-4D5-8_Fv.pdb")
        )
        content = _run_poll_zip(
            client, svc="rfantibody-server", endpoint="rfdiffusion", job_id=job_id,
            body={
                "target_uri": target_uri, "framework_uri": framework_uri,
                "num_designs": 1, "diffuser_t": 25,
                "design_loops": "L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13",
                "hotspots": "T305,T456", "deterministic": True,
            },
            poll_timeout_s=1800,
        )
        return _zip_member(content, "1_rfdiffusion.qv")

    def test_rfdiffusion_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        target_uri = _upload_via_presign(
            client, job_id, "rsv_site3.pdb", _fixture("rfantibody-server", "rsv_site3.pdb")
        )
        framework_uri = _upload_via_presign(
            client, job_id, "hu-4D5-8_Fv.pdb", _fixture("rfantibody-server", "hu-4D5-8_Fv.pdb")
        )
        _run_poll_download(
            client, svc="rfantibody-server", endpoint="rfdiffusion", job_id=job_id,
            body={
                "target_uri": target_uri, "framework_uri": framework_uri,
                "num_designs": 1, "diffuser_t": 25,
                "design_loops": "L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13",
                "hotspots": "T305,T456", "deterministic": True,
            },
            output_suffixes=(".qv",), poll_timeout_s=1800,
        )

    def test_pipeline_proteinmpnn_rf2_end_to_end(self, client):
        # stage 1: rfdiffusion -> 1_rfdiffusion.qv (re-uploaded for stage 2)
        rfd_qv = self._rfdiffusion_qv(client)

        # stage 2: proteinmpnn (1_rfdiffusion.qv -> 2_proteinmpnn.qv)
        mpnn_job = uuid.uuid4().hex[:20]
        mpnn_input_uri = _upload_via_presign(client, mpnn_job, "1_rfdiffusion.qv", rfd_qv)
        mpnn_zip = _run_poll_zip(
            client, svc="rfantibody-server", endpoint="proteinmpnn", job_id=mpnn_job,
            body={"input_uri": mpnn_input_uri, "seqs_per_struct": 1, "deterministic": True},
            poll_timeout_s=1800,
        )
        mpnn_qv = _zip_member(mpnn_zip, "2_proteinmpnn.qv")

        # stage 3: rf2 (2_proteinmpnn.qv -> 3_rf2.qv)
        rf2_job = uuid.uuid4().hex[:20]
        rf2_input_uri = _upload_via_presign(client, rf2_job, "2_proteinmpnn.qv", mpnn_qv)
        _run_poll_download(
            client, svc="rfantibody-server", endpoint="rf2", job_id=rf2_job,
            body={"input_uri": rf2_input_uri, "num_recycles": 2},
            output_suffixes=(".qv",), poll_timeout_s=1800,
        )


@pytest.mark.fc
@_needs
class TestEndToEndPPIflow:
    """PPIFlow has 5 sample modes. The gateway can reach the 4 with a
    ``tasks/`` variant (binder / antibody / nanobody / monomer); scaffolding is
    omitted because its motif CSV references PDBs that must be pre-staged on NAS
    (not self-contained). File inputs are uploaded via presign so the antigen's
    128 KiB async-payload cap doesn't apply — the run body carries only URIs.
    """

    def test_sample_binder_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        target_uri = _upload_via_presign(
            client, job_id, "1IJZ_IL13.pdb", _fixture("ppiflow-server", "1IJZ_IL13.pdb")
        )
        _run_poll_download(
            client, svc="ppiflow-server", endpoint="sample/binder", job_id=job_id,
            body={
                "target_uri": target_uri, "target_chain": "C", "binder_chain": "A",
                "specified_hotspots": "C11,C14,C15",
                "samples_min_length": 60, "samples_max_length": 70,
                "samples_per_target": 1, "name": "gw_binder",
            },
            output_suffixes=(".pdb",), poll_timeout_s=1800,
        )

    def test_sample_antibody_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        antigen_uri = _upload_via_presign(
            client, job_id, "1IJZ_IL13.pdb", _fixture("ppiflow-server", "1IJZ_IL13.pdb")
        )
        framework_uri = _upload_via_presign(
            client, job_id, "6nou_scfv_framework.pdb",
            _fixture("ppiflow-server", "6nou_scfv_framework.pdb"),
        )
        _run_poll_download(
            client, svc="ppiflow-server", endpoint="sample/antibody", job_id=job_id,
            body={
                "antigen_uri": antigen_uri, "framework_uri": framework_uri,
                "antigen_chain": "C", "heavy_chain": "A", "light_chain": "B",
                "specified_hotspots": "C11,C14,C15",
                "cdr_length": "CDRH1,8-8,CDRH2,8-8,CDRH3,10-10,CDRL1,7-7,CDRL2,3-3,CDRL3,9-9",
                "samples_per_target": 1, "name": "gw_antibody",
            },
            output_suffixes=(".pdb",), poll_timeout_s=1800,
        )

    def test_sample_nanobody_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        antigen_uri = _upload_via_presign(
            client, job_id, "1IJZ_IL13.pdb", _fixture("ppiflow-server", "1IJZ_IL13.pdb")
        )
        framework_uri = _upload_via_presign(
            client, job_id, "7eow_nanobody_framework.pdb",
            _fixture("ppiflow-server", "7eow_nanobody_framework.pdb"),
        )
        _run_poll_download(
            client, svc="ppiflow-server", endpoint="sample/nanobody", job_id=job_id,
            body={
                "antigen_uri": antigen_uri, "framework_uri": framework_uri,
                "antigen_chain": "C", "heavy_chain": "A",
                "specified_hotspots": "C11,C14,C15",
                "cdr_length": "CDRH1,8-8,CDRH2,8-8,CDRH3,10-10",
                "samples_per_target": 1, "name": "gw_nanobody",
            },
            output_suffixes=(".pdb",), poll_timeout_s=1800,
        )

    def test_sample_monomer_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="ppiflow-server", endpoint="sample/monomer", job_id=job_id,
            body={"length_subset": [40], "samples_per_target": 1, "name": "gw_monomer"},
            output_suffixes=(".pdb",), poll_timeout_s=1800,
        )


@pytest.mark.fc
@_needs
class TestEndToEndIgGM:
    def test_design_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        fasta_uri = _upload_via_presign(
            client, job_id, "ab_CDR_H3.fasta", _fixture("iggm-server", "ab_CDR_H3.fasta")
        )
        antigen_uri = _upload_via_presign(
            client, job_id, "antigen.pdb", _fixture("iggm-server", "antigen.pdb")
        )
        _run_poll_download(
            client, svc="iggm-server", endpoint="design", job_id=job_id,
            body={
                "fasta_uri": fasta_uri, "antigen_uri": antigen_uri,
                "run_task": "design", "steps": 5, "num_samples": 1, "seed": 42,
            },
            output_suffixes=(".pdb",), poll_timeout_s=1800,
        )

    def test_epitope_end_to_end(self, client):
        """Epitope prediction over an antibody-antigen complex -> epitope.json."""
        job_id = uuid.uuid4().hex[:20]
        fasta_uri = _upload_via_presign(
            client, job_id, "complex.fasta", _fixture("iggm-server", "complex.fasta")
        )
        # complex.fasta chain A matches complex.pdb (not antigen.pdb, whose chain A
        # is a different RBD variant); cal_ppi loads the receptor chain with the
        # fasta sequence, so the structure and fasta must be the same complex.
        antigen_uri = _upload_via_presign(
            client, job_id, "complex.pdb", _fixture("iggm-server", "complex.pdb")
        )
        content = _run_poll_zip(
            client, svc="iggm-server", endpoint="epitope", job_id=job_id,
            body={"fasta_uri": fasta_uri, "antigen_uri": antigen_uri},
            poll_timeout_s=600,
        )
        names = zipfile.ZipFile(io.BytesIO(content)).namelist()
        assert any("epitope.json" in n for n in names), f"no epitope.json in {names}"


@pytest.mark.fc
@_needs
class TestEndToEndDockQ:
    """Full-endpoint coverage of dockq's two task endpoints via the gateway.

      * `score`        — single (model, native) pair; both files uploaded via
        presign and passed as `model_uri` / `native_uri`.
      * `score_batch`  — 1 native + N models. The gateway can't multipart-upload
        the `models` list, so the models are bundled into a zip, uploaded via
        presign, and passed as `models_zip_uri` (oss://...); the gateway rewrites
        it to /mnt/oss/... and dockq extracts it as the models set (needs
        dockq-server >= v0.0.12). Mirrors genie3's `dataset_uri` zip pattern.

    Assertions dig into the mirrored results.zip (DockQ JSON fields, scores.csv
    ordering, per_model JSONs) rather than just checking a file suffix — the
    same checks dockq-server's own test_fc.py makes on its native outputs.
    """

    def test_score_end_to_end(self, client):
        """Single score; assert the DockQ JSON in results.zip carries a valid
        headline score + best_result with DockQ in [0, 1]."""
        job_id = uuid.uuid4().hex[:20]
        model_uri = _upload_via_presign(
            client, job_id, "model.pdb", _fixture("dockq-server", "model.pdb")
        )
        native_uri = _upload_via_presign(
            client, job_id, "native.pdb", _fixture("dockq-server", "native.pdb")
        )
        content = _run_poll_zip(
            client, svc="dockq-server", endpoint="score", job_id=job_id,
            body={"model_uri": model_uri, "native_uri": native_uri, "name": "fc_smoke"},
            poll_timeout_s=600,
        )
        data = json.loads(_zip_member(content, "fc_smoke.json"))
        assert any(k in data for k in ("GlobalDockQ", "total_DockQ", "DockQ")), data.keys()
        assert data.get("best_result"), f"no best_result: {data.keys()}"
        for iface in data["best_result"].values():
            assert 0.0 <= iface["DockQ"] <= 1.0, iface
            assert isinstance(iface["iRMSD"], (int, float))

    def test_score_no_align_end_to_end(self, client):
        """--no_align flag path — trusts residue numbering, still produces JSON."""
        job_id = uuid.uuid4().hex[:20]
        model_uri = _upload_via_presign(
            client, job_id, "model.pdb", _fixture("dockq-server", "model.pdb")
        )
        native_uri = _upload_via_presign(
            client, job_id, "native.pdb", _fixture("dockq-server", "native.pdb")
        )
        _run_poll_download(
            client, svc="dockq-server", endpoint="score", job_id=job_id,
            body={
                "model_uri": model_uri, "native_uri": native_uri,
                "name": "fc_noalign", "no_align": True,
            },
            output_suffixes=(".json",), poll_timeout_s=600,
        )

    def test_score_batch_end_to_end(self, client):
        """Batch of 2 models via models_zip_uri; assert scores.csv (2 rows,
        sorted by DockQ desc) and per-model JSONs in results.zip.

        Both models must share native.pdb's chain set (A/B/C) for DockQ to map
        them — model_alt.pdb is a different complex (A/B/H/L) that DockQ rightly
        refuses to score. We batch model.pdb (a real prediction, DockQ < 1) with
        native.pdb itself (self-score, DockQ == 1) to get two mappable models
        with distinct scores, so the descending-sort assertion is meaningful.
        """
        job_id = uuid.uuid4().hex[:20]
        native_uri = _upload_via_presign(
            client, job_id, "native.pdb", _fixture("dockq-server", "native.pdb")
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.pdb", _fixture("dockq-server", "model.pdb"))
            zf.writestr("self.pdb", _fixture("dockq-server", "native.pdb"))
        models_zip_uri = _upload_via_presign(client, job_id, "models.zip", buf.getvalue())
        content = _run_poll_zip(
            client, svc="dockq-server", endpoint="score_batch", job_id=job_id,
            body={
                "native_uri": native_uri, "models_zip_uri": models_zip_uri,
                "sort_by": "DockQ", "name": "fc_batch",
            },
            poll_timeout_s=1800,
        )
        names = zipfile.ZipFile(io.BytesIO(content)).namelist()
        assert "scores.csv" in names, names
        assert sum(n.startswith("per_model/") and n.endswith(".json") for n in names) == 2, names

        rows = list(csv.DictReader(io.StringIO(_zip_member(content, "scores.csv").decode())))
        assert len(rows) == 2, rows
        assert {"model", "DockQ", "iRMSD", "n_interfaces"} <= set(rows[0].keys())
        dockq_vals = [float(r["DockQ"]) for r in rows]
        assert all(0.0 <= v <= 1.0 for v in dockq_vals), dockq_vals
        assert dockq_vals == sorted(dockq_vals, reverse=True), f"not DockQ-desc: {dockq_vals}"


@pytest.mark.fc
@_needs
class TestEndToEndDeepRankAb:
    def test_score_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        input_pdb_uri = _upload_via_presign(
            client, job_id, "test.pdb", _fixture("deeprank-ab-server", "test.pdb")
        )
        _run_poll_download(
            client, svc="deeprank-ab-server", endpoint="score", job_id=job_id,
            body={
                "input_pdb_uri": input_pdb_uri,
                "heavy_chain_id": "H", "light_chain_id": "L", "antigen_chain_id": "A",
            },
            output_suffixes=(".csv",), poll_timeout_s=1800,
        )


@pytest.mark.fc
@_needs
class TestEndToEndPlip:
    def test_profile_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        input_pdb_uri = _upload_via_presign(
            client, job_id, "1vsn.pdb", _fixture("plip-server", "1vsn.pdb")
        )
        content = _run_poll_zip(
            client, svc="plip-server", endpoint="profile", job_id=job_id,
            body={
                "input_pdb_uri": input_pdb_uri, "name": "gwtest",
                "pymol_session": True,
            },
            poll_timeout_s=1200,
        )
        names = zipfile.ZipFile(io.BytesIO(content)).namelist()
        xml = _zip_member(content, "gwtest.xml").decode()
        assert "<report" in xml
        # pymol_session=True → one PyMOL session (.pse) per binding site.
        assert any(n.endswith(".pse") for n in names), f"no .pse in results.zip: {names}"


@pytest.mark.fc
@_needs
class TestEndToEndDiffDock:
    def test_dock_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        protein_uri = _upload_via_presign(
            client, job_id, "protein.pdb", _fixture("diffdock-server", "1a0q_protein.pdb")
        )
        ligand_uri = _upload_via_presign(
            client, job_id, "ligand.sdf", _fixture("diffdock-server", "1a0q_ligand.sdf")
        )
        _run_poll_download(
            client, svc="diffdock-server", endpoint="dock", job_id=job_id,
            body={
                "protein_uri": protein_uri, "ligand_uri": ligand_uri,
                "complex_name": "1a0q_pdb_sdf", "samples_per_complex": 3,
                "inference_steps": 10, "actual_steps": 10, "batch_size": 3, "seed": 42,
            },
            output_suffixes=(".sdf",), poll_timeout_s=1200,
        )


@pytest.mark.fc
@_needs
class TestEndToEndDiffDockPP:
    def test_dock_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        receptor_uri = _upload_via_presign(
            client, job_id, "receptor.pdb", _fixture("diffdock-pp-server", "1a2k_receptor.pdb")
        )
        ligand_uri = _upload_via_presign(
            client, job_id, "ligand.pdb", _fixture("diffdock-pp-server", "1a2k_ligand.pdb")
        )
        _run_poll_download(
            client, svc="diffdock-pp-server", endpoint="dock", job_id=job_id,
            body={
                "receptor_uri": receptor_uri, "ligand_uri": ligand_uri,
                "num_samples": 4, "top_k": 2, "use_confidence_model": True, "seed": 42,
            },
            output_suffixes=(".pdb",), poll_timeout_s=1200,
        )


@pytest.mark.fc
@_needs
class TestEndToEndOpenBPMD:
    def test_score_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        structure_uri = _upload_via_presign(
            client, job_id, "solvated.rst7", _fixture("openbpmd-server", "solvated.rst7")
        )
        parameters_uri = _upload_via_presign(
            client, job_id, "solvated.prm7", _fixture("openbpmd-server", "solvated.prm7")
        )
        _run_poll_download(
            client, svc="openbpmd-server", endpoint="score", job_id=job_id,
            body={
                "structure_uri": structure_uri, "parameters_uri": parameters_uri,
                "lig_resname": "UNK", "nreps": 1, "sim_ns": 0.02, "equil_steps": 500,
            },
            output_suffixes=(".csv",), poll_timeout_s=1800,
        )


@pytest.mark.fc
@_needs
class TestEndToEndDiffusionHopping:
    """Single `generate` endpoint, two checkpoint sets: gvp_conditional (DiffHopp
    paper main variant) and egnn_conditional. Same protein + reference-ligand
    inputs; only model_variant differs."""

    def _run_variant(self, client, variant: str) -> None:
        job_id = uuid.uuid4().hex[:20]
        protein_uri = _upload_via_presign(
            client, job_id, "1a0q_protein.pdb",
            _fixture("diffusion-hopping-server", "1a0q_protein.pdb"),
        )
        reference_ligand_uri = _upload_via_presign(
            client, job_id, "1a0q_ligand.sdf",
            _fixture("diffusion-hopping-server", "1a0q_ligand.sdf"),
        )
        _run_poll_download(
            client, svc="diffusion-hopping-server", endpoint="generate", job_id=job_id,
            body={
                "protein_uri": protein_uri, "reference_ligand_uri": reference_ligand_uri,
                "num_samples": 3, "model_variant": variant,
            },
            output_suffixes=(".sdf",), poll_timeout_s=1200,
        )

    def test_generate_end_to_end(self, client):
        self._run_variant(client, "gvp_conditional")

    def test_generate_egnn_end_to_end(self, client):
        self._run_variant(client, "egnn_conditional")


@pytest.mark.fc
@_needs
class TestEndToEndDrugHive:
    def test_generate_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        target_uri = _upload_via_presign(
            client, job_id, "5d3h_pocket.pdb", _fixture("drughive-server", "5d3h_pocket.pdb")
        )
        ligand_uri = _upload_via_presign(
            client, job_id, "5d3h_ligand.sdf", _fixture("drughive-server", "5d3h_ligand.sdf")
        )
        _run_poll_download(
            client, svc="drughive-server", endpoint="generate", job_id=job_id,
            body={
                "target_uri": target_uri, "ligand_uri": ligand_uri,
                "n_samples": 5, "pdb_id": "5d3h", "zbetas": 0.0, "temps": 0.5,
            },
            output_suffixes=(".sdf",), poll_timeout_s=900,
        )

    def test_generate_spatial_end_to_end(self, client):
        """Scaffold hopping via a SMARTS substructure pattern (no extra file)."""
        job_id = uuid.uuid4().hex[:20]
        target_uri = _upload_via_presign(
            client, job_id, "5d3h_pocket.pdb", _fixture("drughive-server", "5d3h_pocket.pdb")
        )
        ligand_uri = _upload_via_presign(
            client, job_id, "5d3h_ligand.sdf", _fixture("drughive-server", "5d3h_ligand.sdf")
        )
        _run_poll_download(
            client, svc="drughive-server", endpoint="generate_spatial", job_id=job_id,
            body={
                "target_uri": target_uri, "ligand_uri": ligand_uri,
                "n_samples": 5, "pdb_id": "5d3h",
                "substruct_modify_pattern": "C(=O)N",  # amide; 1 match in 5d3h ligand
                "zbetas": 0.3, "temps": 1.0,
            },
            output_suffixes=(".sdf",), poll_timeout_s=900,
        )

    def test_optimize_end_to_end(self, client):
        """Multi-cycle QVina2 optimization; needs the target .pdbqt for docking."""
        job_id = uuid.uuid4().hex[:20]
        target_uri = _upload_via_presign(
            client, job_id, "5d3h_pocket.pdb", _fixture("drughive-server", "5d3h_pocket.pdb")
        )
        ligand_uri = _upload_via_presign(
            client, job_id, "5d3h_ligand.sdf", _fixture("drughive-server", "5d3h_ligand.sdf")
        )
        target_pdbqt_uri = _upload_via_presign(
            client, job_id, "5d3h_pocket.pdbqt", _fixture("drughive-server", "5d3h_pocket.pdbqt")
        )
        _run_poll_download(
            client, svc="drughive-server", endpoint="optimize", job_id=job_id,
            body={
                "target_uri": target_uri, "ligand_uri": ligand_uri,
                "target_pdbqt_uri": target_pdbqt_uri,
                "pdb_id": "5d3h", "key_opt": "affinity_qvina",
                "n_cycles": 2, "n_samples_initial": 20, "n_samples": 4,
                "n_best_parents": 2, "zbetas": 0.3, "temps": 1.0,
                "save_name": "gw_opt",
            },
            output_suffixes=(".sdf",), poll_timeout_s=1800,
        )


@pytest.mark.fc
@_needs
class TestEndToEndPocketXMol:
    def test_dock_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        protein_uri = _upload_via_presign(
            client, job_id, "8C7Y_TXV_protein.pdb",
            _fixture("pocketxmol-server", "8C7Y_TXV_protein.pdb"),
        )
        ligand_uri = _upload_via_presign(
            client, job_id, "8C7Y_TXV_ligand_start_conf.sdf",
            _fixture("pocketxmol-server", "8C7Y_TXV_ligand_start_conf.sdf"),
        )
        _run_poll_download(
            client, svc="pocketxmol-server", endpoint="dock", job_id=job_id,
            body={
                "protein_uri": protein_uri, "ligand_uri": ligand_uri,
                "num_samples": 3, "batch_size": 3,
                "pocket_coord": [-8.257, 85.181, 19.050], "pocket_radius": 15,
            },
            output_suffixes=(".sdf",), poll_timeout_s=1800,
        )

    def test_sbdd_end_to_end(self, client):
        """Structure-based de novo design into a pocket (protein only)."""
        job_id = uuid.uuid4().hex[:20]
        protein_uri = _upload_via_presign(
            client, job_id, "2ar9_A.pdb", _fixture("pocketxmol-server", "2ar9_A.pdb")
        )
        _run_poll_download(
            client, svc="pocketxmol-server", endpoint="sbdd", job_id=job_id,
            body={
                "protein_uri": protein_uri, "num_samples": 5, "batch_size": 5,
                "pocket_coord": [-8.1603, 36.6972, 38.7714], "pocket_radius": 15,
                "mode": "simple",
            },
            output_suffixes=(".sdf",), poll_timeout_s=1800,
        )

    def test_linking_end_to_end(self, client):
        """Fragment growing/linking from an input fragment SDF."""
        job_id = uuid.uuid4().hex[:20]
        protein_uri = _upload_via_presign(
            client, job_id, "2ar9_A.pdb", _fixture("pocketxmol-server", "2ar9_A.pdb")
        )
        input_ligand_uri = _upload_via_presign(
            client, job_id, "fragment.sdf", _fixture("pocketxmol-server", "fragment.sdf")
        )
        _run_poll_download(
            client, svc="pocketxmol-server", endpoint="linking", job_id=job_id,
            body={
                "protein_uri": protein_uri, "input_ligand_uri": input_ligand_uri,
                "num_samples": 3, "batch_size": 3,
                "fragments": [[0, 1, 2, 3, 4, 5, 6]], "mol_size_mean": 28,
            },
            output_suffixes=(".sdf",), poll_timeout_s=1800,
        )

    def test_pepdesign_end_to_end(self, client):
        """De novo linear peptide design targeting a pocket."""
        job_id = uuid.uuid4().hex[:20]
        protein_uri = _upload_via_presign(
            client, job_id, "3bik_A.pdb", _fixture("pocketxmol-server", "3bik_A.pdb")
        )
        ref_ligand_uri = _upload_via_presign(
            client, job_id, "3bik_A_pocket_coord.sdf",
            _fixture("pocketxmol-server", "3bik_A_pocket_coord.sdf"),
        )
        _run_poll_download(
            client, svc="pocketxmol-server", endpoint="pepdesign", job_id=job_id,
            body={
                "protein_uri": protein_uri, "ref_ligand_uri": ref_ligand_uri,
                "mode": "denovo_linear", "pep_length": 5,
                "num_samples": 3, "batch_size": 3, "pocket_radius": 20,
            },
            output_suffixes=(".pdb", ".sdf"), poll_timeout_s=1800,
        )


@pytest.mark.fc
@_needs
class TestEndToEndMegalodon:
    def test_generate_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="megalodon-server", endpoint="generate", job_id=job_id,
            body={
                "model_name": "drugs_diffusion", "n_molecules": 10,
                "timesteps": 100, "seed": 42,
            },
            output_suffixes=(".sdf",), poll_timeout_s=1800,
        )


@pytest.mark.fc
@_needs
class TestEndToEndChemBounce:
    def test_scaffold_hop_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="chembounce-server", endpoint="scaffold_hop", job_id=job_id,
            body={
                # Inline SMILES (losartan) — chembounce takes a string, not a file.
                "input_smiles": "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl",
                "frag_max_n": 10, "tanimoto_threshold": 0.5, "database": "250mw",
            },
            output_suffixes=(".txt",), poll_timeout_s=1800,
        )


@pytest.mark.fc
@_needs
class TestEndToEndBoltz:
    def test_predict_structure_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="boltz-server", endpoint="predict_structure", job_id=job_id,
            body={
                "name": "fc_smoke", "msa_mode": "empty", "diffusion_samples": 1,
                "recycling_steps": 1, "sampling_steps": 50,
                "sequences": [{
                    "type": "protein", "id": "A",
                    "sequence": "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSC",
                    "msa_uri": "empty",
                }],
            },
            output_suffixes=(".cif",), poll_timeout_s=1800,
        )

    def test_predict_affinity_end_to_end(self, client):
        """Protein + SMILES ligand complex; binder_id points at the ligand chain.
        Asserts the affinity JSON (the distinctive predict_affinity output)."""
        job_id = uuid.uuid4().hex[:20]
        content = _run_poll_zip(
            client, svc="boltz-server", endpoint="predict_affinity", job_id=job_id,
            body={
                "name": "fc_affinity", "binder_id": "B", "msa_mode": "empty",
                "diffusion_samples": 1, "recycling_steps": 1, "sampling_steps": 50,
                "diffusion_samples_affinity": 1, "sampling_steps_affinity": 50,
                "sequences": [
                    {
                        "type": "protein", "id": "A",
                        "sequence": "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSC",
                        "msa_uri": "empty",
                    },
                    {"type": "ligand", "id": "B", "smiles": "c1ccccc1"},
                ],
            },
            poll_timeout_s=1800,
        )
        names = zipfile.ZipFile(io.BytesIO(content)).namelist()
        assert any("affinity" in n and n.endswith(".json") for n in names), (
            f"no affinity json in results.zip: {names}"
        )


@pytest.mark.fc
@_needs
class TestEndToEndBoltzGen:
    """Full-endpoint coverage of boltzgen's two task endpoints via the gateway.

    Both endpoints take a design-spec YAML plus (optionally) reference structure
    files.  The gateway dispatches form fields only — it can't multipart-upload
    the `ref_files` list — so:

      * `design`        — the fc_design / fc_peptide specs are fully self-
        contained (inline `protein:` entities), so only `design_yaml_uri`
        (presigned) is needed.
      * `inverse_fold`  — the 1brs spec references a backbone CIF by path; that
        CIF is bundled into a zip, uploaded via presign, and passed as
        `ref_files_zip_uri` (oss://...).  The gateway rewrites it to /mnt/oss/...
        and boltzgen extracts it next to the spec (needs boltzgen-server
        >= v0.0.13).  This mirrors genie3's `dataset_uri` zip pattern.

    Design runs at num_designs=2/budget=2 (~20-40 min on T4); inverse_fold skips
    the diffusion step but still folds + analyzes the backbone.
    """

    def test_design_end_to_end(self, client):
        """protein-anything design; assert results.zip carries a structure AND
        the analysis metrics CSV (not just any single output)."""
        job_id = uuid.uuid4().hex[:20]
        design_yaml_uri = _upload_via_presign(
            client, job_id, "design_spec.yaml", _fixture("boltzgen-server", "fc_design.yaml")
        )
        content = _run_poll_zip(
            client, svc="boltzgen-server", endpoint="design", job_id=job_id,
            body={
                "design_yaml_uri": design_yaml_uri,
                "protocol": "protein-anything", "num_designs": 2, "budget": 2,
            },
            poll_timeout_s=5400,
        )
        names = zipfile.ZipFile(io.BytesIO(content)).namelist()
        assert any(n.endswith((".pdb", ".cif")) for n in names), f"no structure in {names}"
        assert any(n.endswith(".csv") for n in names), f"no metrics CSV in {names}"

    def test_design_peptide_end_to_end(self, client):
        """peptide-anything protocol — a distinct filter/analysis path (auto-Cys
        filtering, stricter RMSD) over a short designed peptide."""
        job_id = uuid.uuid4().hex[:20]
        design_yaml_uri = _upload_via_presign(
            client, job_id, "design_spec.yaml", _fixture("boltzgen-server", "fc_peptide.yaml")
        )
        _run_poll_download(
            client, svc="boltzgen-server", endpoint="design", job_id=job_id,
            body={
                "design_yaml_uri": design_yaml_uri,
                "protocol": "peptide-anything", "num_designs": 2, "budget": 2,
            },
            output_suffixes=(".pdb", ".cif"), poll_timeout_s=5400,
        )

    def test_inverse_fold_end_to_end(self, client):
        """inverse-fold-only over a real backbone (barnase+barstar, 1BRS).

        The spec references 1brs.cif by path; the CIF is bundled into a zip and
        passed via `ref_files_zip_uri` since the gateway can't upload ref_files.
        """
        job_id = uuid.uuid4().hex[:20]
        design_yaml_uri = _upload_via_presign(
            client, job_id, "design_spec.yaml", _fixture("boltzgen-server", "1brs.yaml")
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("1brs.cif", _fixture("boltzgen-server", "1brs.cif"))
        ref_zip_uri = _upload_via_presign(client, job_id, "ref_files.zip", buf.getvalue())
        _run_poll_download(
            client, svc="boltzgen-server", endpoint="inverse_fold", job_id=job_id,
            body={
                "design_yaml_uri": design_yaml_uri,
                "ref_files_zip_uri": ref_zip_uri,
                "protocol": "protein-anything",
                "budget": 2, "inverse_fold_num_sequences": 2,
            },
            output_suffixes=(".pdb", ".cif"), poll_timeout_s=3600,
        )


@pytest.mark.fc
@_needs
class TestEndToEndProMera:
    def test_cofold_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        input_schema_uri = _upload_via_presign(
            client, job_id, "cofold.json", _fixture("promera-server", "test_target.json")
        )
        _run_poll_download(
            client, svc="promera-server", endpoint="cofold", job_id=job_id,
            body={
                "input_schema_uri": input_schema_uri,
                "num_seeds": 1, "diffusion_samples": 1, "diffusion_steps": 50,
            },
            output_suffixes=(".cif",), poll_timeout_s=2700,
        )

    def test_design_end_to_end(self, client):
        """Minibinder backbone design -> backbone.cif."""
        job_id = uuid.uuid4().hex[:20]
        target_schema_uri = _upload_via_presign(
            client, job_id, "target.json", _fixture("promera-server", "test_target.json")
        )
        _run_poll_download(
            client, svc="promera-server", endpoint="design", job_id=job_id,
            body={
                "target_schema_uri": target_schema_uri,
                "design_type": "minibinder", "num_backbones": 1,
                "diffusion_steps": 50, "inverse_folder_type": "none",
            },
            output_suffixes=(".cif",), poll_timeout_s=2700,
        )


def _build_zip(files: dict[str, str]) -> bytes:
    """Zip mapping archive paths -> on-disk fixture paths (in genie3-server/data)."""
    buf = io.BytesIO()
    base = Path(__file__).resolve().parents[2] / "genie3-server" / "tests" / "data"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, rel in files.items():
            zf.write(base / rel, arcname=arcname)
    return buf.getvalue()


def _genie3_motif_zip() -> bytes:
    return _build_zip({
        "problems/01_1LDB.json": "motifbench/problems/01_1LDB.json",
        "motifs/01_1LDB.pdb": "motifbench/motifs/01_1LDB.pdb",
    })


def _genie3_binder_zip() -> bytes:
    return _build_zip({
        "problems/01_bhrf1.json": "binder/problems/01_bhrf1.json",
        "targets/pdb/01_bhrf1.pdb": "binder/targets/pdb/01_bhrf1.pdb",
        "targets/pdb/01_bhrf1-chain_B.pdb": "binder/targets/pdb/01_bhrf1-chain_B.pdb",
        "targets/fasta/01_bhrf1.fasta": "binder/targets/fasta/01_bhrf1.fasta",
        "targets/fasta/01_bhrf1-chain_B.fasta": "binder/targets/fasta/01_bhrf1-chain_B.fasta",
        "targets/msa/01_bhrf1.a3m": "binder/targets/msa/01_bhrf1.a3m",
        "targets/msa/01_bhrf1-chain_B.a3m": "binder/targets/msa/01_bhrf1-chain_B.a3m",
    })


@pytest.mark.fc
@_needs
class TestEndToEndGenie3:
    """Full-function coverage of genie3's four generation endpoints via the gateway.

    unconditional + custom carry all inputs inline in the run body, so they work
    over the gateway's form-only dispatch directly. motif + binder need a dataset
    zip: it is uploaded via presign to OSS, then referenced by `dataset_uri`
    (oss://...); the gateway rewrites it to /mnt/oss/... and genie3 reads it from
    the mounted bucket (needs genie3-server >= v0.0.20).
    """

    def test_generate_unconditional_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="genie3-server", endpoint="generate/unconditional", job_id=job_id,
            body={"n_sample": 1, "min_length": 50, "max_length": 50, "length_step": 50},
            output_suffixes=(".pdb",), poll_timeout_s=1800,
        )

    def test_generate_motif_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        dataset_uri = _upload_via_presign(
            client, job_id, "motif.zip", _genie3_motif_zip()
        )
        _run_poll_download(
            client, svc="genie3-server", endpoint="generate/motif", job_id=job_id,
            body={"dataset_uri": dataset_uri, "n_sample": 1, "batch_size": 1,
                  "selections": "01_1LDB"},
            output_suffixes=(".pdb",), poll_timeout_s=1800,
        )

    def test_generate_binder_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        dataset_uri = _upload_via_presign(
            client, job_id, "binder.zip", _genie3_binder_zip()
        )
        _run_poll_download(
            client, svc="genie3-server", endpoint="generate/binder", job_id=job_id,
            body={"dataset_uri": dataset_uri, "n_sample": 1, "batch_size": 1,
                  "selections": "01_bhrf1"},
            output_suffixes=(".pdb",), poll_timeout_s=1800,
        )

    def test_generate_custom_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        config_yaml = yaml.safe_dump({
            "experiment": {"name": "gw_custom"},
            "paths": {"rootdir": "PLACEHOLDER_OVERRIDDEN_BY_SERVER"},
            "generation": {
                "dataset": {
                    "source": "unconditional",
                    "min_length": 50, "max_length": 50, "length_step": 50, "n_sample": 1,
                },
                "sampler": {"sampler": {"direction_scale": 0.8}},
            },
            "evaluation": {"version": "unconditional", "folding": {"model_name": "esmfold"}},
        })
        _run_poll_download(
            client, svc="genie3-server", endpoint="generate", job_id=job_id,
            body={"config_yaml": config_yaml},
            output_suffixes=(".pdb",), poll_timeout_s=1800,
        )


_ESMFOLD2_SEQ = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)


@pytest.mark.fc
@_needs
class TestEndToEndESMFold2:
    def test_fold_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="esmfold2-server", endpoint="fold", job_id=job_id,
            body={
                "sequences": [{
                    "type": "protein", "id": "A", "sequence": _ESMFOLD2_SEQ,
                }],
                "num_sampling_steps": 10, "num_loops": 1,
            },
            output_suffixes=(".cif",), poll_timeout_s=1800,
        )

    def test_fold_with_msa_zip_uri_end_to_end(self, client):
        """MSA-conditioned fold via a presigned a3m zip (needs esmfold2 >= v0.0.7).

        The gateway can't multipart-upload msa_files, so the per-chain A3M zip is
        uploaded to OSS via presign and referenced by `msa_zip_uri`; the gateway
        rewrites oss:// → /mnt/oss and esmfold2 extracts it. The a3m is keyed by
        filename stem (= chain id "A").
        """
        job_id = uuid.uuid4().hex[:20]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("A.a3m", f">A\n{_ESMFOLD2_SEQ}\n")
        msa_zip_uri = _upload_via_presign(client, job_id, "msa.zip", buf.getvalue())
        _run_poll_download(
            client, svc="esmfold2-server", endpoint="fold", job_id=job_id,
            body={
                "sequences": [{
                    "type": "protein", "id": "A", "sequence": _ESMFOLD2_SEQ,
                }],
                "msa_zip_uri": msa_zip_uri,
                "num_sampling_steps": 10, "num_loops": 1,
            },
            output_suffixes=(".cif",), poll_timeout_s=1800,
        )


@pytest.mark.fc
@_needs
class TestEndToEndImmuneBuilder:
    def test_predict_nanobody_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="immunebuilder-server", endpoint="predict_nanobody", job_id=job_id,
            body={
                "heavy_sequence": (
                    "QVQLVESGGGLVQAGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAINSGGGSTYYPDSVKG"
                    "RFTISRDNAKNTVYLQMNSLKPEDTAVYYCAKDGYGGSFDYWGQGTQVTVSS"
                ),
                "name": "async_nb", "save_all_models": True,
            },
            output_suffixes=(".pdb",), poll_timeout_s=600,
        )

    def test_predict_antibody_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="immunebuilder-server", endpoint="predict_antibody", job_id=job_id,
            body={
                "heavy_sequence": (
                    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYT"
                    "RYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
                ),
                "light_sequence": (
                    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPS"
                    "RFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
                ),
                "name": "gw_ab", "save_all_models": True,
            },
            output_suffixes=(".pdb",), poll_timeout_s=600,
        )

    def test_predict_tcr_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="immunebuilder-server", endpoint="predict_tcr", job_id=job_id,
            body={
                "alpha_sequence": (
                    "METLLGVSLVILWLQLARVNSQQGEEDPQALSIQEGENATMNCSYKTSINNLQWYRQNSGR"
                    "GLVHLILIRSNEREKHSGRLRVTLDTSKKSSSLLITASRAADTASYFCAVNFGGGKLIFGQGTELSVKP"
                ),
                "beta_sequence": (
                    "NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVP"
                    "NGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSFSTCSANYGYTFGSGTRLTVVEDL"
                ),
                "name": "gw_tcr", "save_all_models": True,
            },
            output_suffixes=(".pdb",), poll_timeout_s=600,
        )


@pytest.mark.fc
@_needs
class TestEndToEndSemlaFlow:
    def test_generate_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="semlaflow-server", endpoint="generate", job_id=job_id,
            body={
                "model_name": "qm9", "n_molecules": 10,
                "integration_steps": 50, "seed": 42,
            },
            output_suffixes=(".sdf",), poll_timeout_s=900,
        )


@pytest.mark.fc
@_needs
class TestEndToEndFlowMol:
    def test_generate_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="flowmol-server", endpoint="generate", job_id=job_id,
            body={
                "n_mols": 10, "n_timesteps": 100,
                "model_variant": "flowmol3", "seed": 42,
            },
            output_suffixes=(".sdf",), poll_timeout_s=900,
        )


# A minimal [scoring] section reused by the reinvent modes that need one
# (scoring / enumeration / staged-learning). QED needs no external data files,
# so it runs on any FC instance. Passed as a dict in the JSON body — the gateway
# JSON-encodes it into a form field, which model_form_depends decodes server-side.
_REINVENT_QED_SCORING = {
    "type": "geometric_mean",
    "component": [{"QED": {"endpoint": [{"name": "QED", "weight": 1.0}]}}],
}


@pytest.mark.fc
@_needs
class TestEndToEndReinvent:
    """Full-pipeline gateway coverage — one test per REINVENT run mode.

    Each drives the full gateway chain (JSON-body submit -> 202 + job_id -> FC
    async /api/tasks/<mode> -> GPU compute -> results.zip mirrored to OSS -> 302
    download). File inputs are pushed to OSS via presign and passed as the
    matching ``*_uri`` field, since gateway dispatch is form-only (no multipart).
    """

    def test_sampling_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="reinvent-server", endpoint="sampling", job_id=job_id,
            body={"generator": "reinvent", "num_smiles": 20},
            output_suffixes=(".csv",), poll_timeout_s=900,
        )

    def test_scoring_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        smiles_uri = _upload_via_presign(
            client, job_id, "compounds.smi",
            b"CCO\nc1ccccc1\nCC(=O)O\nCC(=O)Oc1ccccc1C(=O)O\n",
        )
        _run_poll_download(
            client, svc="reinvent-server", endpoint="scoring", job_id=job_id,
            body={"smiles_file_uri": smiles_uri, "scoring": _REINVENT_QED_SCORING},
            output_suffixes=(".csv",), poll_timeout_s=900,
        )

    def test_enumeration_end_to_end(self, client):
        """PepInvent AA enumeration — masked peptide + amino-acid library CSV.

        Fixtures mirror upstream ``tests/.../enumeration/mock_aa_library.csv``
        (columns SMILES,NAME → so name/smiles columns are overridden to
        uppercase to match).
        """
        job_id = uuid.uuid4().hex[:20]
        peptide_uri = _upload_via_presign(
            client, job_id, "peptides.smi",
            b"N[C@@H](CS)C(=O)|?|N[C@@H](C)C(=O)|?|N[C@@H](C)C(=O)O\n",
        )
        library_uri = _upload_via_presign(
            client, job_id, "amino_acids.csv",
            b"SMILES,NAME\n"
            b"N[C@@H](Cn1c(S)nnc1-c1ccc(F)cc1)C(=O)O,ZN9\n"
            b"NC(=O)CC[C@H](N)C(=O)O,Q\n"
            b"N[C@@H](CS)C(=O)O,C\n"
            b"NCC(=O)O,G\n",
        )
        _run_poll_download(
            client, svc="reinvent-server", endpoint="enumeration", job_id=job_id,
            body={
                "peptide_smiles_uri": peptide_uri,
                "amino_acid_library_uri": library_uri,
                "amino_acid_name_column": "NAME",
                "smiles_column": "SMILES",
                "batch_size": 5,
                "scoring": _REINVENT_QED_SCORING,
            },
            output_suffixes=(".csv",), poll_timeout_s=900,
        )

    def test_transfer_learning_end_to_end(self, client):
        """Fine-tune the reinvent prior for 1 epoch on a tiny SMILES set."""
        job_id = uuid.uuid4().hex[:20]
        smiles_uri = _upload_via_presign(
            client, job_id, "tl_reinvent.smi",
            b"Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1\n"
            b"O=C(Nc1ccc(Oc2nccc(-c3cccnc3)n2)cc1)c1ccc(Cl)cc1\n"
            b"CC(=O)Oc1ccccc1C(=O)O\n"
            b"CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21\n"
            b"COc1ccc2[nH]cc(CCN)c2c1\n"
            b"CC(C)Cc1ccc(C(C)C(=O)O)cc1\n"
            b"CN1CCC[C@H]1c1cccnc1\n"
            b"O=C(O)c1ccccc1O\n"
            b"CCN(CC)CCNC(=O)c1ccc(N)cc1\n"
            b"Clc1ccccc1C1=NCC(=O)Nc2ccccc21\n"
            b"CC(=O)Nc1ccc(O)cc1\n"
            b"Cn1cnc2c1c(=O)n(C)c(=O)n2C\n",
        )
        _run_poll_download(
            client, svc="reinvent-server", endpoint="transfer-learning", job_id=job_id,
            body={
                "generator": "reinvent",
                "smiles_file_uri": smiles_uri,
                "output_model_name": "TL_model.model",
                "num_epochs": 1,
                "save_every_n_epochs": 1,
                "batch_size": 10,
                "num_refs": 0,
                # sample_batch_size omitted → default 100 (upstream enforces ge=100).
            },
            output_suffixes=(".model",), poll_timeout_s=1800,
        )

    def test_staged_learning_end_to_end(self, client):
        """One short RL stage off the NAS-staged reinvent prior (no file input)."""
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="reinvent-server", endpoint="staged-learning", job_id=job_id,
            body={
                "generator": "reinvent",
                "batch_size": 8,
                "stages": [
                    {
                        "chkpt_name": "stage1.chkpt",
                        "max_score": 0.4,
                        "min_steps": 2,
                        "max_steps": 4,
                        "scoring": _REINVENT_QED_SCORING,
                    }
                ],
            },
            output_suffixes=(".csv",), poll_timeout_s=1800,
        )


@pytest.mark.fc
@_needs
class TestEndToEndOpenADMET:
    def test_predict_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="openadmet-server", endpoint="predict", job_id=job_id,
            body={
                "input_smiles": (
                    "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl,"
                    "CC(=O)OC1=CC=CC=C1C(=O)O,"
                    "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
                ),
                "model_names": ["herg-chemeleon-baseline"], "accelerator": "gpu",
            },
            output_suffixes=(".csv",), poll_timeout_s=600,
        )


@pytest.mark.fc
@_needs
class TestEndToEndHaddock3:
    """CNS-free restraints path through the gateway (docking/scoring need a
    licensed CNS staged on the FC NAS, so they are not exercised here).

    NOTE: `restraints/restrain-bodies` is a nested endpoint — requires
    gateway >= v0.0.2 ({endpoint:path} routing)."""

    def test_restrain_bodies_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        structure_uri = _upload_via_presign(
            client, job_id, "complex.pdb", _fixture("haddock3-server", "complex.pdb")
        )
        _run_poll_download(
            client, svc="haddock3-server", endpoint="restraints/restrain-bodies",
            job_id=job_id,
            body={"structure_uri": structure_uri},
            output_suffixes=(".tbl",), poll_timeout_s=600,
        )


@pytest.mark.fc
@_needs
class TestEndToEndLightDock:
    """Full LightDock docking path through the gateway.

    Two file inputs (receptor + ligand) uploaded via presign -> oss:// URIs;
    the gateway rewrites them to /mnt/oss/... for the oss_mount downstream. Tiny
    sampling (swarms=2, glowworms=5, steps=3) so the CPU dock finishes fast.
    Asserts results.zip carries a ranked pose (output/top/top_1.pdb)."""

    def test_dock_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        receptor_uri = _upload_via_presign(
            client, job_id, "receptor.pdb", _fixture("lightdock-server", "receptor.pdb")
        )
        ligand_uri = _upload_via_presign(
            client, job_id, "ligand.pdb", _fixture("lightdock-server", "ligand.pdb")
        )
        _run_poll_download(
            client, svc="lightdock-server", endpoint="dock", job_id=job_id,
            body={
                "receptor_uri": receptor_uri,
                "ligand_uri": ligand_uri,
                "swarms": 2, "glowworms": 5, "steps": 3, "top": 3,
            },
            output_suffixes=("top_1.pdb",), poll_timeout_s=900,
        )


@pytest.mark.fc
@_needs
class TestEndToEndLASErMPNN:
    """LASErMPNN ligand-conditioned design through the gateway.

    Single PDB (protein + protonated ligand) uploaded via presign -> oss:// URI;
    the gateway rewrites it to /mnt/oss/... for the oss_mount downstream. Tiny
    sampling (designs_per_input=1, designs_per_batch=1) so the GPU job finishes
    fast. Asserts results.zip carries a design_*.pdb."""

    def test_design_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        pdb_uri = _upload_via_presign(
            client, job_id, "4jnj-1_prot.pdb", _fixture("lasermpnn-server", "4jnj-1_prot.pdb")
        )
        _run_poll_download(
            client, svc="lasermpnn-server", endpoint="design", job_id=job_id,
            body={
                "pdb_uri": pdb_uri,
                "designs_per_input": 1, "designs_per_batch": 1,
                "sequence_temp": 0.3,
            },
            output_suffixes=("design_0.pdb", ".pdb"), poll_timeout_s=1200,
        )
