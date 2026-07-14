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


def _fixture(svc: str, name: str) -> bytes:
    """Read a downstream service's test fixture (services/<svc>/tests/data/<name>)."""
    return (Path(__file__).resolve().parents[2] / svc / "tests" / "data" / name).read_bytes()


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
    """Shared run -> poll -> 302-OSS-download assertion for gateway e2e tests.

    Asserts: 202 submit, terminal status=completed, the download is an OSS 302
    (output-sink mirror, NOT the downstream proxy fallback), and results.zip
    carries a file whose name ends with one of `output_suffixes`.
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
        status = client.get(f"/v1/jobs/{job_id}").json()
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
    names = zipfile.ZipFile(io.BytesIO(dl.content)).namelist()
    assert any(n.endswith(output_suffixes) for n in names), (
        f"no {output_suffixes} in results.zip: {names}"
    )


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


# ===================================================================
# End-to-end: real job through the gateway (alphafold — slow)
# ===================================================================

# 76-residue ubiquitin — same sequence alphafold-server's own tests use; small
# enough that reduced_dbs MSA + monomer inference fits FC's instance lifetime.
_AF_FASTA = (
    b">test_ubiquitin\n"
    b"MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG\n"
)

# AlphaFold (MSA search + inference) is far slower than proteinmpnn; give it up
# to an hour before declaring the job stuck.
AF_RUN_POLL_TIMEOUT_S = 3600
AF_RUN_POLL_INTERVAL_S = 30


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
        names = zipfile.ZipFile(io.BytesIO(dl.content)).namelist()
        assert any(n.endswith(".pdb") for n in names), f"no predicted PDB in {names}"


# ===================================================================
# End-to-end: real job through the gateway (mmseqs2 — inline FASTA, slow MSA)
# ===================================================================

# Unlike proteinmpnn/alphafold, mmseqs2 uses the ColabFold protocol: the FASTA
# is passed inline in the `q` form field (no file upload / presign). Same 52-aa
# monomer + mode="all" (UniRef30-only) that mmseqs2-server's own async task test
# uses, so the run does NOT require the colabfold_envdb to be staged on NAS.
_MMSEQS_MONOMER = "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVN"
_MMSEQS_Q = f">probe1\n{_MMSEQS_MONOMER}\n"

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
        names = zipfile.ZipFile(io.BytesIO(dl.content)).namelist()
        assert any(n.endswith(".a3m") for n in names), f"no MSA a3m in {names}"


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
    def test_generate_end_to_end(self, client):
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


@pytest.mark.fc
@_needs
class TestEndToEndRFdiffusion2:
    def test_generate_end_to_end(self, client):
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


@pytest.mark.fc
@_needs
class TestEndToEndRFantibody:
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


@pytest.mark.fc
@_needs
class TestEndToEndPPIflow:
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


@pytest.mark.fc
@_needs
class TestEndToEndDockQ:
    def test_score_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        model_uri = _upload_via_presign(
            client, job_id, "model.pdb", _fixture("dockq-server", "model.pdb")
        )
        native_uri = _upload_via_presign(
            client, job_id, "native.pdb", _fixture("dockq-server", "native.pdb")
        )
        _run_poll_download(
            client, svc="dockq-server", endpoint="score", job_id=job_id,
            body={"model_uri": model_uri, "native_uri": native_uri, "name": "fc_smoke"},
            output_suffixes=(".json",), poll_timeout_s=600,
        )


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
    def test_generate_end_to_end(self, client):
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
                "num_samples": 3, "model_variant": "gvp_conditional",
            },
            output_suffixes=(".sdf",), poll_timeout_s=1200,
        )


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


@pytest.mark.fc
@_needs
class TestEndToEndBoltzGen:
    def test_design_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        design_yaml_uri = _upload_via_presign(
            client, job_id, "design_spec.yaml", _fixture("boltzgen-server", "fc_design.yaml")
        )
        _run_poll_download(
            client, svc="boltzgen-server", endpoint="design", job_id=job_id,
            body={
                "design_yaml_uri": design_yaml_uri,
                "protocol": "protein-anything", "num_designs": 2, "budget": 2,
            },
            output_suffixes=(".pdb", ".cif"), poll_timeout_s=5400,
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


@pytest.mark.fc
@_needs
class TestEndToEndGenie3:
    def test_generate_unconditional_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="genie3-server", endpoint="generate/unconditional", job_id=job_id,
            body={"n_sample": 1, "min_length": 50, "max_length": 50, "length_step": 50},
            output_suffixes=(".pdb",), poll_timeout_s=1800,
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
                    "type": "protein", "id": "A",
                    "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
                }],
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


@pytest.mark.fc
@_needs
class TestEndToEndReinvent:
    def test_sampling_end_to_end(self, client):
        job_id = uuid.uuid4().hex[:20]
        _run_poll_download(
            client, svc="reinvent-server", endpoint="sampling", job_id=job_id,
            body={"generator": "reinvent", "num_smiles": 20},
            output_suffixes=(".csv",), poll_timeout_s=900,
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
