"""FC async task mode tests for drughive-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/drughive-server/tests/test_fc_task.py -v

Validates ``/api/tasks/generate`` + ``/api/tasks/generate_spatial`` +
``/api/tasks/optimize`` end-to-end under FC async task mode
(``X-Fc-Invocation-Type: Async``).

Async task mode pins the FC instance for the whole job (no 30s HTTP-gateway
recycle risk) and dedups by ``X-Fc-Async-Task-Id`` at the platform layer.

Payload sizing — all three tests use the vendored 5d3h example, which fits
comfortably under FC's 128 KiB async payload cap:

    pocket PDB (5d3h_pocket.pdb):    ~29 KB
    ligand SDF  (5d3h_ligand.sdf):    ~4 KB
    pocket PDBQT (5d3h_pocket.pdbqt): ~28 KB
    ─────────────────────────────────────
    worst case (optimize)          :  ~62 KB  ≪ 128 KiB

No sync-bootstrap trick needed.
"""

from __future__ import annotations

import io
import time
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

SERVICE = "drughive-server"

DATA_DIR = Path(__file__).resolve().parent / "data"
POCKET_PDB = DATA_DIR / "5d3h_pocket.pdb"
LIGAND_SDF = DATA_DIR / "5d3h_ligand.sdf"
POCKET_PDBQT = DATA_DIR / "5d3h_pocket.pdbqt"

# SMARTS pattern for scaffold hopping: amide (C=O)N.  Verified to have
# exactly 1 match in 5d3h_ligand.sdf via rdkit; benzene "c1ccccc1" would
# NOT match (5d3h ligand's aromatic ring fails RDKit kekulization due to
# an OH substituent conflict).  If GetSubstructMatches returns empty,
# upstream strips all atoms → downstream PCA crashes with
# "Found array with 0 sample(s)".
SMARTS_PATTERN = "C(=O)N"

# generate/spatial: DrugHIVE sampling of 10 candidates + FF opt runs
# ~30-90s on Tesla T4.  optimize is a full QVina2 loop: even trimmed to
# n_cycles=2, n_samples_initial=20, n_samples=4, n_best_parents=2 it
# needs the full docking pass — allow ~20-30 min.
POLL_TIMEOUT_S = 1800
POLL_INTERVAL_S = 20

TIMEOUT = httpx.Timeout(connect=30, read=300, write=600, pool=30)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url(SERVICE, start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str):
    with httpx.Client(base_url=base_url, timeout=TIMEOUT) as c:
        yield c


@pytest.fixture(scope="module")
def generate_task_id() -> str:
    return f"fc-async-gen-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def spatial_task_id() -> str:
    return f"fc-async-spatial-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def optimize_task_id() -> str:
    return f"fc-async-opt-{int(time.time())}-{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _async_headers(task_id: str) -> dict[str, str]:
    return {
        "X-Fc-Invocation-Type": "Async",
        "X-Bioagent-Job-Id": task_id,
        "X-Fc-Async-Task-Id": task_id,
    }


def _get_with_retry(
    client: httpx.Client,
    path: str,
    *,
    max_attempts: int = 15,
    backoff_s: int = 20,
) -> httpx.Response:
    """GET that retries on FC HTTP-gateway 429 throttling.

    After a long-running async task, the FC HTTP gateway can rate-limit
    subsequent GETs to ``/api/jobs/...``.  Platform-layer artifact — see
    ``project_fc_http_polling_unreliable_at_concurrency.md``.
    """
    last: httpx.Response | None = None
    for _attempt in range(max_attempts):
        last = client.get(path)
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    assert last is not None
    return last


def _poll_to_completion(client: httpx.Client, task_id: str) -> dict:
    # Default max_transient_errors (10 × 15s = 2.5 min) is too tight
    # for FC's 4-7 min 429 window under contention.  60 × 20s = 20 min.
    final = poll_job(
        client, "", task_id,
        timeout_s=POLL_TIMEOUT_S,
        interval_s=POLL_INTERVAL_S,
        max_transient_errors=60,
    )
    assert final["status"] == "completed", f"task did not complete: {final}"
    return final


def _sdf_looks_valid(content: bytes) -> bool:
    """SDF terminator + at least one atom-typish token."""
    text = content.decode("utf-8", errors="replace")
    return "$$$$" in text and any(k in text for k in (" C ", " N ", " O "))


# ---------------------------------------------------------------------------
# Per-endpoint submit + task fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generate_submit_response(
    client: httpx.Client, generate_task_id: str
) -> httpx.Response:
    with open(POCKET_PDB, "rb") as fp, open(LIGAND_SDF, "rb") as fl:
        return client.post(
            "/api/tasks/generate",
            files={
                "target": (POCKET_PDB.name, fp.read(), "chemical/x-pdb"),
                "ligand": (LIGAND_SDF.name, fl.read(), "chemical/x-mdl-sdfile"),
            },
            data={
                "n_samples": "5",
                "pdb_id": "5d3h",
                "zbetas": "0.0",
                "temps": "0.5",
            },
            headers=_async_headers(generate_task_id),
        )


@pytest.fixture(scope="module")
def generate_task(
    client: httpx.Client,
    generate_task_id: str,
    generate_submit_response: httpx.Response,
) -> dict:
    assert generate_submit_response.status_code == 202, (
        f"async generate submit returned "
        f"{generate_submit_response.status_code}: "
        f"{generate_submit_response.text!r}"
    )
    return _poll_to_completion(client, generate_task_id)


@pytest.fixture(scope="module")
def spatial_submit_response(
    client: httpx.Client, spatial_task_id: str
) -> httpx.Response:
    with open(POCKET_PDB, "rb") as fp, open(LIGAND_SDF, "rb") as fl:
        return client.post(
            "/api/tasks/generate_spatial",
            files={
                "target": (POCKET_PDB.name, fp.read(), "chemical/x-pdb"),
                "ligand": (LIGAND_SDF.name, fl.read(), "chemical/x-mdl-sdfile"),
            },
            data={
                "n_samples": "5",
                "pdb_id": "5d3h",
                "substruct_modify_pattern": SMARTS_PATTERN,
                "zbetas": "0.3",
                "temps": "1.0",
            },
            headers=_async_headers(spatial_task_id),
        )


@pytest.fixture(scope="module")
def spatial_task(
    client: httpx.Client,
    spatial_task_id: str,
    spatial_submit_response: httpx.Response,
) -> dict:
    assert spatial_submit_response.status_code == 202, (
        f"async spatial submit returned "
        f"{spatial_submit_response.status_code}: "
        f"{spatial_submit_response.text!r}"
    )
    return _poll_to_completion(client, spatial_task_id)


@pytest.fixture(scope="module")
def optimize_submit_response(
    client: httpx.Client, optimize_task_id: str
) -> httpx.Response:
    """Quick-run optimize params — full defaults are 4-8 h."""
    with (
        open(POCKET_PDB, "rb") as fp,
        open(LIGAND_SDF, "rb") as fl,
        open(POCKET_PDBQT, "rb") as fq,
    ):
        return client.post(
            "/api/tasks/optimize",
            files={
                "target": (POCKET_PDB.name, fp.read(), "chemical/x-pdb"),
                "ligand": (LIGAND_SDF.name, fl.read(), "chemical/x-mdl-sdfile"),
                "target_pdbqt": (POCKET_PDBQT.name, fq.read(), "chemical/x-pdbqt"),
            },
            data={
                "pdb_id": "5d3h",
                "key_opt": "affinity_qvina",
                "n_cycles": "2",
                "n_samples_initial": "20",
                "n_samples": "4",
                "n_best_parents": "2",
                "zbetas": "0.3",   # scalar → broadcast to [0.3, 0.3]
                "temps": "1.0",
                "save_name": "async_opt_smoke",
            },
            headers=_async_headers(optimize_task_id),
        )


@pytest.fixture(scope="module")
def optimize_task(
    client: httpx.Client,
    optimize_task_id: str,
    optimize_submit_response: httpx.Response,
) -> dict:
    assert optimize_submit_response.status_code == 202, (
        f"async optimize submit returned "
        f"{optimize_submit_response.status_code}: "
        f"{optimize_submit_response.text!r}"
    )
    return _poll_to_completion(client, optimize_task_id)


# ===========================================================================
# Section 1: Submit semantics + OpenAPI registration.
# ===========================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_generate_returns_202(self, generate_submit_response):
        assert generate_submit_response.status_code == 202, (
            f"expected 202; got {generate_submit_response.status_code} "
            f"body={generate_submit_response.text!r}"
        )

    def test_spatial_returns_202(self, spatial_submit_response):
        assert spatial_submit_response.status_code == 202, (
            f"expected 202; got {spatial_submit_response.status_code} "
            f"body={spatial_submit_response.text!r}"
        )

    def test_optimize_returns_202(self, optimize_submit_response):
        assert optimize_submit_response.status_code == 202, (
            f"expected 202; got {optimize_submit_response.status_code} "
            f"body={optimize_submit_response.text!r}"
        )

    def test_task_endpoints_registered_in_openapi(self, client):
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        expected = {
            "/api/tasks/generate",
            "/api/tasks/generate_spatial",
            "/api/tasks/optimize",
        }
        missing = expected - set(spec.get("paths", {}))
        assert not missing, (
            f"task endpoints missing from OpenAPI: {missing}; "
            f"settings.task_endpoints_enabled may be False"
        )


# ===========================================================================
# Section 2: Per-endpoint completion + outputs.
# ===========================================================================


def _assert_completed(task: dict, task_id: str) -> None:
    assert task["status"] == "completed"
    assert task["job_id"] == task_id
    assert task.get("started_at") is not None
    assert task.get("completed_at") is not None
    d = task.get("duration_seconds")
    assert d is not None and d > 0, f"duration missing: {d!r}"
    assert task.get("output_count", 0) > 0, (
        f"output_count zero — pipeline ran but produced no files: {task}"
    )


@pytest.mark.fc
class TestAsyncGenerate:
    def test_completed(self, generate_task, generate_task_id, client):
        _assert_completed(generate_task, generate_task_id)
        files = _get_with_retry(
            client, f"/api/jobs/{generate_task_id}/files"
        ).json()["files"]
        # Upstream writes into a nested layout:
        #   output/<gen_name>/<pdb_id>/mols_gen.sdf  (+ mols_gen_opt.sdf if ffopt)
        # gen_name defaults to "prior" when zbetas=[0,0,0,0].
        assert any(
            f.endswith("/mols_gen.sdf") or f.endswith("/mols_gen_opt.sdf")
            for f in files
        ), f"mols_gen[_opt].sdf missing from generate output: {files}"

    def test_input_params_echoed(self, generate_task):
        params = generate_task.get("input_params") or {}
        assert params.get("n_samples") == 5
        assert params.get("pdb_id") == "5d3h"

    def test_output_sdf_downloadable(self, client, generate_task_id, generate_task):
        files = _get_with_retry(
            client, f"/api/jobs/{generate_task_id}/files"
        ).json()["files"]
        # Prefer the FF-optimized SDF if present, else the raw mols_gen.sdf.
        sdf_name = next(
            (f for f in files if f.endswith("/mols_gen_opt.sdf")),
            next(f for f in files if f.endswith("/mols_gen.sdf")),
        )
        r = _get_with_retry(client, f"/api/jobs/{generate_task_id}/file/{sdf_name}")
        assert r.status_code == 200
        assert _sdf_looks_valid(r.content), (
            f"SDF content looks malformed (first 200 bytes): {r.content[:200]!r}"
        )


@pytest.mark.fc
class TestAsyncGenerateSpatial:
    def test_completed(self, spatial_task, spatial_task_id, client):
        _assert_completed(spatial_task, spatial_task_id)
        files = _get_with_retry(
            client, f"/api/jobs/{spatial_task_id}/files"
        ).json()["files"]
        # MolGeneratorSpatial writes into the same nested layout as MolGenerator:
        # output/<gen_name>/<pdb_id>/mols_gen.sdf.  Some versions also emit
        # mols_pred*.sdf variants for the modified region.
        assert any(
            f.endswith("/mols_gen.sdf")
            or f.endswith("/mols_gen_opt.sdf")
            or ("/mols_pred" in f and f.endswith(".sdf"))
            for f in files
        ), f"mols_gen/mols_pred SDF missing from spatial output: {files}"

    def test_input_params_echoed(self, spatial_task):
        params = spatial_task.get("input_params") or {}
        assert params.get("substruct_modify_pattern") == SMARTS_PATTERN
        assert params.get("n_samples") == 5

    def test_output_sdf_downloadable(self, client, spatial_task_id, spatial_task):
        files = _get_with_retry(
            client, f"/api/jobs/{spatial_task_id}/files"
        ).json()["files"]
        sdf_name = next(
            f for f in files
            if f.endswith("/mols_gen_opt.sdf")
            or f.endswith("/mols_gen.sdf")
            or ("/mols_pred" in f and f.endswith(".sdf"))
        )
        r = _get_with_retry(client, f"/api/jobs/{spatial_task_id}/file/{sdf_name}")
        assert r.status_code == 200
        assert _sdf_looks_valid(r.content), (
            f"SDF content looks malformed: {r.content[:200]!r}"
        )


@pytest.mark.fc
class TestAsyncOptimize:
    def test_completed(self, optimize_task, optimize_task_id, client):
        _assert_completed(optimize_task, optimize_task_id)
        files = _get_with_retry(
            client, f"/api/jobs/{optimize_task_id}/files"
        ).json()["files"]
        # optimize starts by generating the initial population under
        # output/pdbzinc_initial/<pdb_id>/mols_gen.sdf — any nested
        # mols_gen*.sdf proves the pipeline got past model load.
        assert any(
            "/mols_gen" in f and f.endswith(".sdf") for f in files
        ), f"initial-pool mols_gen*.sdf missing from optimize output: {files}"

    def test_input_params_echoed(self, optimize_task):
        params = optimize_task.get("input_params") or {}
        assert params.get("key_opt") == "affinity_qvina"
        assert params.get("n_cycles") == 2
        assert params.get("n_samples_initial") == 20

    def test_qvina_scores_written(self, client, optimize_task_id, optimize_task):
        """QVina2 should have written _qvina.csv files as it docked
        candidates.  Any *_qvina.csv proves the docking loop ran."""
        files = _get_with_retry(
            client, f"/api/jobs/{optimize_task_id}/files"
        ).json()["files"]
        assert any(f.endswith("_qvina.csv") for f in files), (
            f"no *_qvina.csv found — QVina2 docking may not have executed: {files}"
        )


# ===========================================================================
# Section 3: Identity — X-Bioagent-Job-Id propagates to JobInfo.job_id.
# ===========================================================================


@pytest.mark.fc
class TestAsyncTaskIdentity:
    def test_generate_job_id_matches_task_id(self, generate_task, generate_task_id):
        assert generate_task["job_id"] == generate_task_id, (
            "task endpoint must use X-Bioagent-Job-Id as JobInfo.job_id"
        )

    def test_spatial_job_id_matches_task_id(self, spatial_task, spatial_task_id):
        assert spatial_task["job_id"] == spatial_task_id

    def test_optimize_job_id_matches_task_id(self, optimize_task, optimize_task_id):
        assert optimize_task["job_id"] == optimize_task_id


# ===========================================================================
# Section 4: Job lifecycle on the (cheapest) generate task.
# ===========================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_job_visible_via_status_endpoint(
        self, client, generate_task_id, generate_task
    ):
        r = _get_with_retry(client, f"/api/jobs/{generate_task_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == generate_task_id
        assert body["status"] == "completed"

    def test_job_log_endpoint(self, client, generate_task_id, generate_task):
        r = _get_with_retry(client, f"/api/jobs/{generate_task_id}/log")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == generate_task_id
        log_text = body.get("log") or body.get("text", "")
        assert isinstance(log_text, str)

    def test_job_download_zip(self, client, generate_task_id, generate_task):
        r = _get_with_retry(client, f"/api/jobs/{generate_task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any("mols_gen" in n and n.endswith(".sdf") for n in names), (
            f"mols_gen SDF missing from zip (nested paths OK): {names}"
        )

    def test_single_file_download_missing_returns_404(
        self, client, generate_task_id, generate_task
    ):
        r = _get_with_retry(
            client, f"/api/jobs/{generate_task_id}/file/nonexistent.xyz"
        )
        assert r.status_code == 404


# ===========================================================================
# Section 5: Duplicate dedup — FC platform + framework layer.
# ===========================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    """Resubmitting same X-Fc-Async-Task-Id must not re-run the pipeline.

    Per engineering/decisions/2026-06-17-fc-async-task-mode.md and project
    memory ``project_fc_async_dedup_at_platform_layer.md``: FC dedups by
    ``X-Fc-Async-Task-Id`` at platform layer (409, request not delivered).
    If FC forwards, ``execute_task`` returns the existing JobInfo without
    re-running.  We run this test on the generate task_id (cheapest).
    """

    def test_duplicate_does_not_rerun(
        self,
        client: httpx.Client,
        generate_task_id: str,
        generate_task: dict,
    ):
        first_created_at = generate_task["created_at"]
        first_completed_at = generate_task["completed_at"]
        first_pdb_id = (generate_task.get("input_params") or {}).get("pdb_id")

        # Resubmit same task_id with a different pdb_id — must not stick.
        with open(POCKET_PDB, "rb") as fp, open(LIGAND_SDF, "rb") as fl:
            r2 = client.post(
                "/api/tasks/generate",
                files={
                    "target": (POCKET_PDB.name, fp.read(), "chemical/x-pdb"),
                    "ligand": (LIGAND_SDF.name, fl.read(), "chemical/x-mdl-sdfile"),
                },
                data={
                    "n_samples": "5",
                    "pdb_id": "should_not_apply",
                    "zbetas": "0.0",
                    "temps": "0.5",
                },
                headers=_async_headers(generate_task_id),
            )
        assert r2.status_code in (202, 409), (
            f"expected 409 (FC platform dedup) or 202 (FC forwards → framework "
            f"dedups); got {r2.status_code} body={r2.text!r}"
        )
        if r2.status_code == 202:
            time.sleep(15)

        re_query = _get_with_retry(client, f"/api/jobs/{generate_task_id}").json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created_at, (
            "duplicate async submit must not reset created_at"
        )
        assert re_query["completed_at"] == first_completed_at, (
            "duplicate async submit must not re-run the pipeline"
        )
        assert (re_query.get("input_params") or {}).get("pdb_id") == first_pdb_id, (
            "duplicate async submit must not overwrite input_params"
        )
