"""FC async task mode tests for diffdock-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/diffdock-server/tests/test_fc_task.py -v

Validates ``/api/tasks/dock`` end-to-end under FC async task mode
(``X-Fc-Invocation-Type: Async``) across all three input combos:
PDB + SDF, PDB + SMILES, protein_sequence + SMILES (ESMFold path).

Payload sizing — all three tests use vendored 1a0q:

    1a0q_protein.pdb : ~500 KB
    1a0q_ligand.sdf  :   ~3 KB
    ─────────────────────────
    worst case       : ~503 KB (still under FC's 6 MB body cap)
"""

from __future__ import annotations

import io
import time
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest

from bioagent_service.fc_testing import fc_url, poll_job

SERVICE = "diffdock-server"

DATA_DIR = Path(__file__).resolve().parent / "data"
PROTEIN_PDB = DATA_DIR / "1a0q_protein.pdb"
LIGAND_SDF = DATA_DIR / "1a0q_ligand.sdf"

# 1a0q sequence chain A (first 300 aa) — enough to fold with ESMFold and
# small enough to fit on T4/A10 without swap.  This trims the test wall time
# meaningfully vs feeding the full ~1000 aa 1a0q sequence.
LIGAND_SMILES = "COc1ccc(C#N)cc1"   # simple benzonitrile — RDKit-parseable
# Short sanity sequence — 300 aa is a common docking-friendly target size.
# Sourced from 1a0q chain A residues 1-300 in FASTA form (trimmed for the
# test wall-time budget).
PROTEIN_SEQUENCE = (
    "MSFEFTKGWWLPFEEIDSFQGTLNAAVAWKRVYNKRQTINMIRSHFEEHVILADQAEQFN"
    "AKFTFERASSEQNAIYYVYENTLAAPRSFAADFEQFKSMLAKVGGLHESQLIRETLAKAI"
    "VYRDYDRPAKQATNMKFRIENVFDVLGAFYAKVSAPTFAKMKAMYAKLYARDATLPQFVA"
    "MMTNKRAFATNEDVDIAQFYIKKAADAAAYFEQLYKPTKFVEHSLSAPFPAVLPYFEKGA"
    "VAKSPFDLDAALKAKPTSDPRSAFYYFTMKGGDHLKYAKKPKTS"
)

# DiffDock-L single-complex sampling on A10 takes ~60-180 s for
# samples_per_complex=10, inference_steps=20.  ESMFold path adds ~30-60 s.
# Allow a comfortable timeout for FC's cold-start + 429 windows.
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
def pdb_sdf_task_id() -> str:
    return f"fc-async-diffdock-pdb-sdf-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def pdb_smiles_task_id() -> str:
    return f"fc-async-diffdock-pdb-smi-{int(time.time())}-{uuid.uuid4().hex[:6]}"


@pytest.fixture(scope="module")
def seq_smiles_task_id() -> str:
    return f"fc-async-diffdock-seq-smi-{int(time.time())}-{uuid.uuid4().hex[:6]}"


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
    """GET that retries on FC HTTP-gateway 429 throttling."""
    last: httpx.Response | None = None
    for _attempt in range(max_attempts):
        last = client.get(path)
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    assert last is not None
    return last


def _poll_to_completion(client: httpx.Client, task_id: str) -> dict:
    final = poll_job(
        client, "", task_id,
        timeout_s=POLL_TIMEOUT_S,
        interval_s=POLL_INTERVAL_S,
        max_transient_errors=60,
    )
    assert final["status"] == "completed", f"task did not complete: {final}"
    return final


def _sdf_looks_valid(content: bytes) -> bool:
    text = content.decode("utf-8", errors="replace")
    return "$$$$" in text and any(k in text for k in (" C ", " N ", " O "))


# ---------------------------------------------------------------------------
# Per-endpoint submit + task fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pdb_sdf_submit_response(
    client: httpx.Client, pdb_sdf_task_id: str,
) -> httpx.Response:
    with open(PROTEIN_PDB, "rb") as fp, open(LIGAND_SDF, "rb") as fl:
        return client.post(
            "/api/tasks/dock",
            files={
                "protein": (PROTEIN_PDB.name, fp.read(), "chemical/x-pdb"),
                "ligand": (LIGAND_SDF.name, fl.read(), "chemical/x-mdl-sdfile"),
            },
            data={
                "complex_name": "1a0q_pdb_sdf",
                # Quick-run params to keep wall time reasonable.
                "samples_per_complex": "3",
                "inference_steps": "10",
                "actual_steps": "10",
                "batch_size": "3",
                "seed": "42",
            },
            headers=_async_headers(pdb_sdf_task_id),
        )


@pytest.fixture(scope="module")
def pdb_sdf_task(
    client: httpx.Client,
    pdb_sdf_task_id: str,
    pdb_sdf_submit_response: httpx.Response,
) -> dict:
    assert pdb_sdf_submit_response.status_code == 202, (
        f"async pdb+sdf submit returned "
        f"{pdb_sdf_submit_response.status_code}: "
        f"{pdb_sdf_submit_response.text!r}"
    )
    return _poll_to_completion(client, pdb_sdf_task_id)


@pytest.fixture(scope="module")
def pdb_smiles_submit_response(
    client: httpx.Client, pdb_smiles_task_id: str,
) -> httpx.Response:
    with open(PROTEIN_PDB, "rb") as fp:
        return client.post(
            "/api/tasks/dock",
            files={
                "protein": (PROTEIN_PDB.name, fp.read(), "chemical/x-pdb"),
            },
            data={
                "ligand_description": LIGAND_SMILES,
                "complex_name": "1a0q_pdb_smi",
                "samples_per_complex": "3",
                "inference_steps": "10",
                "actual_steps": "10",
                "batch_size": "3",
                "seed": "42",
            },
            headers=_async_headers(pdb_smiles_task_id),
        )


@pytest.fixture(scope="module")
def pdb_smiles_task(
    client: httpx.Client,
    pdb_smiles_task_id: str,
    pdb_smiles_submit_response: httpx.Response,
) -> dict:
    assert pdb_smiles_submit_response.status_code == 202, (
        f"async pdb+smiles submit returned "
        f"{pdb_smiles_submit_response.status_code}: "
        f"{pdb_smiles_submit_response.text!r}"
    )
    return _poll_to_completion(client, pdb_smiles_task_id)


@pytest.fixture(scope="module")
def seq_smiles_submit_response(
    client: httpx.Client, seq_smiles_task_id: str,
) -> httpx.Response:
    return client.post(
        "/api/tasks/dock",
        data={
            "protein_sequence": PROTEIN_SEQUENCE,
            "ligand_description": LIGAND_SMILES,
            "complex_name": "seq_smiles",
            "samples_per_complex": "3",
            "inference_steps": "10",
            "actual_steps": "10",
            "batch_size": "3",
            "seed": "42",
        },
        headers=_async_headers(seq_smiles_task_id),
    )


@pytest.fixture(scope="module")
def seq_smiles_task(
    client: httpx.Client,
    seq_smiles_task_id: str,
    seq_smiles_submit_response: httpx.Response,
) -> dict:
    assert seq_smiles_submit_response.status_code == 202, (
        f"async seq+smiles submit returned "
        f"{seq_smiles_submit_response.status_code}: "
        f"{seq_smiles_submit_response.text!r}"
    )
    return _poll_to_completion(client, seq_smiles_task_id)


# ===========================================================================
# Section 1: Submit semantics + OpenAPI registration.
# ===========================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_pdb_sdf_returns_202(self, pdb_sdf_submit_response):
        assert pdb_sdf_submit_response.status_code == 202, (
            f"expected 202; got {pdb_sdf_submit_response.status_code} "
            f"body={pdb_sdf_submit_response.text!r}"
        )

    def test_pdb_smiles_returns_202(self, pdb_smiles_submit_response):
        assert pdb_smiles_submit_response.status_code == 202, (
            f"expected 202; got {pdb_smiles_submit_response.status_code} "
            f"body={pdb_smiles_submit_response.text!r}"
        )

    def test_seq_smiles_returns_202(self, seq_smiles_submit_response):
        assert seq_smiles_submit_response.status_code == 202, (
            f"expected 202; got {seq_smiles_submit_response.status_code} "
            f"body={seq_smiles_submit_response.text!r}"
        )

    def test_task_endpoint_registered_in_openapi(self, client):
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200
        assert "/api/tasks/dock" in r.json().get("paths", {})


# ===========================================================================
# Section 2: Per-input-combo completion + outputs.
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
class TestAsyncDockPdbSdf:
    """Baseline case: PDB + SDF file upload."""

    def test_completed(self, pdb_sdf_task, pdb_sdf_task_id, client):
        _assert_completed(pdb_sdf_task, pdb_sdf_task_id)
        files = _get_with_retry(
            client, f"/api/jobs/{pdb_sdf_task_id}/files"
        ).json()["files"]
        # Output layout: output/<complex_name>/rank1.sdf +
        #                output/<complex_name>/rank<r>_confidence<c>.sdf
        assert any(
            f.endswith("/rank1.sdf") for f in files
        ), f"rank1.sdf missing: {files}"
        assert any(
            "_confidence" in f and f.endswith(".sdf") for f in files
        ), f"no rank*_confidence*.sdf files: {files}"

    def test_input_params_echoed(self, pdb_sdf_task):
        params = pdb_sdf_task.get("input_params") or {}
        assert params.get("complex_name") == "1a0q_pdb_sdf"
        assert params.get("samples_per_complex") == 3

    def test_confidence_scores_json_written(
        self, client, pdb_sdf_task_id, pdb_sdf_task,
    ):
        """The wrapper post-processor writes confidence_scores.json."""
        files = _get_with_retry(
            client, f"/api/jobs/{pdb_sdf_task_id}/files"
        ).json()["files"]
        json_name = next(
            (f for f in files if f.endswith("/confidence_scores.json")),
            None,
        )
        assert json_name is not None, (
            f"confidence_scores.json missing: {files}"
        )
        r = _get_with_retry(
            client, f"/api/jobs/{pdb_sdf_task_id}/file/{json_name}",
        )
        assert r.status_code == 200
        import json
        entries = json.loads(r.content)
        assert isinstance(entries, list)
        assert len(entries) > 0
        assert all("rank" in e and "confidence" in e for e in entries)


@pytest.mark.fc
class TestAsyncDockPdbSmiles:
    """PDB structure + SMILES ligand string."""

    def test_completed(self, pdb_smiles_task, pdb_smiles_task_id, client):
        _assert_completed(pdb_smiles_task, pdb_smiles_task_id)
        files = _get_with_retry(
            client, f"/api/jobs/{pdb_smiles_task_id}/files"
        ).json()["files"]
        assert any(
            f.endswith("/rank1.sdf") for f in files
        ), f"rank1.sdf missing: {files}"

    def test_input_params_echoed(self, pdb_smiles_task):
        params = pdb_smiles_task.get("input_params") or {}
        assert params.get("ligand_description") == LIGAND_SMILES
        assert params.get("complex_name") == "1a0q_pdb_smi"

    def test_output_sdf_downloadable(
        self, client, pdb_smiles_task_id, pdb_smiles_task,
    ):
        files = _get_with_retry(
            client, f"/api/jobs/{pdb_smiles_task_id}/files"
        ).json()["files"]
        sdf_name = next(f for f in files if f.endswith("/rank1.sdf"))
        r = _get_with_retry(
            client, f"/api/jobs/{pdb_smiles_task_id}/file/{sdf_name}",
        )
        assert r.status_code == 200
        assert _sdf_looks_valid(r.content), (
            f"SDF content malformed: {r.content[:200]!r}"
        )


@pytest.mark.fc
class TestAsyncDockSequenceSmiles:
    """protein_sequence (→ ESMFold) + SMILES.  Slowest path."""

    def test_completed(self, seq_smiles_task, seq_smiles_task_id, client):
        _assert_completed(seq_smiles_task, seq_smiles_task_id)
        files = _get_with_retry(
            client, f"/api/jobs/{seq_smiles_task_id}/files"
        ).json()["files"]
        # In this branch upstream writes seq_smiles_esmfold.pdb next to
        # the ranked SDFs.  Its presence is proof the ESMFold branch ran.
        assert any(
            f.endswith("_esmfold.pdb") for f in files
        ), (f"ESMFold intermediate PDB not found; the sequence branch may not "
             f"have run: {files}")
        assert any(
            f.endswith("/rank1.sdf") for f in files
        ), f"rank1.sdf missing: {files}"

    def test_input_params_echoed(self, seq_smiles_task):
        params = seq_smiles_task.get("input_params") or {}
        assert params.get("protein_sequence") == PROTEIN_SEQUENCE
        assert params.get("ligand_description") == LIGAND_SMILES

    def test_esmfold_pdb_looks_valid(
        self, client, seq_smiles_task_id, seq_smiles_task,
    ):
        files = _get_with_retry(
            client, f"/api/jobs/{seq_smiles_task_id}/files"
        ).json()["files"]
        pdb_name = next(f for f in files if f.endswith("_esmfold.pdb"))
        r = _get_with_retry(
            client, f"/api/jobs/{seq_smiles_task_id}/file/{pdb_name}",
        )
        assert r.status_code == 200
        text = r.content.decode("utf-8", errors="replace")
        assert "ATOM" in text
        assert "END" in text or text.count("\n") > 100


# ===========================================================================
# Section 3: Identity — X-Bioagent-Job-Id propagates to JobInfo.job_id.
# ===========================================================================


@pytest.mark.fc
class TestAsyncTaskIdentity:
    def test_pdb_sdf_id_matches(self, pdb_sdf_task, pdb_sdf_task_id):
        assert pdb_sdf_task["job_id"] == pdb_sdf_task_id, (
            "task endpoint must use X-Bioagent-Job-Id as JobInfo.job_id"
        )

    def test_pdb_smiles_id_matches(self, pdb_smiles_task, pdb_smiles_task_id):
        assert pdb_smiles_task["job_id"] == pdb_smiles_task_id

    def test_seq_smiles_id_matches(self, seq_smiles_task, seq_smiles_task_id):
        assert seq_smiles_task["job_id"] == seq_smiles_task_id


# ===========================================================================
# Section 4: Job lifecycle on the cheapest (pdb_sdf) task.
# ===========================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_job_visible_via_status_endpoint(
        self, client, pdb_sdf_task_id, pdb_sdf_task,
    ):
        r = _get_with_retry(client, f"/api/jobs/{pdb_sdf_task_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == pdb_sdf_task_id
        assert body["status"] == "completed"

    def test_job_log_endpoint(self, client, pdb_sdf_task_id, pdb_sdf_task):
        r = _get_with_retry(client, f"/api/jobs/{pdb_sdf_task_id}/log")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == pdb_sdf_task_id
        log_text = body.get("log") or body.get("text", "")
        assert isinstance(log_text, str)

    def test_job_download_zip(self, client, pdb_sdf_task_id, pdb_sdf_task):
        r = _get_with_retry(client, f"/api/jobs/{pdb_sdf_task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert any(
            n.endswith("/rank1.sdf") for n in names
        ), f"rank1.sdf missing from zip: {names}"

    def test_single_file_download_missing_returns_404(
        self, client, pdb_sdf_task_id, pdb_sdf_task,
    ):
        r = _get_with_retry(
            client, f"/api/jobs/{pdb_sdf_task_id}/file/nonexistent.xyz"
        )
        assert r.status_code == 404


# ===========================================================================
# Section 5: Duplicate dedup — FC platform + framework layer.
# ===========================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    """Resubmitting same X-Fc-Async-Task-Id must not re-run the pipeline.

    Per engineering/decisions/2026-06-17-fc-async-task-mode.md.  Run on
    the pdb_sdf task_id (cheapest).
    """

    def test_duplicate_does_not_rerun(
        self,
        client: httpx.Client,
        pdb_sdf_task_id: str,
        pdb_sdf_task: dict,
    ):
        first_created_at = pdb_sdf_task["created_at"]
        first_completed_at = pdb_sdf_task["completed_at"]
        first_complex_name = (
            pdb_sdf_task.get("input_params") or {}
        ).get("complex_name")

        with open(PROTEIN_PDB, "rb") as fp, open(LIGAND_SDF, "rb") as fl:
            r2 = client.post(
                "/api/tasks/dock",
                files={
                    "protein": (PROTEIN_PDB.name, fp.read(), "chemical/x-pdb"),
                    "ligand": (LIGAND_SDF.name, fl.read(), "chemical/x-mdl-sdfile"),
                },
                data={
                    "complex_name": "SHOULD_NOT_APPLY",
                    "samples_per_complex": "3",
                    "inference_steps": "10",
                    "actual_steps": "10",
                    "batch_size": "3",
                    "seed": "42",
                },
                headers=_async_headers(pdb_sdf_task_id),
            )
        assert r2.status_code in (202, 409), (
            f"expected 409 (FC platform dedup) or 202 (FC forwards → framework "
            f"dedups); got {r2.status_code} body={r2.text!r}"
        )
        if r2.status_code == 202:
            time.sleep(15)

        re_query = _get_with_retry(client, f"/api/jobs/{pdb_sdf_task_id}").json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created_at, (
            "duplicate async submit must not reset created_at"
        )
        assert re_query["completed_at"] == first_completed_at, (
            "duplicate async submit must not re-run the pipeline"
        )
        assert (re_query.get("input_params") or {}).get("complex_name") == first_complex_name, (
            "duplicate async submit must not overwrite input_params"
        )
