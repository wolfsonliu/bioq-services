"""Offline FastAPI route tests for mmseqs2-server.

No real mmseqs binary is needed: ``runner.submit`` is replaced with a stub that
returns a fake ``JobInfo`` so the input-validation + response-shape paths can
be exercised end-to-end. Covers Task 4.1 of the Stage 4 plan.
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Iterator

import pytest
from bioagent_service import JobInfo, JobStatus
from fastapi.testclient import TestClient

SERVICE_DIR = Path(__file__).resolve().parent.parent


def _reload_server_pkg() -> None:
    """Drop cached `server.*` modules and re-register the package spec.

    `tests/conftest.py` does this once at import time, but our env-var
    monkeypatch needs to take effect at *settings load time*, which means
    re-importing `server.app` against a clean module table. Without
    re-registering the package spec the next ``import server.app`` would
    raise ``ModuleNotFoundError`` because we popped ``server`` from
    ``sys.modules``.
    """
    for mod in [m for m in sys.modules if m == "server" or m.startswith("server.")]:
        sys.modules.pop(mod, None)
    spec = importlib.util.spec_from_file_location(
        "server",
        SERVICE_DIR / "__init__.py",
        submodule_search_locations=[str(SERVICE_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["server"] = module
    spec.loader.exec_module(module)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """TestClient against a freshly-imported server.app.

    ``server.app`` builds settings + adapter + FastAPI at import time, so any
    test that needs an isolated jobs_base_dir must monkeypatch the env vars
    *before* the import — which means dropping anything cached under
    ``sys.modules`` first.
    """
    monkeypatch.setenv("MMSEQS2_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("MMSEQS2_DB_DIR", str(tmp_path / "db"))
    # Disable the keepalive thread — it's harmless but spams logs during tests.
    monkeypatch.setenv("MMSEQS2_KEEPALIVE_INTERVAL_S", "0")
    (tmp_path / "db").mkdir(parents=True, exist_ok=True)

    _reload_server_pkg()
    server_app = importlib.import_module("server.app")
    with TestClient(server_app.app) as tc:
        yield tc


@pytest.fixture
def stub_submit(client: TestClient) -> list[dict]:
    """Replace ``app.state.runner.submit`` with a capturing stub.

    Returns a list that the test inspects after issuing the request. Each
    captured entry has ``label``, ``input_params``, ``argv``, ``job_id``.
    """
    captured: list[dict] = []

    def _fake_submit(*, build_argv, label, env=None, cwd=None, input_params=None):
        job_id = f"stub-{label}-{len(captured)}"
        # Run build_argv against a real tmp dir so the closure's side-effects
        # (fasta write) succeed; some tests rely on argv inspection.
        import tempfile
        job_dir = Path(tempfile.mkdtemp(prefix="mmseqs2-test-"))
        argv = build_argv(job_id, job_dir)
        captured.append({
            "job_id": job_id,
            "label": label,
            "argv": argv,
            "input_params": input_params,
            "job_dir": job_dir,
        })
        return JobInfo(job_id=job_id, status=JobStatus.PENDING)

    client.app.state.runner.submit = _fake_submit
    return captured


# ---------------------------------------------------------------------------
# Health / manifest
# ---------------------------------------------------------------------------


def test_healthz_returns_ok(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "mmseqs2"


def test_healthz_detail_has_mmseqs2_fields(client: TestClient) -> None:
    body = client.get("/healthz/detail").json()
    # The mmseqs2 override exposes these — distinct from the framework default.
    assert "db_loaded" in body
    assert "gpu_free_mb" in body
    assert "active_jobs" in body
    assert body["service"] == "mmseqs2"


def test_manifest_returns_protocol_info(client: TestClient) -> None:
    body = client.get("/api/manifest").json()
    assert body["service"] == "mmseqs2"
    # service_specific carries the ColabFold protocol summary.
    assert "ColabFold" in body["service_specific"]["protocol"]


# ---------------------------------------------------------------------------
# POST /ticket/msa — happy path + validation
# ---------------------------------------------------------------------------


_VALID_MONOMER_Q = ">q1\nMKQHKAMIVALIVICITAVVAAL\n"


def test_post_ticket_msa_valid_returns_pending(
    client: TestClient, stub_submit: list[dict]
) -> None:
    resp = client.post(
        "/ticket/msa",
        data={"q": _VALID_MONOMER_Q, "mode": "env"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "PENDING"
    assert isinstance(body["id"], str) and body["id"]
    # Stub was invoked once with the right label.
    assert len(stub_submit) == 1
    assert stub_submit[0]["label"] == "msa"


@pytest.mark.parametrize(
    "data,reason",
    [
        ({"q": "", "mode": "env"},                             "empty q"),
        ({"q": _VALID_MONOMER_Q, "mode": "nonsense"},          "bad mode"),
        ({"q": ">q1\nMKQHJZB\n", "mode": "env"},               "invalid AA"),
        ({"q": _VALID_MONOMER_Q, "mode": "pairgreedy"},        "paired-mode-on-msa-endpoint"),
    ],
)
def test_post_ticket_msa_invalid_returns_error_envelope(
    client: TestClient,
    stub_submit: list[dict],
    data: dict,
    reason: str,
) -> None:
    """Bad inputs return HTTP 200 + ``{"status": "ERROR"}`` (no id field)."""
    resp = client.post("/ticket/msa", data=data)
    assert resp.status_code == 200, reason
    body = resp.json()
    assert body["status"] == "ERROR", reason
    assert "id" not in body, reason
    # Runner was never reached.
    assert len(stub_submit) == 0, reason


# ---------------------------------------------------------------------------
# POST /ticket/pair — happy path + validation
# ---------------------------------------------------------------------------


_VALID_PAIRED_Q = ">chainA\nMKQHKAM\n>chainB\nLLLLLLL\n"


def test_post_ticket_pair_valid_returns_pending(
    client: TestClient, stub_submit: list[dict]
) -> None:
    resp = client.post(
        "/ticket/pair",
        data={"q": _VALID_PAIRED_Q, "mode": "pairgreedy"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "PENDING"
    assert isinstance(body["id"], str) and body["id"]
    assert len(stub_submit) == 1
    assert stub_submit[0]["label"] == "pair"


def test_post_ticket_pair_serializes_chains_with_colon(
    client: TestClient, stub_submit: list[dict]
) -> None:
    """Regression: /ticket/pair must collapse multi-record FASTA into a
    single ``:``-joined complex record before handing off to the orchestrator.

    Root cause of an earlier FC bug: the orchestrator's ``get_queries``
    treats each ``>`` record as an independent monomer; without the collapse,
    a 2-chain query is misclassified as 2 monomers and paired search is
    skipped (WARNING: "pair-mode=paired requested but query is a monomer").
    See services/mmseqs2-server/_colabfold_helpers.py:212-271.
    """
    resp = client.post(
        "/ticket/pair",
        data={"q": _VALID_PAIRED_Q, "mode": "pairgreedy"},
    )
    assert resp.status_code == 200

    # Inspect the FASTA the closure wrote for the orchestrator.
    job_dir = stub_submit[0]["job_dir"]
    fasta_text = (job_dir / "input" / "query.fasta").read_text(encoding="utf-8")
    # Should be ONE record with chains joined by ':' (ColabFold complex format).
    assert fasta_text.count(">") == 1, (
        f"pair fasta must be a single-record complex, got:\n{fasta_text}"
    )
    assert "MKQHKAM:LLLLLLL" in fasta_text, (
        f"pair fasta must join chains with ':' — got:\n{fasta_text}"
    )


def test_post_ticket_msa_keeps_multi_record_layout(
    client: TestClient, stub_submit: list[dict]
) -> None:
    """Unpaired /ticket/msa must NOT collapse — each record stays independent.

    The mirror of the paired-collapse regression: for /ticket/msa, multiple
    FASTA records represent N independent monomer queries and the
    orchestrator emits one MSA per record.  Collapsing would silently merge
    them into a fake complex.
    """
    multi_monomer = ">q1\nMKQHKAM\n>q2\nLLLLLLL\n"
    resp = client.post(
        "/ticket/msa",
        data={"q": multi_monomer, "mode": "env"},
    )
    assert resp.status_code == 200

    job_dir = stub_submit[0]["job_dir"]
    fasta_text = (job_dir / "input" / "query.fasta").read_text(encoding="utf-8")
    assert fasta_text.count(">") == 2, (
        f"msa fasta must preserve N-record layout, got:\n{fasta_text}"
    )
    # Chains must NOT be joined by ':' (that's the paired-complex shape).
    assert ":" not in fasta_text


@pytest.mark.parametrize(
    "data,reason",
    [
        ({"q": _VALID_MONOMER_Q, "mode": "pairgreedy"},  "single chain on /ticket/pair"),
        ({"q": _VALID_PAIRED_Q, "mode": "env"},          "monomer mode on /ticket/pair"),
    ],
)
def test_post_ticket_pair_invalid_returns_error_envelope(
    client: TestClient,
    stub_submit: list[dict],
    data: dict,
    reason: str,
) -> None:
    resp = client.post("/ticket/pair", data=data)
    assert resp.status_code == 200, reason
    body = resp.json()
    assert body["status"] == "ERROR", reason
    assert len(stub_submit) == 0, reason


# ---------------------------------------------------------------------------
# GET /ticket/{id}
# ---------------------------------------------------------------------------


def test_get_ticket_unknown_id_returns_error_envelope(client: TestClient) -> None:
    """Unknown job id → 200 + ``{"status": "ERROR", "error": "job not found"}``.

    The ColabFold client doesn't handle 404 gracefully so the protocol uses
    200 + ERROR.
    """
    resp = client.get("/ticket/does-not-exist")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ERROR"
    assert body["error"] == "job not found"


def test_get_ticket_completed_job_returns_complete(client: TestClient) -> None:
    info = JobInfo(job_id="abc123", status=JobStatus.COMPLETED)
    client.app.state.job_store.get = lambda jid: info if jid == "abc123" else None

    body = client.get("/ticket/abc123").json()
    assert body["status"] == "COMPLETE"
    assert body["id"] == "abc123"


def test_get_ticket_failed_job_returns_error_truncated(client: TestClient) -> None:
    """Long error strings are truncated to <= 200 chars before being returned."""
    long_err = "x" * 500
    info = JobInfo(
        job_id="bad", status=JobStatus.FAILED, error_summary=long_err,
    )
    client.app.state.job_store.get = lambda jid: info if jid == "bad" else None

    body = client.get("/ticket/bad").json()
    assert body["status"] == "ERROR"
    assert body["error"] is not None
    assert len(body["error"]) <= 200


# ---------------------------------------------------------------------------
# GET /result/download/{id}
# ---------------------------------------------------------------------------


def test_get_result_download_unknown_id_returns_503(client: TestClient) -> None:
    resp = client.get("/result/download/no-such-job")
    assert resp.status_code == 503
    assert resp.json()["status"] == "ERROR"


def test_get_result_download_completed_streams_tar_gz(
    client: TestClient, tmp_path: Path
) -> None:
    """A COMPLETED job whose output dir has .a3m files yields a tar.gz body."""
    job_id = "okjob"
    info = JobInfo(job_id=job_id, status=JobStatus.COMPLETED)
    client.app.state.job_store.get = lambda jid: info if jid == job_id else None

    # Lay down the expected output layout under the *real* adapter's job_dir.
    adapter = client.app.state.adapter
    out_dir = adapter.output_dir(adapter.job_dir(job_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "q1.a3m").write_text(">q1\nMKQHKAM\n")
    (out_dir / "q2.a3m").write_text(">q2\nLLLLLLL\n")

    resp = client.get(f"/result/download/{job_id}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-tar")

    # Verify the tarball is well-formed + contains both files.
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tf:
        members = sorted(m.name for m in tf.getmembers())
    assert members == ["q1.a3m", "q2.a3m"]


# ---------------------------------------------------------------------------
# Privacy: sequence content must NOT leak into JobInfo.input_params
# ---------------------------------------------------------------------------


def test_submit_does_not_leak_sequence_into_input_params(
    client: TestClient, stub_submit: list[dict]
) -> None:
    """JobInfo.input_params is persisted to NAS — never put the raw seq there."""
    # Use a unique substring we can search for in the captured input_params.
    unique = "WWWMKQHEEEE"
    q = f">probe\n{unique}\n"
    resp = client.post("/ticket/msa", data={"q": q, "mode": "env"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "PENDING"

    assert len(stub_submit) == 1
    serialized = json.dumps(stub_submit[0]["input_params"])
    assert unique not in serialized
    # Sanity: the count + mode WERE captured (so the test is meaningfully
    # checking absence of the seq specifically, not absence of everything).
    assert stub_submit[0]["input_params"]["sequence_count"] == 1
    assert stub_submit[0]["input_params"]["mode"] == "env"


# ---------------------------------------------------------------------------
# /api/tasks/{msa,pair} — synchronous task endpoints (FC Async Task Mode)
#
# Task endpoints share the argv builder with /ticket/* but execute via
# `execute_task` (blocking, single thread). We patch `SubprocessRunner.run`
# so the suite stays fast — without real mmseqs the subprocess would always
# fail anyway, but mocking makes the COMPLETED path testable too. The
# ticket-endpoint tests above continue to use `stub_submit` which intercepts
# at a higher layer (`runner.submit`).
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_task_subprocess(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace SubprocessRunner.run + adapter.detect_outputs so task endpoints
    finalize as COMPLETED without invoking real mmseqs.

    Returns a dict that lets tests configure ``rc`` / ``outputs_present``.
    """
    cfg: dict = {"rc": 0, "outputs_present": True, "calls": []}

    def _fake_run(argv, log_path, env=None, cwd=None) -> int:
        cfg["calls"].append({"argv": list(argv), "log_path": str(log_path)})
        # Touch log so finalize_job's tail-read does not 500 on missing file.
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("stubbed mmseqs run\n", encoding="utf-8")
        return cfg["rc"]

    monkeypatch.setattr(
        "bioagent_service.task_endpoint.SubprocessRunner.run", staticmethod(_fake_run)
    )

    # `execute_task` ends with `finalize_job` which calls `adapter.detect_outputs`
    # to decide COMPLETED vs FAILED on rc=0. Force it to True so we don't need
    # to fabricate real .a3m files for every test.
    from server import adapter as adapter_mod  # type: ignore

    monkeypatch.setattr(
        adapter_mod.MMseqs2JobAdapter,
        "detect_outputs",
        lambda self, job_dir: cfg["outputs_present"],
    )
    return cfg


def test_post_tasks_msa_returns_completed(
    client: TestClient, stub_task_subprocess: dict
) -> None:
    resp = client.post(
        "/api/tasks/msa",
        data={"q": _VALID_MONOMER_Q, "mode": "env"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["job_id"]
    assert body["completed_at"] is not None
    # Exactly one subprocess invocation.
    assert len(stub_task_subprocess["calls"]) == 1
    # input_params carries the summary, never the raw sequence.
    assert body["input_params"]["mode"] == "env"
    assert body["input_params"]["sequence_count"] == 1
    serialized = json.dumps(body["input_params"])
    assert "MKQHKAM" not in serialized  # raw query absent


def test_post_tasks_msa_honors_x_bioagent_job_id(
    client: TestClient, stub_task_subprocess: dict
) -> None:
    resp = client.post(
        "/api/tasks/msa",
        data={"q": _VALID_MONOMER_Q, "mode": "env"},
        headers={"X-Bioagent-Job-Id": "task-msa-fixed-id"},
    )
    assert resp.status_code == 200
    assert resp.json()["job_id"] == "task-msa-fixed-id"


def test_post_tasks_msa_falls_back_to_x_fc_async_task_id(
    client: TestClient, stub_task_subprocess: dict
) -> None:
    """X-Fc-Async-Task-Id is the secondary header when X-Bioagent-Job-Id absent."""
    resp = client.post(
        "/api/tasks/msa",
        data={"q": _VALID_MONOMER_Q, "mode": "env"},
        headers={"X-Fc-Async-Task-Id": "fc-async-id-001"},
    )
    assert resp.status_code == 200
    assert resp.json()["job_id"] == "fc-async-id-001"


def test_post_tasks_msa_duplicate_is_idempotent(
    client: TestClient, stub_task_subprocess: dict
) -> None:
    """Same X-Bioagent-Job-Id twice → second call returns the existing JobInfo
    without re-running the subprocess (framework-level dedup)."""
    headers = {"X-Bioagent-Job-Id": "dup-task-1"}
    first = client.post(
        "/api/tasks/msa",
        data={"q": _VALID_MONOMER_Q, "mode": "env"},
        headers=headers,
    )
    assert first.status_code == 200
    assert len(stub_task_subprocess["calls"]) == 1
    first_created = first.json()["created_at"]

    second = client.post(
        "/api/tasks/msa",
        data={"q": _VALID_MONOMER_Q, "mode": "all"},  # different mode — must be ignored
        headers=headers,
    )
    assert second.status_code == 200
    # Subprocess must NOT have been invoked a second time.
    assert len(stub_task_subprocess["calls"]) == 1
    assert second.json()["job_id"] == "dup-task-1"
    assert second.json()["created_at"] == first_created


@pytest.mark.parametrize(
    "data,reason",
    [
        ({"q": "", "mode": "env"},                       "empty q"),
        ({"q": _VALID_MONOMER_Q, "mode": "nonsense"},    "bad mode"),
        ({"q": ">q1\nMKQHJZB\n", "mode": "env"},         "invalid AA"),
        ({"q": _VALID_MONOMER_Q, "mode": "pairgreedy"},  "paired mode on /api/tasks/msa"),
    ],
)
def test_post_tasks_msa_invalid_returns_422(
    client: TestClient,
    stub_task_subprocess: dict,
    data: dict,
    reason: str,
) -> None:
    """Task endpoints use HTTP 422 for input errors (not ColabFold's 200+ERROR)."""
    resp = client.post("/api/tasks/msa", data=data)
    assert resp.status_code == 422, reason
    assert len(stub_task_subprocess["calls"]) == 0, reason


def test_post_tasks_pair_returns_completed(
    client: TestClient, stub_task_subprocess: dict
) -> None:
    resp = client.post(
        "/api/tasks/pair",
        data={"q": _VALID_PAIRED_Q, "mode": "pairgreedy"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["input_params"]["sequence_count"] == 2
    assert body["input_params"]["mode"] == "pairgreedy"


@pytest.mark.parametrize(
    "data,reason",
    [
        ({"q": _VALID_MONOMER_Q, "mode": "pairgreedy"},  "single chain on /api/tasks/pair"),
        ({"q": _VALID_PAIRED_Q, "mode": "env"},          "monomer mode on /api/tasks/pair"),
    ],
)
def test_post_tasks_pair_invalid_returns_422(
    client: TestClient,
    stub_task_subprocess: dict,
    data: dict,
    reason: str,
) -> None:
    resp = client.post("/api/tasks/pair", data=data)
    assert resp.status_code == 422, reason


def test_post_tasks_msa_failed_subprocess_returns_200_with_failed_status(
    client: TestClient, stub_task_subprocess: dict
) -> None:
    """Non-zero rc → 200 + JobInfo.status='failed' (not an HTTP error)."""
    stub_task_subprocess["rc"] = 7
    stub_task_subprocess["outputs_present"] = False

    resp = client.post(
        "/api/tasks/msa",
        data={"q": _VALID_MONOMER_Q, "mode": "env"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["failure_kind"] == "subprocess_error"
    # finalize_job writes "msa failed (rc=7)" into JobInfo.message.
    assert "rc=7" in (body.get("message") or "")


def test_task_endpoints_skipped_when_flag_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When task_endpoints_enabled=False the routes are not registered.

    Reloads the server with the env override so the ``if
    settings.task_endpoints_enabled:`` guard at module load time picks it up.
    """
    monkeypatch.setenv("MMSEQS2_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("MMSEQS2_DB_DIR", str(tmp_path / "db"))
    monkeypatch.setenv("MMSEQS2_KEEPALIVE_INTERVAL_S", "0")
    monkeypatch.setenv("MMSEQS2_TASK_ENDPOINTS_ENABLED", "false")
    (tmp_path / "db").mkdir(parents=True, exist_ok=True)

    _reload_server_pkg()
    server_app = importlib.import_module("server.app")
    with TestClient(server_app.app) as tc:
        # /ticket/* still there
        assert tc.post("/ticket/msa", data={"q": "", "mode": "env"}).status_code == 200
        # /api/tasks/* not registered → 404
        assert tc.post("/api/tasks/msa", data={"q": "", "mode": "env"}).status_code == 404
        assert tc.post("/api/tasks/pair", data={"q": "", "mode": "env"}).status_code == 404
