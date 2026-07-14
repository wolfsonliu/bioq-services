"""Offline tests for esmfold2-server (no real ESMFold2 model / GPU needed).

The endpoint handlers' subprocess call is replaced with a synchronous stub via
`monkeypatch` so we can exercise the JSON-building + argv-assembly logic
without a GPU or the actual ESMFold2 model installed.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


# ---- Shared fixtures ----


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ESMFOLD2_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("ESMFOLD2_ROOT", str(tmp_path / "esmfold2"))
    monkeypatch.setenv("ESMFOLD2_PYTHON", "/bin/true")
    monkeypatch.setenv("ESMFOLD2_INFERENCE_SCRIPT", "/bin/true")
    monkeypatch.setenv("ESMFOLD2_MODEL_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("ESMFOLD2_CCD_PATH", str(tmp_path / "weights" / "ccd.pkl"))
    (tmp_path / "esmfold2").mkdir(parents=True, exist_ok=True)
    (tmp_path / "weights").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


@pytest.fixture
def captured_argv(monkeypatch):
    captured: list[dict] = []

    def _fake_submit(build_argv, label, **kwargs):
        from bioagent_service import JobInfo, JobStatus

        job_id = f"stub-{label}-{len(captured)}"
        import tempfile

        job_dir = Path(tempfile.mkdtemp(prefix="esmfold2-test-"))
        argv = build_argv(job_id, job_dir)
        captured.append({"job_id": job_id, "label": label, "argv": argv, "job_dir": job_dir})
        return JobInfo(job_id=job_id, status=JobStatus.PENDING)

    return captured, _fake_submit


@pytest.fixture
def client_with_stub_runner(client, captured_argv):
    captured, fake_submit = captured_argv
    client.app.state.runner.submit = fake_submit
    return client, captured


# ---- Health / manifest ----


def test_health(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "esmfold2"
    assert "version" in health


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "esmfold2"


def test_manifest_lists_fold_endpoint(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/fold" in paths


def test_manifest_model_is_esmfold2(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["model"]["name"] == "ESMFold2"
    assert "mmCIF" in extras["model"]["output_format"]


def test_manifest_has_config_tips(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert "num_loops" in extras["config_tips"]
    assert "num_sampling_steps" in extras["config_tips"]


def test_endpoint_examples_present(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    assert by_path["/api/fold"]["examples"]


# ---- Settings ----


def test_settings_defaults():
    from server.settings import ESMFold2Settings

    class _Off(ESMFold2Settings):
        model_config = SettingsConfigDict(
            env_prefix="ESMFOLD2_TEST_", env_file=None, extra="ignore"
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/esmfold2_jobs")
    assert s.root == Path("/opt/esmfold2")
    assert s.python == "/opt/esmfold2/.venv/bin/python"
    assert s.inference_script == "/opt/esmfold2/inference.py"
    assert s.model_dir == Path("/opt/esmfold2/weights")
    assert s.ccd_path == Path("/opt/esmfold2/weights/ccd.pkl")
    assert s.max_concurrent_jobs == 1


def test_settings_env_override(monkeypatch):
    from server.settings import ESMFold2Settings

    monkeypatch.setenv("ESMFOLD2_PYTHON", "/custom/python")
    monkeypatch.setenv("ESMFOLD2_MODEL_DIR", "/nas/weights")
    s = ESMFold2Settings()
    assert s.python == "/custom/python"
    assert s.model_dir == Path("/nas/weights")


# ---- Model validation ----


def test_sequence_entry_ligand_requires_smiles_xor_ccd():
    from server.models import SequenceEntry

    with pytest.raises(ValueError, match="exactly one of"):
        SequenceEntry(type="ligand", id="B", smiles="CCO", ccd=["ATP"])
    with pytest.raises(ValueError, match="exactly one of"):
        SequenceEntry(type="ligand", id="B")
    SequenceEntry(type="ligand", id="B", smiles="CCO")
    SequenceEntry(type="ligand", id="B", ccd=["ATP"])


def test_sequence_entry_protein_requires_sequence():
    from server.models import SequenceEntry

    with pytest.raises(ValueError, match="requires `sequence`"):
        SequenceEntry(type="protein", id="A")
    SequenceEntry(type="protein", id="A", sequence="MKT")


def test_sequence_entry_protein_rejects_smiles():
    from server.models import SequenceEntry

    with pytest.raises(ValueError, match="cannot have"):
        SequenceEntry(type="protein", id="A", sequence="MKT", smiles="CCO")


def test_sequence_entry_ligand_rejects_sequence():
    from server.models import SequenceEntry

    with pytest.raises(ValueError, match="must not have"):
        SequenceEntry(type="ligand", id="B", smiles="CCO", sequence="MKT")


def test_fold_request_requires_sequences():
    from server.models import FoldRequest

    with pytest.raises(ValueError):
        FoldRequest(sequences=[])


def test_fold_request_valid():
    from server.models import FoldRequest, SequenceEntry

    req = FoldRequest(
        sequences=[SequenceEntry(type="protein", id="A", sequence="MKT")],
        num_loops=5,
        num_sampling_steps=100,
    )
    assert req.num_loops == 5
    assert req.num_sampling_steps == 100


# ---- Input JSON construction ----


def test_build_input_json_protein_only(tmp_path):
    from server.models import FoldRequest, SequenceEntry
    from server.tools import build_input_json

    req = FoldRequest(
        sequences=[SequenceEntry(type="protein", id="A", sequence="MKT")],
    )
    job_dir = tmp_path / "jobs" / "job1"
    job_dir.mkdir(parents=True)

    json_path = build_input_json(req, job_dir=job_dir, saved_msa_paths={})
    doc = json.loads(json_path.read_text())
    assert len(doc["sequences"]) == 1
    assert doc["sequences"][0]["type"] == "protein"
    assert doc["sequences"][0]["id"] == "A"
    assert doc["sequences"][0]["sequence"] == "MKT"


def test_build_input_json_with_ligand(tmp_path):
    from server.models import FoldRequest, SequenceEntry
    from server.tools import build_input_json

    req = FoldRequest(
        sequences=[
            SequenceEntry(type="protein", id="A", sequence="MKT"),
            SequenceEntry(type="ligand", id="B", ccd=["ATP"]),
        ],
    )
    job_dir = tmp_path / "jobs" / "job1"
    job_dir.mkdir(parents=True)

    json_path = build_input_json(req, job_dir=job_dir, saved_msa_paths={})
    doc = json.loads(json_path.read_text())
    assert len(doc["sequences"]) == 2
    assert doc["sequences"][1]["type"] == "ligand"
    assert doc["sequences"][1]["ccd"] == ["ATP"]


def test_build_input_json_with_msa(tmp_path):
    from server.models import FoldRequest, SequenceEntry
    from server.tools import build_input_json

    req = FoldRequest(
        sequences=[SequenceEntry(type="protein", id="A", sequence="MKT")],
    )
    job_dir = tmp_path / "jobs" / "job1"
    job_dir.mkdir(parents=True)
    msa_path = tmp_path / "A.a3m"
    msa_path.write_text(">A\nMKT\n")

    json_path = build_input_json(
        req, job_dir=job_dir, saved_msa_paths={"A": msa_path}
    )
    doc = json.loads(json_path.read_text())
    assert doc["sequences"][0]["msa_path"] == str(msa_path)


def test_build_input_json_with_modifications(tmp_path):
    from server.models import FoldRequest, Modification, SequenceEntry
    from server.tools import build_input_json

    req = FoldRequest(
        sequences=[
            SequenceEntry(
                type="protein",
                id="A",
                sequence="MKT",
                modifications=[Modification(position=1, ccd="SEP")],
            ),
        ],
    )
    job_dir = tmp_path / "jobs" / "job1"
    job_dir.mkdir(parents=True)

    json_path = build_input_json(req, job_dir=job_dir, saved_msa_paths={})
    doc = json.loads(json_path.read_text())
    assert doc["sequences"][0]["modifications"] == [{"position": 1, "ccd": "SEP"}]


# ---- argv assembly ----


def test_fold_argv_basic(tmp_path):
    from server.models import FoldRequest, SequenceEntry
    from server.settings import ESMFold2Settings
    from server.tools import fold_argv

    settings = ESMFold2Settings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path,
        python="/bin/true",
        inference_script="/opt/esmfold2/inference.py",
        model_dir=tmp_path / "weights",
        ccd_path=tmp_path / "weights" / "ccd.pkl",
    )
    req = FoldRequest(
        sequences=[SequenceEntry(type="protein", id="A", sequence="MKT")],
        num_loops=5,
        num_sampling_steps=100,
    )
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    input_json = tmp_path / "input.json"
    input_json.write_text("{}")

    argv = fold_argv(req, job_dir=job_dir, input_json=input_json, settings=settings)
    assert argv[0] == "/bin/true"
    assert argv[1] == "/opt/esmfold2/inference.py"
    assert "--input-json" in argv
    assert "--num-loops" in argv
    assert argv[argv.index("--num-loops") + 1] == "5"
    assert "--num-sampling-steps" in argv
    assert argv[argv.index("--num-sampling-steps") + 1] == "100"


def test_fold_argv_with_seed(tmp_path):
    from server.models import FoldRequest, SequenceEntry
    from server.settings import ESMFold2Settings
    from server.tools import fold_argv

    settings = ESMFold2Settings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path,
        python="/bin/true",
        inference_script="/opt/esmfold2/inference.py",
        model_dir=tmp_path / "weights",
        ccd_path=tmp_path / "weights" / "ccd.pkl",
    )
    req = FoldRequest(
        sequences=[SequenceEntry(type="protein", id="A", sequence="MKT")],
        seed=42,
        noise_scale=1.5,
    )
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    input_json = tmp_path / "input.json"
    input_json.write_text("{}")

    argv = fold_argv(req, job_dir=job_dir, input_json=input_json, settings=settings)
    assert "--seed" in argv
    assert argv[argv.index("--seed") + 1] == "42"
    assert "--noise-scale" in argv
    assert argv[argv.index("--noise-scale") + 1] == "1.5"


def test_fold_argv_omits_none_params(tmp_path):
    from server.models import FoldRequest, SequenceEntry
    from server.settings import ESMFold2Settings
    from server.tools import fold_argv

    settings = ESMFold2Settings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path,
        python="/bin/true",
        inference_script="/opt/esmfold2/inference.py",
        model_dir=tmp_path / "weights",
        ccd_path=tmp_path / "weights" / "ccd.pkl",
    )
    req = FoldRequest(
        sequences=[SequenceEntry(type="protein", id="A", sequence="MKT")],
    )
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    input_json = tmp_path / "input.json"
    input_json.write_text("{}")

    argv = fold_argv(req, job_dir=job_dir, input_json=input_json, settings=settings)
    assert "--seed" not in argv
    assert "--noise-scale" not in argv
    assert "--step-scale" not in argv


# ---- Endpoint smoke (uses stubbed runner) ----


def test_fold_endpoint_smoke(client_with_stub_runner):
    client, captured = client_with_stub_runner
    payload = {
        "sequences": '[{"type":"protein","id":"A","sequence":"MKT"}]',
    }
    r = client.post("/api/fold", data=payload)
    assert r.status_code == 200, r.text
    assert r.json()["status"] in ("pending", "running")
    assert len(captured) == 1
    assert captured[0]["label"] == "fold"
    assert "--input-json" in captured[0]["argv"]


def test_fold_endpoint_with_params(client_with_stub_runner):
    client, captured = client_with_stub_runner
    payload = {
        "sequences": '[{"type":"protein","id":"A","sequence":"MKT"}]',
        "num_loops": "5",
        "num_sampling_steps": "100",
        "seed": "42",
    }
    r = client.post("/api/fold", data=payload)
    assert r.status_code == 200, r.text

    argv = captured[0]["argv"]
    assert argv[argv.index("--num-loops") + 1] == "5"
    assert argv[argv.index("--num-sampling-steps") + 1] == "100"
    assert argv[argv.index("--seed") + 1] == "42"


def test_fold_endpoint_writes_input_json(client_with_stub_runner):
    client, captured = client_with_stub_runner
    payload = {
        "sequences": (
            '[{"type":"protein","id":"A","sequence":"MKT"},'
            '{"type":"ligand","id":"B","smiles":"CCO"}]'
        ),
    }
    r = client.post("/api/fold", data=payload)
    assert r.status_code == 200, r.text

    input_json = captured[0]["job_dir"] / "input" / "input.json"
    doc = json.loads(input_json.read_text())
    assert len(doc["sequences"]) == 2
    assert doc["sequences"][0]["type"] == "protein"
    assert doc["sequences"][1]["type"] == "ligand"
    assert doc["sequences"][1]["smiles"] == "CCO"


def test_fold_endpoint_rejects_empty_sequences(client):
    r = client.post("/api/fold", data={"sequences": "[]"})
    assert r.status_code == 422


# ---- Task endpoint smoke ----


def test_fold_task_endpoint_returns_terminal_status(client):
    """POST /api/tasks/fold blocks until subprocess exits."""
    resp = client.post(
        "/api/tasks/fold",
        data={"sequences": '[{"type":"protein","id":"A","sequence":"MKT"}]'},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in {"completed", "failed"}
    assert body["completed_at"] is not None


def test_fold_task_endpoint_honors_job_id_header(client):
    resp = client.post(
        "/api/tasks/fold",
        data={"sequences": '[{"type":"protein","id":"A","sequence":"MKT"}]'},
        headers={"X-Bioagent-Job-Id": "esmfold-task-001"},
    )
    assert resp.status_code == 200
    assert resp.json()["job_id"] == "esmfold-task-001"


# ---- MSA via zip URI (gateway path) ----


def test_extract_msa_zip_from_file_uri(client, tmp_path):
    """`_extract_msa_zip` resolves a zip URI and lands per-chain a3m keyed by stem.

    This is the gateway path: instead of multipart `msa_files`, a zip of A3M
    files is referenced by URI (oss:// through the gateway; file:// here).
    """
    import importlib
    import zipfile

    server_app = importlib.import_module("server.app")
    zpath = tmp_path / "msa.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("A.a3m", ">A\nMKT\n")
        zf.writestr("B.a3m", ">B\nGGG\n")

    input_dir = tmp_path / "job" / "input"
    saved = server_app._extract_msa_zip(f"file://{zpath}", input_dir)

    assert set(saved) == {"A", "B"}
    assert saved["A"].read_text().startswith(">A")
    assert (input_dir / "msa" / "A.a3m").exists()


def test_fold_task_endpoint_accepts_msa_zip_uri(client, tmp_path):
    """The task endpoint accepts an MSA zip via `msa_zip_uri` (no multipart)."""
    import zipfile

    zpath = tmp_path / "msa.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("A.a3m", ">A\nMKT\n")

    resp = client.post(
        "/api/tasks/fold",
        data={
            "sequences": '[{"type":"protein","id":"A","sequence":"MKT"}]',
            "msa_zip_uri": f"file://{zpath}",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] in {"completed", "failed"}


def test_fold_task_endpoint_rejects_bad_msa_zip_uri(client, tmp_path):
    """A non-zip referenced by msa_zip_uri surfaces 422 (not 500)."""
    bad = tmp_path / "junk.zip"
    bad.write_bytes(b"not a zip")

    resp = client.post(
        "/api/tasks/fold",
        data={
            "sequences": '[{"type":"protein","id":"A","sequence":"MKT"}]',
            "msa_zip_uri": f"file://{bad}",
        },
    )
    assert resp.status_code == 422
