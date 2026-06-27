"""Offline tests for boltz-server (no real `boltz` binary needed).

The endpoint handlers' subprocess call is replaced with a synchronous stub via
`monkeypatch` so we can exercise the YAML-building + argv-assembly logic
without a GPU or the actual Boltz package installed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


# ---- Shared fixtures ----

@pytest.fixture
def client(tmp_path, monkeypatch):
    """Import `server.app` fresh against patched env vars + a stub boltz binary.

    Endpoint signatures rely on module-level globals (settings, adapter, app)
    constructed at import time, so we re-import after `monkeypatch.setenv`.
    """
    monkeypatch.setenv("BOLTZ_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("BOLTZ_ROOT", str(tmp_path / "boltz"))
    monkeypatch.setenv("BOLTZ_BINARY", "/bin/true")  # never actually executed
    monkeypatch.setenv("BOLTZ_CACHE_DIR", str(tmp_path / "cache"))
    (tmp_path / "boltz").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


@pytest.fixture
def captured_argv(monkeypatch):
    """Capture argv from `runner.submit` instead of running it.

    Returns a list[list[str]] that grows on each endpoint call so the tests
    can inspect what argv would have been executed.
    """
    captured: list[dict] = []

    def _fake_submit(build_argv, label, **kwargs):
        from bioagent_service import JobInfo, JobStatus

        job_id = f"stub-{label}-{len(captured)}"
        # Run build_argv in a temp dir so its side effects (YAML write) are
        # capture-able; the test fixture provides the dir.
        import tempfile
        job_dir = Path(tempfile.mkdtemp(prefix="boltz-test-"))
        argv = build_argv(job_id, job_dir)
        captured.append({"job_id": job_id, "label": label, "argv": argv, "job_dir": job_dir})
        return JobInfo(job_id=job_id, status=JobStatus.PENDING)

    return captured, _fake_submit


@pytest.fixture
def client_with_stub_runner(client, captured_argv):
    """Variant of `client` whose runner.submit is replaced with a capturing stub."""
    captured, fake_submit = captured_argv
    client.app.state.runner.submit = fake_submit
    return client, captured


# ---- Health / manifest ----

def test_health(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "boltz"
    assert "version" in health


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "boltz"


def test_manifest_lists_both_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/predict_structure" in paths
    assert "/api/predict_affinity" in paths


def test_manifest_model_is_boltz2(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["model"]["name"] == "boltz2"
    assert extras["model"]["supports_affinity"] is True


def test_manifest_lists_msa_modes(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert set(extras["msa_modes"].keys()) == {"auto", "provided", "empty"}


def test_manifest_not_in_scope_mentions_boltz1(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert "boltz-1" in extras["not_in_scope_v0_0_1"].lower()


def test_endpoint_examples_present(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    assert by_path["/api/predict_structure"]["examples"]
    assert by_path["/api/predict_affinity"]["examples"]


# ---- Settings ----

def test_settings_defaults():
    from server.settings import BoltzSettings

    class _Off(BoltzSettings):
        model_config = SettingsConfigDict(
            env_prefix="BOLTZ_TEST_", env_file=None, extra="ignore"
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/boltz_jobs")
    assert s.root == Path("/opt/boltz")
    assert s.binary == "/opt/boltz/.venv/bin/boltz"
    # Weights externalized to NAS — default points at the FC mount path.
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    assert s.cache_dir == Path("/data/models/boltz")
    assert s.max_concurrent_jobs == 1
    assert s.oss_region == "cn-hangzhou"


def test_settings_env_override(monkeypatch):
    from server.settings import BoltzSettings
    monkeypatch.setenv("BOLTZ_BINARY", "/custom/boltz")
    monkeypatch.setenv("BOLTZ_CACHE_DIR", "/nas/boltz_cache")
    s = BoltzSettings()
    assert s.binary == "/custom/boltz"
    assert s.cache_dir == Path("/nas/boltz_cache")


# ---- Model validation ----

def test_sequence_entry_ligand_requires_smiles_xor_ccd():
    from server.models import SequenceEntry

    with pytest.raises(ValueError, match="exactly one of"):
        SequenceEntry(type="ligand", id="B", smiles="CCO", ccd="ATP")
    with pytest.raises(ValueError, match="exactly one of"):
        SequenceEntry(type="ligand", id="B")
    SequenceEntry(type="ligand", id="B", smiles="CCO")  # OK
    SequenceEntry(type="ligand", id="B", ccd="ATP")  # OK


def test_sequence_entry_protein_requires_sequence():
    from server.models import SequenceEntry

    with pytest.raises(ValueError, match="requires `sequence`"):
        SequenceEntry(type="protein", id="A")
    SequenceEntry(type="protein", id="A", sequence="MKT")  # OK


def test_template_entry_exactly_one_source():
    from server.models import TemplateEntry

    with pytest.raises(ValueError, match="exactly one"):
        TemplateEntry(cif_uri="x.cif", pdb_uri="y.pdb")
    with pytest.raises(ValueError, match="exactly one"):
        TemplateEntry()
    TemplateEntry(cif_uri="x.cif")  # OK


def test_request_raw_yaml_xor_sequences():
    from server.models import PredictStructureRequest, SequenceEntry

    with pytest.raises(ValueError, match="mutually exclusive"):
        PredictStructureRequest(
            sequences=[SequenceEntry(type="protein", id="A", sequence="MKT")],
            raw_yaml="version: 1\nsequences: []\n",
        )
    with pytest.raises(ValueError, match="must supply"):
        PredictStructureRequest()


def test_msa_provided_requires_msa_uri():
    from server.models import PredictStructureRequest, SequenceEntry

    with pytest.raises(ValueError, match="provided.*requires.*msa_uri"):
        PredictStructureRequest(
            msa_mode="provided",
            sequences=[SequenceEntry(type="protein", id="A", sequence="MKT")],
        )
    PredictStructureRequest(
        msa_mode="provided",
        sequences=[
            SequenceEntry(type="protein", id="A", sequence="MKT", msa_uri="empty")
        ],
    )


def test_affinity_binder_must_be_ligand():
    from server.models import PredictAffinityRequest, SequenceEntry

    with pytest.raises(ValueError, match="must be a ligand"):
        PredictAffinityRequest(
            binder_id="A",
            sequences=[
                SequenceEntry(type="protein", id="A", sequence="MKT"),
                SequenceEntry(type="ligand", id="B", smiles="CCO"),
            ],
        )


def test_affinity_binder_must_exist():
    from server.models import PredictAffinityRequest, SequenceEntry

    with pytest.raises(ValueError, match="not found"):
        PredictAffinityRequest(
            binder_id="Z",
            sequences=[
                SequenceEntry(type="protein", id="A", sequence="MKT"),
                SequenceEntry(type="ligand", id="B", smiles="CCO"),
            ],
        )


def test_affinity_binder_accepts_list_id():
    from server.models import PredictAffinityRequest, SequenceEntry

    req = PredictAffinityRequest(
        binder_id="C",
        sequences=[
            SequenceEntry(type="protein", id="A", sequence="MKT"),
            SequenceEntry(type="ligand", id=["B", "C"], smiles="CCO"),
        ],
    )
    assert req.binder_id == "C"


# ---- YAML construction ----

def test_build_yaml_structure_only(tmp_path):
    from server.models import PredictStructureRequest, SequenceEntry
    from server.settings import BoltzSettings
    from server.tools import build_yaml

    settings = BoltzSettings(
        jobs_base_dir=tmp_path / "jobs", root=tmp_path, cache_dir=tmp_path / "cache"
    )
    req = PredictStructureRequest(
        msa_mode="empty",
        sequences=[
            SequenceEntry(type="protein", id="A", sequence="MKT", msa_uri="empty"),
        ],
    )
    job_dir = tmp_path / "jobs" / "job1"
    job_dir.mkdir(parents=True)

    yaml_path = build_yaml(
        req, job_dir=job_dir, settings=settings,
        saved_msa_paths={}, saved_template_paths={},
    )
    doc = yaml.safe_load(yaml_path.read_text())
    assert doc["version"] == 1
    assert doc["sequences"][0]["protein"]["id"] == "A"
    assert doc["sequences"][0]["protein"]["msa"] == "empty"
    assert "properties" not in doc


def test_build_yaml_affinity_adds_properties(tmp_path):
    from server.models import PredictAffinityRequest, SequenceEntry
    from server.settings import BoltzSettings
    from server.tools import build_yaml

    settings = BoltzSettings(
        jobs_base_dir=tmp_path / "jobs", root=tmp_path, cache_dir=tmp_path / "cache"
    )
    req = PredictAffinityRequest(
        binder_id="B",
        msa_mode="empty",
        sequences=[
            SequenceEntry(type="protein", id="A", sequence="MKT", msa_uri="empty"),
            SequenceEntry(type="ligand", id="B", smiles="CCO"),
        ],
    )
    job_dir = tmp_path / "jobs" / "job1"
    job_dir.mkdir(parents=True)

    yaml_path = build_yaml(
        req, job_dir=job_dir, settings=settings,
        saved_msa_paths={}, saved_template_paths={},
    )
    doc = yaml.safe_load(yaml_path.read_text())
    assert doc["properties"][0]["affinity"]["binder"] == "B"


def test_build_yaml_with_provided_msa(tmp_path):
    from server.models import PredictStructureRequest, SequenceEntry
    from server.settings import BoltzSettings
    from server.tools import build_yaml

    settings = BoltzSettings(
        jobs_base_dir=tmp_path / "jobs", root=tmp_path, cache_dir=tmp_path / "cache"
    )
    job_dir = tmp_path / "jobs" / "job1"
    job_dir.mkdir(parents=True)
    msa_dir = job_dir / "input" / "msa"
    msa_dir.mkdir(parents=True)
    a3m_path = msa_dir / "A.a3m"
    a3m_path.write_text(">A\nMKT\n")

    req = PredictStructureRequest(
        msa_mode="provided",
        sequences=[
            SequenceEntry(type="protein", id="A", sequence="MKT", msa_uri="upload"),
        ],
    )
    yaml_path = build_yaml(
        req, job_dir=job_dir, settings=settings,
        saved_msa_paths={"A": a3m_path}, saved_template_paths={},
    )
    doc = yaml.safe_load(yaml_path.read_text())
    assert doc["sequences"][0]["protein"]["msa"] == str(a3m_path)


def test_build_yaml_raw_yaml_passthrough(tmp_path):
    from server.models import PredictStructureRequest
    from server.settings import BoltzSettings
    from server.tools import build_yaml

    settings = BoltzSettings(
        jobs_base_dir=tmp_path / "jobs", root=tmp_path, cache_dir=tmp_path / "cache"
    )
    raw = "version: 1\nsequences:\n  - protein:\n      id: X\n      sequence: MKT\n"
    req = PredictStructureRequest(raw_yaml=raw)
    job_dir = tmp_path / "jobs" / "job1"
    job_dir.mkdir(parents=True)

    yaml_path = build_yaml(
        req, job_dir=job_dir, settings=settings,
        saved_msa_paths={}, saved_template_paths={},
    )
    assert yaml_path.read_text() == raw


def test_validate_raw_yaml_rejects_non_dict():
    from fastapi import HTTPException
    from server.tools import validate_raw_yaml

    with pytest.raises(HTTPException) as exc:
        validate_raw_yaml("- 1\n- 2\n")
    assert exc.value.status_code == 422


def test_validate_raw_yaml_rejects_missing_sequences():
    from fastapi import HTTPException
    from server.tools import validate_raw_yaml

    with pytest.raises(HTTPException) as exc:
        validate_raw_yaml("version: 1\n")
    assert exc.value.status_code == 422


# ---- argv assembly ----

def test_argv_hardcodes_boltz2(tmp_path):
    from server.models import PredictStructureRequest, SequenceEntry
    from server.settings import BoltzSettings
    from server.tools import predict_argv

    settings = BoltzSettings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path,
        binary="/bin/true",
        cache_dir=tmp_path / "cache",
    )
    req = PredictStructureRequest(
        msa_mode="empty",
        sequences=[
            SequenceEntry(type="protein", id="A", sequence="MKT", msa_uri="empty"),
        ],
    )
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    yaml_path = tmp_path / "input.yaml"
    yaml_path.write_text("version: 1\n")

    argv = predict_argv(req, job_dir=job_dir, yaml_path=yaml_path, settings=settings)
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "boltz2"
    assert "--accelerator" in argv
    assert argv[argv.index("--accelerator") + 1] == "gpu"


def test_argv_msa_auto_adds_use_msa_server(tmp_path):
    from server.models import PredictStructureRequest, SequenceEntry
    from server.settings import BoltzSettings
    from server.tools import predict_argv

    settings = BoltzSettings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path,
        binary="/bin/true",
        cache_dir=tmp_path / "cache",
    )
    req = PredictStructureRequest(
        msa_mode="auto",
        sequences=[SequenceEntry(type="protein", id="A", sequence="MKT")],
    )
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    yaml_path = tmp_path / "input.yaml"
    yaml_path.write_text("version: 1\n")

    argv = predict_argv(req, job_dir=job_dir, yaml_path=yaml_path, settings=settings)
    assert "--use_msa_server" in argv
    assert "--msa_pairing_strategy" in argv


def test_argv_affinity_adds_affinity_flags(tmp_path):
    from server.models import PredictAffinityRequest, SequenceEntry
    from server.settings import BoltzSettings
    from server.tools import predict_argv

    settings = BoltzSettings(
        jobs_base_dir=tmp_path / "jobs",
        root=tmp_path,
        binary="/bin/true",
        cache_dir=tmp_path / "cache",
    )
    req = PredictAffinityRequest(
        binder_id="B",
        msa_mode="empty",
        sequences=[
            SequenceEntry(type="protein", id="A", sequence="MKT", msa_uri="empty"),
            SequenceEntry(type="ligand", id="B", smiles="CCO"),
        ],
        affinity_mw_correction=True,
        sampling_steps_affinity=150,
        diffusion_samples_affinity=3,
    )
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    yaml_path = tmp_path / "input.yaml"
    yaml_path.write_text("version: 1\n")

    argv = predict_argv(req, job_dir=job_dir, yaml_path=yaml_path, settings=settings)
    assert "--affinity_mw_correction" in argv
    assert "--sampling_steps_affinity" in argv
    assert argv[argv.index("--sampling_steps_affinity") + 1] == "150"
    assert "--diffusion_samples_affinity" in argv
    assert argv[argv.index("--diffusion_samples_affinity") + 1] == "3"


# ---- Endpoint smoke (uses stubbed runner) ----

def test_predict_structure_endpoint_smoke(client_with_stub_runner):
    client, captured = client_with_stub_runner
    payload = {
        "name": "smoke",
        "msa_mode": "empty",
        "sequences": '[{"type":"protein","id":"A","sequence":"MKT","msa_uri":"empty"}]',
    }
    r = client.post("/api/predict_structure", data=payload)
    assert r.status_code == 200, r.text
    assert r.json()["status"] in ("pending", "running")
    assert len(captured) == 1
    assert captured[0]["label"] == "predict_structure"
    assert "--model" in captured[0]["argv"]


def test_predict_affinity_endpoint_smoke(client_with_stub_runner):
    client, captured = client_with_stub_runner
    payload = {
        "name": "smoke",
        "binder_id": "B",
        "msa_mode": "empty",
        "sequences": (
            '[{"type":"protein","id":"A","sequence":"MKT","msa_uri":"empty"},'
            '{"type":"ligand","id":"B","smiles":"CCO"}]'
        ),
    }
    r = client.post("/api/predict_affinity", data=payload)
    assert r.status_code == 200, r.text
    assert len(captured) == 1
    assert captured[0]["label"] == "predict_affinity"

    # Confirm YAML rendered the affinity property block
    yaml_path = captured[0]["job_dir"] / "input" / "input.yaml"
    doc = yaml.safe_load(yaml_path.read_text())
    assert doc["properties"][0]["affinity"]["binder"] == "B"


def test_predict_affinity_rejects_protein_binder(client):
    payload = {
        "name": "smoke",
        "binder_id": "A",  # points to protein, not ligand
        "msa_mode": "empty",
        "sequences": (
            '[{"type":"protein","id":"A","sequence":"MKT","msa_uri":"empty"},'
            '{"type":"ligand","id":"B","smiles":"CCO"}]'
        ),
    }
    r = client.post("/api/predict_affinity", data=payload)
    assert r.status_code == 422


def test_predict_structure_rejects_empty_input(client):
    r = client.post("/api/predict_structure", data={"name": "smoke", "msa_mode": "empty"})
    assert r.status_code == 422


# ---- Task endpoint smoke (blocking; uses /bin/true as boltz binary) ----

def test_predict_structure_task_endpoint_returns_terminal_status(client):
    """POST /api/tasks/predict_structure blocks until subprocess exits."""
    payload = {
        "name": "smoke",
        "msa_mode": "empty",
        "sequences": '[{"type":"protein","id":"A","sequence":"MKT","msa_uri":"empty"}]',
    }
    resp = client.post("/api/tasks/predict_structure", data=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in {"completed", "failed"}
    assert body["completed_at"] is not None


def test_predict_affinity_task_endpoint_returns_terminal_status(client):
    """POST /api/tasks/predict_affinity blocks until subprocess exits."""
    payload = {
        "name": "smoke",
        "binder_id": "B",
        "msa_mode": "empty",
        "sequences": (
            '[{"type":"protein","id":"A","sequence":"MKT","msa_uri":"empty"},'
            '{"type":"ligand","id":"B","smiles":"CCO"}]'
        ),
    }
    resp = client.post("/api/tasks/predict_affinity", data=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in {"completed", "failed"}
    assert body["completed_at"] is not None
