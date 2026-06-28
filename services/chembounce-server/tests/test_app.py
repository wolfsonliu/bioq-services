"""Offline tests for chembounce-server.

Real ChemBounce never runs in offline tests — `CHEMBOUNCE_PYTHON=/bin/true`
stubs the subprocess.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


LOSARTAN = "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMBOUNCE_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("CHEMBOUNCE_ROOT", str(tmp_path / "upstream"))
    monkeypatch.setenv("CHEMBOUNCE_PYTHON", "/bin/true")
    monkeypatch.setenv("CHEMBOUNCE_ENTRYPOINT", str(tmp_path / "chembounce.py"))
    monkeypatch.setenv("CHEMBOUNCE_WEIGHTS_DIR", str(tmp_path / "data"))
    (tmp_path / "upstream").mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Touch 250mw DB files so healthz/detail reports it as present.
    (data_dir / "scaffolds_250mw.txt").write_text("CC\n")
    (data_dir / "scaffold_fingerprints_250mw.npz").write_bytes(b"\x00")

    sys.modules.pop("server.app", None)
    sys.modules.pop("server.settings", None)
    sys.modules.pop("server.adapter", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Health / manifest -----


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "chembounce"


def test_healthz_detail_reports_db_status(client):
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    # 250mw was touched in the fixture → present; full was not → missing
    assert body["database_status"]["250mw"] is True
    assert body["database_status"]["full"] is False
    # weights_loaded is "service usable" → True (250mw default works)
    assert body["weights_loaded"] is True
    assert "scaffolds.txt" in body["weights_missing"]


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "chembounce"


def test_manifest_lists_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/scaffold_hop" in paths


def test_manifest_extras_have_model_info(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["model"]["name"] == "ChemBounce"
    assert "ligand-based" in extras["model"]["task"]
    assert "license_note" in extras["model"]
    assert "scaffold_hop" in extras["tool_outputs"]


def test_manifest_examples_have_curl(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    examples = by_path["/api/scaffold_hop"]["examples"]
    assert len(examples) >= 2
    assert any("input_smiles" in (e.get("curl") or "") for e in examples)


def test_openapi_served(client):
    r = client.get("/openapi.json")
    r.raise_for_status()
    assert "/api/scaffold_hop" in r.json()["paths"]


# ----- Validation errors -----


def test_missing_smiles_returns_422(client):
    # No input_smiles → pydantic + form validation rejects it
    r = client.post("/api/scaffold_hop", data={"frag_max_n": "10"})
    assert r.status_code == 422


def test_tanimoto_out_of_range_returns_422(client):
    r = client.post(
        "/api/scaffold_hop",
        data={"input_smiles": LOSARTAN, "tanimoto_threshold": "1.5"},
    )
    assert r.status_code == 422


def test_invalid_database_returns_422(client):
    r = client.post(
        "/api/scaffold_hop",
        data={"input_smiles": LOSARTAN, "database": "nonexistent"},
    )
    assert r.status_code == 422


def test_smiles_too_long_returns_422(client):
    r = client.post(
        "/api/scaffold_hop",
        data={"input_smiles": "C" * 600},
    )
    assert r.status_code == 422


# ----- Smoke (subprocess stubbed via /bin/true) -----


def test_scaffold_hop_returns_job_with_input_params(client):
    r = client.post(
        "/api/scaffold_hop",
        data={
            "input_smiles": LOSARTAN,
            "frag_max_n": "5",
            "tanimoto_threshold": "0.6",
            "database": "250mw",
            "wo_lipinski": "true",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body["input_params"]["input_smiles"] == LOSARTAN
    assert body["input_params"]["frag_max_n"] == 5
    assert body["input_params"]["tanimoto_threshold"] == 0.6
    assert body["input_params"]["database"] == "250mw"
    assert body["input_params"]["wo_lipinski"] is True


def test_scaffold_hop_default_params(client):
    r = client.post(
        "/api/scaffold_hop",
        data={"input_smiles": LOSARTAN},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["input_params"]["frag_max_n"] == 100  # default
    assert body["input_params"]["tanimoto_threshold"] == 0.5  # default
    assert body["input_params"]["database"] == "250mw"  # default


# ----- Settings -----


def test_settings_defaults():
    from server.settings import ChemBounceSettings

    class _Off(ChemBounceSettings):
        model_config = SettingsConfigDict(
            env_prefix="CHEMBOUNCE_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/chembounce_jobs")
    assert s.root == Path("/opt/chembounce/upstream")
    # Per project convention: weights_dir → NAS even though contents here
    # are scaffold DB, not model weights.
    # See engineering/decisions/2026-06-26-service-weights-externalization.md.
    assert s.weights_dir == Path("/data/models/chembounce/data")
    assert s.fingerprint_250mw == Path(
        "/data/models/chembounce/data/scaffold_fingerprints_250mw.npz"
    )
    assert s.fingerprint_full == Path(
        "/data/models/chembounce/data/scaffold_fingerprints.npz"
    )
    assert s.max_concurrent_jobs == 1


def test_settings_env_override(monkeypatch):
    from server.settings import ChemBounceSettings
    monkeypatch.setenv("CHEMBOUNCE_PYTHON", "/custom/python")
    monkeypatch.setenv("CHEMBOUNCE_WEIGHTS_DIR", "/mnt/scratch/chembounce")
    s = ChemBounceSettings()
    assert s.python == "/custom/python"
    assert s.weights_dir == Path("/mnt/scratch/chembounce")
    # Computed paths track weights_dir.
    assert s.fingerprint_250mw == Path(
        "/mnt/scratch/chembounce/scaffold_fingerprints_250mw.npz"
    )


# ----- tools.argv builder -----


def test_scaffold_hop_argv_has_required_flags(tmp_path):
    from server.models import ScaffoldHopRequest
    from server.settings import ChemBounceSettings
    from server.tools import scaffold_hop_argv

    class _Off(ChemBounceSettings):
        model_config = SettingsConfigDict(
            env_prefix="CHEMBOUNCE_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off(
        python="/bin/python",
        entrypoint="/opt/chembounce.py",
        weights_dir=tmp_path / "data",
    )
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    argv = scaffold_hop_argv(
        ScaffoldHopRequest(
            input_smiles=LOSARTAN,
            frag_max_n=42,
            tanimoto_threshold=0.7,
            database="250mw",
            qed_min=0.5,
            wo_lipinski=True,
        ),
        job_dir=job_dir,
        settings=s,
    )
    assert argv[0] == "/bin/python"
    assert "/opt/chembounce.py" in argv
    assert "-i" in argv and LOSARTAN in argv
    assert "-o" in argv and str(job_dir / "output") in argv
    assert "-n" in argv and "42" in argv
    assert "-t" in argv and "0.7" in argv
    assert "--scaffold-db" in argv
    assert str(tmp_path / "data" / "scaffolds_250mw.txt") in argv
    assert "--fingerprint-db" in argv
    assert str(tmp_path / "data" / "scaffold_fingerprints_250mw.npz") in argv
    assert "--qed_min" in argv and "0.5" in argv
    assert "--wo_lipinski" in argv


def test_scaffold_hop_argv_full_db(tmp_path):
    from server.models import ScaffoldHopRequest
    from server.settings import ChemBounceSettings
    from server.tools import scaffold_hop_argv

    class _Off(ChemBounceSettings):
        model_config = SettingsConfigDict(
            env_prefix="CHEMBOUNCE_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off(weights_dir=tmp_path / "data")
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    argv = scaffold_hop_argv(
        ScaffoldHopRequest(input_smiles=LOSARTAN, database="full"),
        job_dir=job_dir,
        settings=s,
    )
    assert str(tmp_path / "data" / "scaffolds.txt") in argv
    assert str(tmp_path / "data" / "scaffold_fingerprints.npz") in argv


def test_scaffold_hop_argv_optional_threshold_only_when_set(tmp_path):
    from server.models import ScaffoldHopRequest
    from server.settings import ChemBounceSettings
    from server.tools import scaffold_hop_argv

    class _Off(ChemBounceSettings):
        model_config = SettingsConfigDict(
            env_prefix="CHEMBOUNCE_TEST_",
            env_file=None,
            extra="ignore",
        )

    s = _Off(weights_dir=tmp_path / "data")
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    # No threshold set → flag must NOT appear
    argv = scaffold_hop_argv(
        ScaffoldHopRequest(input_smiles=LOSARTAN),
        job_dir=job_dir, settings=s,
    )
    assert "--qed_min" not in argv
    assert "--mw_max" not in argv
    assert "--wo_lipinski" not in argv

    # With wo_lipinski=True → flag appears
    argv = scaffold_hop_argv(
        ScaffoldHopRequest(input_smiles=LOSARTAN, wo_lipinski=True),
        job_dir=job_dir, settings=s,
    )
    assert "--wo_lipinski" in argv
