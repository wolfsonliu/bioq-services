"""Offline tests for megalodon-server.

Real Megalodon sampling never runs offline — the subprocess is stubbed via
MEGALODON_PYTHON=/bin/true so no GPU / weights needed. Config synthesis
(configs.build_config) does run against the vendored upstream YAML.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict

SERVICE_DIR = Path(__file__).resolve().parent.parent
CONF_DIR = SERVICE_DIR / "upstream" / "scripts" / "conf"

# All 6 variants share two datasets; core stats files per dataset.
CORE_STATS = (
    "train_atom_types_h.npy",
    "train_bond_types_h.npy",
    "train_charges_prior_h.npy",
    "train_n_h.pickle",
    "train_smiles.pickle",
)


def _stage_dataset(weights_dir: Path, dataset: str, ckpts: list[str], *, stats=True) -> None:
    ck = weights_dir / "ckpts" / dataset
    ck.mkdir(parents=True, exist_ok=True)
    for c in ckpts:
        (ck / c).write_bytes(b"\x00")
    if stats:
        sd = weights_dir / "stats" / dataset
        sd.mkdir(parents=True, exist_ok=True)
        for f in CORE_STATS:
            (sd / f).write_bytes(b"\x00")
        # drugs_fm needs the no-_h alias.
        if dataset == "drugs":
            (sd / "train_charges_prior.npy").write_bytes(b"\x00")


def _base_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MEGALODON_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("MEGALODON_ROOT", str(tmp_path / "megalodon"))
    monkeypatch.setenv("MEGALODON_PYTHON", "/bin/true")
    monkeypatch.setenv("MEGALODON_INFERENCE_SCRIPT", str(tmp_path / "inference.py"))
    monkeypatch.setenv("MEGALODON_WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("MEGALODON_CONF_DIR", str(CONF_DIR))
    (tmp_path / "megalodon").mkdir(parents=True, exist_ok=True)


@pytest.fixture
def client(tmp_path, monkeypatch):
    _base_env(monkeypatch, tmp_path)
    wd = tmp_path / "weights"
    # drugs fully staged; qm9 ckpts present but stats missing → not ready.
    _stage_dataset(wd, "drugs",
                   ["megalodon_large_diffusion.ckpt", "megalodon_fm.ckpt",
                    "megalodon_small_diffusion.ckpt"], stats=True)
    _stage_dataset(wd, "qm9",
                   ["megalodon_diffusion.ckpt", "megalodon_fm.ckpt",
                    "megalodon_small_diffusion.ckpt"], stats=False)

    for m in ("server.app", "server.settings", "server.adapter"):
        sys.modules.pop(m, None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Health / manifest -----


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "megalodon"
    assert "version" in body


def test_healthz_detail_reports_variants(client):
    body = client.get("/healthz/detail").json()
    assert body["status"] == "ok"
    assert body["service"] == "megalodon"
    # drugs staged → ready; qm9 stats missing → not ready.
    assert body["models"]["drugs_diffusion"]["ready"] is True
    assert body["models"]["qm9_diffusion"]["ready"] is False
    assert body["models"]["qm9_diffusion"]["stats"] is False
    # at least one ready → weights_loaded True
    assert body["weights_loaded"] is True


def test_api_models_lists_registry(client):
    body = client.get("/api/models").json()
    names = {m["name"]: m for m in body["models"]}
    assert set(names) == {
        "drugs_diffusion", "drugs_fm", "drugs_quick",
        "qm9_diffusion", "qm9_fm", "qm9_quick",
    }
    assert names["drugs_diffusion"]["dataset"] == "drugs"
    assert names["drugs_diffusion"]["ready"] is True
    assert names["qm9_fm"]["ready"] is False


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "megalodon"


def test_manifest_lists_endpoints(client):
    paths = {e["path"] for e in client.get("/api/manifest").json()["endpoints"]}
    assert "/api/generate" in paths


def test_manifest_extras_have_model_info(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert extras["model"]["name"] == "Megalodon"
    assert "generate" in extras["tool_outputs"]


def test_manifest_extras_have_endpoint_examples(client):
    body = client.get("/api/manifest").json()
    by_path = {e["path"]: e for e in body["endpoints"]}
    examples = by_path["/api/generate"]["examples"]
    assert len(examples) >= 2
    assert any("model_name" in (e.get("curl") or "") for e in examples)


def test_openapi_served(client):
    r = client.get("/openapi.json")
    r.raise_for_status()
    paths = r.json()["paths"]
    assert "/api/generate" in paths
    assert "/api/tasks/generate" in paths
    assert "/api/models" in paths


# ----- Validation errors -----


def test_generate_invalid_model_returns_422(client):
    assert client.post("/api/generate", data={"model_name": "not_a_model"}).status_code == 422


def test_generate_n_molecules_too_low_returns_422(client):
    assert client.post("/api/generate", data={"n_molecules": "0"}).status_code == 422


def test_generate_n_molecules_too_high_returns_422(client):
    assert client.post("/api/generate", data={"n_molecules": "20000"}).status_code == 422


def test_generate_timesteps_out_of_range_returns_422(client):
    assert client.post("/api/generate", data={"timesteps": "5"}).status_code == 422


def test_generate_n_atoms_out_of_range_returns_422(client):
    assert client.post("/api/generate", data={"n_atoms_per_mol": "2"}).status_code == 422


# ----- Smoke (subprocess stubbed via /bin/true) -----


def test_generate_returns_job_with_input_params(client):
    r = client.post("/api/generate", data={
        "model_name": "drugs_diffusion",
        "n_molecules": "50",
        "timesteps": "200",
        "n_atoms_per_mol": "25",
        "seed": "42",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body["input_params"]["model_name"] == "drugs_diffusion"
    assert body["input_params"]["n_molecules"] == 50
    assert body["input_params"]["timesteps"] == 200
    assert body["input_params"]["n_atoms_per_mol"] == 25
    assert body["input_params"]["seed"] == 42


def test_generate_defaults_survive(client):
    r = client.post("/api/generate", data={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input_params"]["model_name"] == "drugs_diffusion"
    assert body["input_params"]["n_molecules"] == 100
    assert body["input_params"]["timesteps"] == 500
    assert body["input_params"]["n_atoms_per_mol"] is None


def test_generate_writes_job_config(client, tmp_path):
    """generate_argv (via submit) must synthesize config.yaml with rewritten
    statistics paths pointing at the flat NAS stats dir."""
    r = client.post("/api/generate", data={"model_name": "drugs_diffusion"})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    cfg_path = tmp_path / "jobs" / job_id / "config.yaml"
    assert cfg_path.is_file()
    cfg = yaml.safe_load(cfg_path.read_text())
    stats = str(tmp_path / "weights" / "stats" / "drugs")
    # custom_prior repointed at flat stats dir (basenames preserved).
    priors = [v.get("custom_prior") for v in cfg["interpolant"]["variables"]
              if v.get("custom_prior")]
    assert priors and all(p.startswith(stats) for p in priors)
    assert any(p.endswith("train_charges_prior_h.npy") for p in priors)
    # node_distribution repointed too.
    assert cfg["sample"]["node_distribution"].startswith(stats)
    assert cfg["wandb_params"]["mode"] == "disabled"


# ----- Settings -----


def test_settings_defaults():
    from server.settings import MegalodonSettings

    class _Off(MegalodonSettings):
        model_config = SettingsConfigDict(
            env_prefix="MEGALODON_TEST_", env_file=None, extra="ignore",
        )

    s = _Off()
    assert s.jobs_base_dir == Path("/data/megalodon_jobs")
    assert s.root == Path("/opt/megalodon")
    assert s.weights_dir == Path("/data/models/megalodon")
    assert s.max_concurrent_jobs == 1
    assert s.default_model == "drugs_diffusion"


def test_settings_ckpt_and_stats_paths():
    from server.settings import MegalodonSettings

    class _Off(MegalodonSettings):
        model_config = SettingsConfigDict(
            env_prefix="MEGALODON_TEST_", env_file=None, extra="ignore",
        )

    s = _Off(weights_dir=Path("/w"))
    assert s.ckpt_path("drugs", "megalodon_large_diffusion.ckpt") == Path(
        "/w/ckpts/drugs/megalodon_large_diffusion.ckpt")
    assert s.stats_dir("qm9") == Path("/w/stats/qm9")
