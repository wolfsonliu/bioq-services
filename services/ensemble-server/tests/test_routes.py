"""End-to-end route tests using FastAPI TestClient.

Replaces the registered adapters with fake adapters (no real FC, no real
GPU) via app.state monkeypatching in fixtures.  The Orchestrator itself is
exercised for real.
"""

from __future__ import annotations

import hashlib
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from pipelines.framework.dispatcher import DispatchHandle, TaskStatus

from server.adapters.base import MethodAdapter
from server.adapters.registry import MethodRegistry
from server.folding.aggregator import aggregate_folding
from server.folding.schemas import FoldingMethodResult, StructureFile
from server.orchestrator.orchestrator import Orchestrator
from server.orchestrator.store import EnsembleJobStore
from server.routes import folding as folding_routes
from server.routes import jobs as jobs_routes
from server.routes import manifest as manifest_routes
from server.settings import APIKeyConfig, EnsembleSettings
from server.task_kind import TaskKind


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeOptions(BaseModel):
    pass


class _FakeFoldingAdapter(MethodAdapter):
    task_kind = TaskKind.FOLDING
    method_options_schema = _FakeOptions

    def __init__(self, name: str, fc_mock: MagicMock) -> None:
        super().__init__(fc_mock)
        self.name = name

    def build_request(self, input, options):
        return "/api/tasks/fake", {"name": "x"}, {}

    def normalize_output(self, sub_task_id, downloaded_dir):
        return FoldingMethodResult(
            method=self.name,
            status="completed",
            structures=[StructureFile(
                rank=0, format="cif",
                url=f"/v1/jobs/{sub_task_id.split('__')[0]}/structures/{self.name}/fake.cif",
                plddt=0.85,
            )],
            confidence={"plddt": 0.85},
        )


def _make_fc_mock(function_name: str) -> MagicMock:
    m = MagicMock()
    m.function = function_name
    m.submit.return_value = DispatchHandle(
        backend="fc", task_id="<set-per-call>",
        backend_ref={"invocation_id": "fake-inv-id", "function": function_name},
    )
    return m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

API_SECRET = "test_secret_42"
API_KEY_HEADER = {"X-API-Key": API_SECRET}


@pytest.fixture
def app(tmp_path) -> FastAPI:
    """Build a fresh FastAPI app with fake adapters + a TestClient-compatible state."""
    app = FastAPI(title="Ensemble Test", version="test")

    # Build fresh registry, two fake adapters
    registry = MethodRegistry()
    fc_mocks: dict[str, MagicMock] = {}
    for name in ("fake_a", "fake_b"):
        fc_mock = _make_fc_mock(f"{name}-server")
        registry.register(_FakeFoldingAdapter(name, fc_mock))
        fc_mocks[name] = fc_mock

    # Settings with one API key
    settings = EnsembleSettings(
        jobs_base_dir=tmp_path / "jobs",
        api_keys=[APIKeyConfig(
            key_id="ek_test",
            secret_hash=hashlib.sha256(API_SECRET.encode()).hexdigest(),
            customer_id="customer_a",
        )],
    )
    settings.jobs_base_dir.mkdir(parents=True, exist_ok=True)

    # Orchestrator
    store = EnsembleJobStore(settings.jobs_base_dir)
    orchestrator = Orchestrator(
        registry=registry, store=store,
        aggregators={TaskKind.FOLDING: aggregate_folding},
    )

    app.state.settings = settings
    app.state.registry = registry
    app.state.store = store
    app.state.orchestrator = orchestrator
    app.state.fc_mocks = fc_mocks  # for tests to manipulate

    app.include_router(manifest_routes.router)
    app.include_router(folding_routes.router)
    app.include_router(jobs_routes.router)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# healthz / methods / manifest
# ---------------------------------------------------------------------------

def test_healthz_returns_ok(client: TestClient):
    r = client.get("/v1/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "ensemble"


def test_methods_lists_registered_adapters(client: TestClient):
    r = client.get("/v1/methods?task_kind=folding")
    assert r.status_code == 200
    names = {m["name"] for m in r.json()["methods"]}
    assert names == {"fake_a", "fake_b"}


def test_methods_rejects_unknown_task_kind(client: TestClient):
    r = client.get("/v1/methods?task_kind=banana")
    assert r.status_code == 422


def test_manifest_lists_folding(client: TestClient):
    r = client.get("/v1/manifest")
    assert r.status_code == 200
    body = r.json()
    assert "folding" in body["methods"]
    assert set(body["methods"]["folding"]) == {"fake_a", "fake_b"}


# ---------------------------------------------------------------------------
# Auth on /v1/folding/ensemble
# ---------------------------------------------------------------------------

def test_folding_ensemble_requires_api_key(client: TestClient):
    r = client.post(
        "/v1/folding/ensemble",
        json={"input": {"sequences": [{"id": "A", "sequence": "MKQH"}]}},
    )
    assert r.status_code == 401 or r.status_code == 422


def test_folding_ensemble_rejects_bad_api_key(client: TestClient):
    r = client.post(
        "/v1/folding/ensemble",
        json={"input": {"sequences": [{"id": "A", "sequence": "MKQH"}]}},
        headers={"X-API-Key": "wrong"},
    )
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Submit + poll happy path
# ---------------------------------------------------------------------------

def test_folding_ensemble_submit_returns_task_id(client: TestClient):
    r = client.post(
        "/v1/folding/ensemble",
        json={
            "input": {"sequences": [{"id": "A", "sequence": "MKQH"}], "msa_mode": "empty"},
            "methods": ["fake_a"],
        },
        headers=API_KEY_HEADER,
    )
    assert r.status_code == 202
    body = r.json()
    assert body["task_id"].startswith("ens_fold_")
    assert body["requested_methods"] == ["fake_a"]


def test_folding_ensemble_rejects_unknown_method(client: TestClient):
    r = client.post(
        "/v1/folding/ensemble",
        json={
            "input": {"sequences": [{"id": "A", "sequence": "MKQH"}]},
            "methods": ["does_not_exist"],
        },
        headers=API_KEY_HEADER,
    )
    assert r.status_code == 422


def test_get_job_returns_running_then_completed(client: TestClient, app, tmp_path):
    # Submit
    r = client.post(
        "/v1/folding/ensemble",
        json={
            "input": {"sequences": [{"id": "A", "sequence": "MKQH"}], "msa_mode": "empty"},
            "methods": ["fake_a"],
        },
        headers=API_KEY_HEADER,
    )
    task_id = r.json()["task_id"]

    # First GET: still running (mock get_status returns RUNNING by default)
    app.state.fc_mocks["fake_a"].get_status.return_value = TaskStatus.RUNNING
    r1 = client.get(f"/v1/jobs/{task_id}", headers=API_KEY_HEADER)
    assert r1.status_code == 200
    assert r1.json()["completed_at"] is None

    # Now mock SUCCEEDED + fake zip
    fake_zip = tmp_path / "fake.zip"
    with zipfile.ZipFile(fake_zip, "w") as zf:
        zf.writestr("placeholder.cif", "loop_")
    app.state.fc_mocks["fake_a"].get_status.return_value = TaskStatus.SUCCEEDED
    app.state.fc_mocks["fake_a"].fetch_result.return_value = fake_zip

    r2 = client.get(f"/v1/jobs/{task_id}", headers=API_KEY_HEADER)
    assert r2.status_code == 200
    body = r2.json()
    assert body["completed_at"] is not None
    assert body["sub_tasks"]["fake_a"]["status"] == "succeeded"
    assert body["aggregated_output"] is not None


def test_get_job_returns_404_for_other_customer(client: TestClient, app):
    # Submit as customer_a
    r = client.post(
        "/v1/folding/ensemble",
        json={
            "input": {"sequences": [{"id": "A", "sequence": "MKQH"}], "msa_mode": "empty"},
            "methods": ["fake_a"],
        },
        headers=API_KEY_HEADER,
    )
    task_id = r.json()["task_id"]

    # Add a second API key for a different customer
    app.state.settings.api_keys.append(APIKeyConfig(
        key_id="ek_other",
        secret_hash=hashlib.sha256("other_secret".encode()).hexdigest(),
        customer_id="customer_b",
    ))
    r2 = client.get(f"/v1/jobs/{task_id}", headers={"X-API-Key": "other_secret"})
    assert r2.status_code == 404


def test_get_unknown_job_returns_404(client: TestClient):
    r = client.get("/v1/jobs/ens_fold_does_not_exist", headers=API_KEY_HEADER)
    assert r.status_code == 404


def test_download_structure_rejects_path_traversal(client: TestClient, app):
    # Submit and let it complete
    r = client.post(
        "/v1/folding/ensemble",
        json={
            "input": {"sequences": [{"id": "A", "sequence": "MKQH"}], "msa_mode": "empty"},
            "methods": ["fake_a"],
        },
        headers=API_KEY_HEADER,
    )
    task_id = r.json()["task_id"]

    bad = client.get(
        f"/v1/jobs/{task_id}/structures/fake_a/..%2Fetc%2Fpasswd",
        headers=API_KEY_HEADER,
    )
    # FastAPI may normalize the path before route matching; either 400 or 404 is acceptable
    assert bad.status_code in (400, 404)
