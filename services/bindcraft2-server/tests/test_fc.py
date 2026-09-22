"""针对已部署 FC 实例的集成测试（默认 skip；`-m fc` 或 RUN_FC_TESTS=1 才跑）。

⚠️ design 是真实 GPU campaign。smoke 用 shipped target hPDL1 + 1 trailer + 1 个
final design，仍然需要数分钟；不是单元测试。
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest
from bioq_service.fc_testing import fc_url, make_retrying_client

pytestmark = pytest.mark.fc

SERVICE = "bindcraft2-server"
TIMEOUT = 900.0

# 跨测试传递的 job id（模块级 dict，不往 pytest 模块上挂属性）。
STATE: dict[str, str] = {}


@pytest.fixture(scope="module")
def client():
    """会话亲和要求 submit 与之后的每次 poll 都带同一个 header 值——所以 headers
    挂在 client 上，而不是逐个请求传（否则 poll 会被当成新会话，FC 起一堆实例）。"""
    url = fc_url(SERVICE, start=Path(__file__))
    headers = {"bioagent-session-id": f"fc-bindcraft2-{uuid.uuid4().hex[:8]}"}
    with make_retrying_client(
        url, timeout=TIMEOUT, max_retries=10, backoff_s=20.0, headers=headers
    ) as c:
        yield c


def _poll(client, job_id: str, timeout_s: float = 3600.0) -> dict:
    deadline = time.monotonic() + timeout_s
    body: dict = {}
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(15)
    raise AssertionError(f"job {job_id} did not finish within {timeout_s}s: {body}")


def test_healthz(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_healthz_detail_reports_weights_and_gpu(client):
    detail = client.get("/healthz/detail").json()
    assert detail["service"] == "bindcraft2"
    assert detail["weights_loaded"] is True, detail["weights_missing"]
    assert detail["gpu_backend"] == "gpu", detail


def test_manifest_lists_endpoints_with_examples(client):
    body = client.get("/api/manifest").json()
    eps = {e["path"]: e for e in body["endpoints"]}
    for path in (
        "/api/design",
        "/api/tasks/design",
        "/api/rank",
        "/api/tasks/rank",
        "/api/filter",
        "/api/tasks/filter",
    ):
        assert eps[path]["examples"], f"{path} missing examples"


def test_design_shipped_target_smoke(client):
    """一个 trajectory 的最小 campaign（shipped target，保证结构可解析）。"""
    resp = client.post(
        "/api/design",
        data={
            "target_name": "hPDL1",
            "modality": "binder",
            "binder_lengths": "[60,60]",
            "number_of_final_designs": "1",
            "max_trajectories": "1",
            "campaign_name": "fc_smoke",
        },
    )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]

    body = _poll(client, job_id)
    assert body["status"] == "completed", body

    files = client.get(f"/api/jobs/{job_id}/files").json()["files"]
    assert "summary.csv" in files
    assert "campaign_metadata.json" in files
    assert "1_Trajectories/!_Trajectories.csv" in files

    # 记住这个 job 供 rank / filter 接续
    STATE["design_job_id"] = job_id


def test_design_rejects_bad_target_selection(client):
    resp = client.post("/api/design", data={})
    assert resp.status_code == 422
    assert "Exactly one of" in resp.text


def test_rank_on_previous_campaign(client):
    design_job_id = STATE.get("design_job_id")
    if not design_job_id:
        pytest.skip("design smoke did not run")
    resp = client.post(
        "/api/rank",
        data={"campaign_uri": f"job://{design_job_id}", "on": '["i_pTM"]'},
    )
    resp.raise_for_status()
    body = _poll(client, resp.json()["job_id"], timeout_s=900)
    assert body["status"] == "completed", body
    files = client.get(f"/api/jobs/{resp.json()['job_id']}/files").json()["files"]
    assert "ranked_by_i_pTM.csv" in files


def test_filter_on_previous_campaign(client):
    design_job_id = STATE.get("design_job_id")
    if not design_job_id:
        pytest.skip("design smoke did not run")
    resp = client.post(
        "/api/filter",
        data={
            "campaign_uri": f"job://{design_job_id}",
            "where": '["i_pAE=0.45","Interface_Residues>=7"]',
        },
    )
    resp.raise_for_status()
    body = _poll(client, resp.json()["job_id"], timeout_s=900)
    assert body["status"] == "completed", body
    files = client.get(f"/api/jobs/{resp.json()['job_id']}/files").json()["files"]
    assert "filtered.csv" in files


def test_rank_rejects_unknown_campaign(client):
    resp = client.post(
        "/api/rank",
        data={"campaign_uri": "job://definitely-not-a-job"},
    )
    assert resp.status_code == 404
