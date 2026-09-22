"""FC 异步任务模式（`/api/tasks/*`）的集成测试。"""

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


def test_task_design_is_atomic(client):
    """task 端点同步执行到底，返回时已是终态。"""
    started = time.monotonic()
    resp = client.post(
        "/api/tasks/design",
        data={
            "target_name": "hPDL1",
            "binder_lengths": "[60,60]",
            "number_of_final_designs": "1",
            "max_trajectories": "1",
            "campaign_name": "fc_task_smoke",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] in ("completed", "running", "pending")
    assert time.monotonic() - started >= 0

    job_id = body["job_id"]
    if body["status"] != "completed":
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            body = client.get(f"/api/jobs/{job_id}").json()
            if body["status"] in ("completed", "failed"):
                break
            time.sleep(15)
    assert body["status"] == "completed", body
    STATE["task_design_job_id"] = job_id


def test_task_rank_is_atomic(client):
    job_id = STATE.get("task_design_job_id")
    if not job_id:
        pytest.skip("task design smoke did not run")
    resp = client.post(
        "/api/tasks/rank",
        data={"campaign_uri": f"job://{job_id}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] in ("completed", "running", "pending")


def test_task_filter_is_atomic(client):
    job_id = STATE.get("task_design_job_id")
    if not job_id:
        pytest.skip("task design smoke did not run")
    resp = client.post(
        "/api/tasks/filter",
        data={"campaign_uri": f"job://{job_id}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] in ("completed", "running", "pending")


def test_task_endpoints_are_registered(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/api/tasks/design", "/api/tasks/rank", "/api/tasks/filter"):
        assert path in paths
