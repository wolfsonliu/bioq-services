"""离线 HTTP 端到端：真实 server.app + 真实 runner，子进程换成 stub。"""

from __future__ import annotations

import importlib
import time

from fastapi.testclient import TestClient


def _client(settings, monkeypatch) -> TestClient:
    # server.app 在 import 期用自己的 settings 构造 app；把 env 指到 tmp，
    # 再 reload，避免写到 /data/bindcraft2_jobs。
    monkeypatch.setenv("BINDCRAFT2_JOBS_BASE_DIR", str(settings.jobs_base_dir))
    monkeypatch.setenv("BINDCRAFT2_ROOT", str(settings.root))
    monkeypatch.setenv("BINDCRAFT2_SHIPPED_WEIGHTS_DIR", str(settings.shipped_weights_dir))
    monkeypatch.setenv("BINDCRAFT2_PYTHON", settings.python)
    monkeypatch.setenv("BINDCRAFT2_MODULE", "")
    monkeypatch.setenv("BINDCRAFT2_ALPHAFOLD_PARAMS_DIR", str(settings.alphafold_params_dir))
    monkeypatch.setenv("BINDCRAFT2_COMPILE_CACHE_DIR", str(settings.compile_cache_dir))
    monkeypatch.setenv("BINDCRAFT2_GPU_PROBE_TTL_SECONDS", "0")
    server_app = importlib.reload(importlib.import_module("server.app"))
    return TestClient(server_app.app)


def _wait(client: TestClient, job_id: str, timeout_s: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout_s
    body: dict = {}
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {body}")


def test_health_and_detail(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "bindcraft2"

    detail = client.get("/healthz/detail").json()
    assert detail["service"] == "bindcraft2"
    assert detail["gpu_backend"] == "gpu"  # stub 探针
    assert detail["weights_loaded"] is False
    assert len(detail["weights_missing"]) == 7


def test_detail_reports_loaded_weights(offline_settings, monkeypatch, tmp_path):
    af = offline_settings.alphafold_params_dir / "params"
    af.mkdir(parents=True)
    from server.tools import CAMPAIGN_MODELS

    for name in CAMPAIGN_MODELS:
        (af / f"params_{name}.npz").write_bytes(b"x" * (101 << 20))
    # ProteinMPNN 随包：在 settings.root 下造三变体
    for variant in ("neutral", "negative", "positive"):
        d = offline_settings.root / "bindcraft" / "weights" / "proteinmpnn" / f"weights_{variant}"
        d.mkdir(parents=True)
        (d / "v_48_020.npz").write_bytes(b"x" * (2 << 20))

    client = _client(offline_settings, monkeypatch)
    detail = client.get("/healthz/detail").json()
    assert detail["weights_loaded"] is True
    assert detail["weights_missing"] == []
    assert detail["proteinmpnn_weights_loaded"] is True


def test_design_with_shipped_target(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    r = client.post(
        "/api/design",
        data={"target_name": "hPDL1", "number_of_final_designs": "1", "max_trajectories": "1"},
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    body = _wait(client, job_id)
    assert body["status"] == "completed", body

    files = client.get(f"/api/jobs/{job_id}/files").json()
    assert "3_Ranked/!_Ranked.csv" in files["files"]
    # 落盘的 campaign 内容可核对
    campaign = (
        offline_settings.jobs_base_dir / job_id / "input" / "campaign.json"
    ).read_text(encoding="utf-8")
    assert '"target": "hPDL1"' in campaign
    assert '"max_trajectories": 1' in campaign


def test_design_with_upload(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    r = client.post(
        "/api/design",
        data={"target_chains": "A", "hotspots": "54,56"},
        files={"target": ("target.pdb", b"ATOM\n", "chemical/x-pdb")},
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    body = _wait(client, job_id)
    assert body["status"] == "completed", body

    campaign = (
        offline_settings.jobs_base_dir / job_id / "input" / "campaign.json"
    ).read_text(encoding="utf-8")
    assert '"name": "target"' in campaign
    assert (offline_settings.jobs_base_dir / job_id / "input" / "target.pdb").is_file()


def test_design_rejects_no_target(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    r = client.post("/api/design", data={"number_of_final_designs": "1"})
    assert r.status_code == 422
    assert "Exactly one of" in r.text


def test_design_rejects_target_name_plus_upload(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    r = client.post(
        "/api/design",
        data={"target_name": "hPDL1"},
        files={"target": ("target.pdb", b"ATOM\n", "chemical/x-pdb")},
    )
    assert r.status_code == 422


def test_design_rejects_descending_binder_lengths(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    r = client.post(
        "/api/design",
        data={"target_name": "hPDL1", "binder_lengths": "[100,60]"},
    )
    assert r.status_code == 422


def test_design_rejects_invalid_json_in_complex_field(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    r = client.post("/api/design", data={"target_name": "hPDL1", "binder_lengths": "80,80"})
    # model_form_depends 对复杂字段要求合法 JSON，逗号写法是 CLI 专用。
    assert r.status_code == 422
    assert "Invalid JSON" in r.text


def test_rank_reads_previous_job(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    design = client.post(
        "/api/design", data={"target_name": "hPDL1", "max_trajectories": "1"}
    ).json()
    _wait(client, design["job_id"])

    r = client.post(
        "/api/rank",
        data={"campaign_uri": f"job://{design['job_id']}", "on": '["i_pTM"]'},
    )
    assert r.status_code == 200, r.text
    body = _wait(client, r.json()["job_id"])
    assert body["status"] == "completed", body

    files = client.get(f"/api/jobs/{r.json()['job_id']}/files").json()
    assert "ranked_by_i_pTM.csv" in files["files"]


def test_filter_reads_previous_job(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    design = client.post(
        "/api/design", data={"target_name": "hPDL1", "max_trajectories": "1"}
    ).json()
    _wait(client, design["job_id"])

    r = client.post(
        "/api/filter",
        data={"campaign_uri": f"job://{design['job_id']}", "where": '["i_pAE=0.45"]'},
    )
    assert r.status_code == 200, r.text
    body = _wait(client, r.json()["job_id"])
    assert body["status"] == "completed", body
    files = client.get(f"/api/jobs/{r.json()['job_id']}/files").json()
    assert "filtered.csv" in files["files"]


def test_rank_rejects_bad_campaign_uri(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    r = client.post("/api/rank", data={"campaign_uri": "job://does-not-exist"})
    assert r.status_code == 404


def test_task_design_runs_atomically(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    r = client.post(
        "/api/tasks/design",
        data={"target_name": "hPDL1", "max_trajectories": "1"},
        headers={"bioagent-session-id": "s1"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "completed"


def test_task_rank_and_filter(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    design = client.post(
        "/api/tasks/design", data={"target_name": "hPDL1", "max_trajectories": "1"}
    ).json()
    job_id = design["job_id"]

    rank = client.post("/api/tasks/rank", data={"campaign_uri": f"job://{job_id}"})
    assert rank.status_code == 200, rank.text
    assert rank.json()["status"] == "completed"

    filt = client.post("/api/tasks/filter", data={"campaign_uri": f"job://{job_id}"})
    assert filt.status_code == 200, filt.text
    assert filt.json()["status"] == "completed"


def test_manifest_lists_six_endpoints_with_examples(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
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
        assert path in eps, f"missing {path}"
        assert eps[path]["examples"], f"{path} has no examples"
    assert body["service"] == "bindcraft2"
    assert body["service_specific"]["tool_outputs"]["ranked"].endswith("!_Ranked.csv")


def test_openapi_exposes_six_body_schemas(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    models = set(client.get("/openapi.json").json()["components"]["schemas"].keys())
    expected = {
        "Body_run_design_api_design_post",
        "Body_run_design_task_api_tasks_design_post",
        "Body_run_rank_api_rank_post",
        "Body_run_rank_task_api_tasks_rank_post",
        "Body_run_filter_api_filter_post",
        "Body_run_filter_task_api_tasks_filter_post",
    }
    assert not (expected - models), sorted(expected - models)


def test_upload_field_names_follow_convention(offline_settings, monkeypatch):
    client = _client(offline_settings, monkeypatch)
    eps = {e["path"]: e for e in client.get("/api/manifest").json()["endpoints"]}
    for path in ("/api/design", "/api/tasks/design"):
        fields = {f["name"]: f for f in eps[path]["request_fields"]}
        assert fields["target"]["is_file"] is True
        assert "target_uri" in fields
        # 不得出现旧式裸 input_uri
        assert "input_uri" not in fields


def test_gpu_probe_runs_out_of_process(offline_settings, monkeypatch):
    """GPU 探针必须走子进程：HTTP 进程绝不能 import jax。

    在 HTTP 进程里 import jax 会初始化 CUDA context、占掉 campaign 需要的显存；
    无卡时还会静默回退 CPU。stub 只在 `-c` 分支打印 `gpu`，因此
    `gpu_backend == "gpu"` 本身就证明探针是在子进程里跑的。
    """
    import sys

    client = _client(offline_settings, monkeypatch)
    detail = client.get("/healthz/detail").json()
    assert detail["gpu_backend"] == "gpu"
    assert "jax" not in sys.modules
