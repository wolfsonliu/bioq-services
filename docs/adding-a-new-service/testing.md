# Testing — 测试骨架

日期: 2026-07-14
适用: [新增 bioagent service cookbook](./index.md) 的测试部分
相关: [skeleton](./skeleton.md) · [deploy](./deploy.md) · [总览](./index.md)

> ← 返回 [新增 service cookbook 总览](./index.md)

本页覆盖四套测试骨架：`test_app.py`（离线 HTTP 单测）/ `test_cli.py`（CLI 批处理）/
`test_fc.py`（FC 部署后回归）/ `test_fc_task.py`（FC 异步任务模式集成）。经 gateway
调用的服务还应在 `gateway/tests/test_fc.py` 加一个 `TestEndToEnd<Svc>`
e2e 类（见 迁移到 OSS mount）。

### 10. `services/<svc>/tests/test_app.py`

```python
"""Offline tests for <svc>-server (no real algorithm / GPU needed).

`conftest.py` registers the service dir as `server` package, so
`from server.settings import ...` works without pip install.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("<SVC>_JOBS_BASE_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("<SVC>_ROOT", str(tmp_path / "<svc>"))
    (tmp_path / "<svc>").mkdir(parents=True, exist_ok=True)

    sys.modules.pop("server.app", None)
    server_app = importlib.import_module("server.app")
    return TestClient(server_app.app)


# ----- Healthcheck / manifest -----

def test_health(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["service"] == "<svc>"
    assert "version" in health


def test_manifest_service_name(client):
    body = client.get("/api/manifest").json()
    assert body["service"] == "<svc>"


def test_manifest_lists_endpoints(client):
    body = client.get("/api/manifest").json()
    paths = {e["path"] for e in body["endpoints"]}
    assert "/api/generate" in paths


def test_manifest_extras_has_tool_outputs(client):
    extras = client.get("/api/manifest").json()["service_specific"]
    assert "generate" in extras["tool_outputs"]


# ----- Settings -----

def test_settings_defaults():
    from server.settings import <Svc>Settings

    class _Off(<Svc>Settings):
        model_config = SettingsConfigDict(env_prefix="<SVC>_TEST_", env_file=None, extra="ignore")

    s = _Off()
    assert s.jobs_base_dir == Path("/data/<svc>_jobs")
    assert s.root == Path("/opt/<svc>")
    # 权重外置化：default 必须指向 NAS 挂载点
    assert s.weights_dir == Path("/data/models/<svc>")


# ----- Endpoint smoke (no real pipeline) -----

def test_endpoint_returns_job(client):
    resp = client.post("/api/generate", data={"n_samples": "2"})
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["input_params"] is not None
```

> **测试要点**：(1) `client` fixture 用 `monkeypatch.setenv` + `importlib.import_module` 重新导入 app，确保 settings 使用测试目录。(2) 每个 endpoint 至少有一个 smoke test 验证提交返回 `job_id`。(3) manifest 测试覆盖 service 名、endpoint 列表、extras。(4) settings 测试验证默认值与 env override。

### 11. `services/<svc>/tests/test_cli.py`

CLI 批处理模式的离线测试。覆盖三个层面：(1) endpoint 注册正确性；(2) `build_argv` 回调生成正确的命令行；(3) 端到端 `create_cli` 流程（mock SubprocessRunner）。

```python
"""CLI batch-mode tests for <svc>-server.

Tests endpoint registration, build_argv callbacks, and end-to-end create_cli.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from bioq_service.cli import CLIEndpoint, create_cli

from server.adapter import <Svc>Adapter
from server.models import GenerateRequest
from server.settings import <Svc>Settings
from server.tools import generate_argv


class _Off(<Svc>Settings):
    """Settings subclass that ignores .env and real env vars."""
    model_config = SettingsConfigDict(
        env_prefix="<SVC>_TEST_", env_file=None, extra="ignore",
    )


def _generate_build(req, inputs, job_dir, settings):
    return generate_argv(
        req, job_dir=job_dir, input_pdb=inputs["input_pdb"], settings=settings,
    )


ENDPOINTS = {
    "generate": CLIEndpoint(
        name="generate",
        help="Run generation on an input PDB",
        request_model=GenerateRequest,
        build_argv=_generate_build,
        inputs={"input_pdb": ("Input PDB file", True)},
    ),
}


# ---- Endpoint registration ----


def test_endpoint_keys():
    assert set(ENDPOINTS.keys()) == {"generate"}


def test_generate_endpoint_fields():
    ep = ENDPOINTS["generate"]
    assert ep.request_model is GenerateRequest
    assert ep.inputs["input_pdb"][1] is True  # required


# ---- Build_argv callbacks ----


def test_generate_build_argv(tmp_path):
    s = _Off()
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    pdb = tmp_path / "input.pdb"
    pdb.write_text("ATOM")

    argv = _generate_build(
        GenerateRequest(n_samples=3),
        {"input_pdb": pdb},
        job_dir,
        s,
    )
    # Assert argv structure matches what tools.py generates.
    # Concrete assertions depend on generate_argv implementation.
    assert len(argv) > 0


# ---- End-to-end create_cli ----


def test_cli_generate_success(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = <Svc>Adapter(settings=s)

    pdb = tmp_path / "input.pdb"
    pdb.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "generate",
        "--input-pdb", str(pdb),
        "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0


def test_cli_generate_json_output(tmp_path, capsys):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = <Svc>Adapter(settings=s)

    pdb = tmp_path / "input.pdb"
    pdb.write_text("ATOM")
    output_dir = tmp_path / "run" / "output"

    with patch.object(sys, "argv", [
        "prog", "generate",
        "--input-pdb", str(pdb),
        "--json", "--output-dir", str(output_dir),
    ]):
        with patch("bioq_service.cli.SubprocessRunner") as mock_runner:
            mock_runner.run.return_value = 0
            with patch.object(adapter, "detect_outputs", return_value=True):
                with pytest.raises(SystemExit) as exc_info:
                    create_cli(adapter, s, ENDPOINTS, version="0.0.1")
                assert exc_info.value.code == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert result["return_code"] == 0


def test_cli_no_subcommand_exits_2(tmp_path):
    s = _Off(jobs_base_dir=tmp_path / "jobs")
    adapter = <Svc>Adapter(settings=s)

    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(SystemExit, match="2"):
            create_cli(adapter, s, ENDPOINTS)
```

> **测试要点**：
> - **`_Off` settings** — 用独立的 `env_prefix` + `env_file=None` 隔离环境变量，避免开发机的 `.env` 或真实环境变量干扰测试
> - **Endpoint registration** — 验证 `ENDPOINTS` dict 的 key 集合、每个 endpoint 的 `request_model` 和 `inputs` 声明正确
> - **Build_argv** — 直接调用 `_xxx_build()` 回调，断言生成的 argv 包含预期的脚本路径、关键参数；使用 `tests/data/` 中的 fixture 文件做 "with data file" 变体
> - **End-to-end create_cli** — mock `SubprocessRunner` 和 `adapter.detect_outputs`，用 `patch.object(sys, "argv", [...])` 注入 CLI 参数，断言 `SystemExit.code`。success / json output / failure / no-subcommand 四种场景覆盖
> - **复杂参数** — `dict` / `list[Model]` 等无法用 argparse flag 表达的字段，在 end-to-end 测试中用 `--params-json` 传入（参考 rfdiffusion2-server 的 `contig_atoms`、boltz-server 的 `sequences`）

### 12. `services/<svc>/tests/test_fc.py` + `tests/data/`

部署后的 FC URL 回归测试。Marker = `fc`，默认 skip；显式开启用 `pytest -m fc` 或 `RUN_FC_TESTS=1`。URL 从 [services.yaml](../../services.yaml) 读 —— 部署完成后把新条目写进那个文件。

**测试数据放置原则（默认自包含）**：任何测试脚本（`test_fc.py` / `test_fc_task.py` / `test_cli.py` / …）需要的 fixture（PDB / JSON / FASTA / prm7+rst7 / etc.）**必须尽量**复制到该测试所在目录下的 `tests/data/` 并一起 commit —— 让 suite 自包含，新 clone 直接能跑。**绝不**引用 `opensource/*`（gitignore，新 clone 缺失）或其它服务目录。测试里用 `DATA_DIR = Path(__file__).resolve().parent / "data"` 定位，别用相对 `opensource` 的路径。

**唯一例外——文件太大不适合进 git**：单个 fixture 达到多 MB 级别（repo 目前最大 tracked fixture ~700 KB，且没有 git-LFS）时，才不 commit。此时：

- 保留 `tests/data/` 作为 fixture 的**默认查找位置**，但把该目录（或具体文件）加进 `.gitignore`；
- 测试用 env override（如 `<SVC>_TEST_STRUCTURE` / `_PARAMETERS`）覆盖默认路径，并在 fixture 缺失时 `pytest.mark.skipif` 跳过而非报错；
- 在 service README + 测试 docstring 里写明从哪里取 / 如何 stage 到 `tests/data/`。

> 判断标准：**能塞进 git 就塞**（几百 KB～数 MB 的单个 MD/结构 fixture 通常可接受，尤其是没有更小的合法替代时——如 OpenBPMD 的 solvated 体系 ~10 MB 仍选择 commit 以保自包含）。只有当文件大到明显拖累 clone（几十 MB+）或本就是二进制权重/数据集时才走例外路径。有疑问时优先自包含。

```python
"""End-to-end tests against the deployed <Svc> Function Compute service.

Marked `@pytest.mark.fc`, skipped by default. Run with:

    pytest -m fc services/<svc>-server/tests/test_fc.py

Test fixtures ship in `tests/data/`, so the suite is self-contained — no
dependency on `opensource/`.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_PDB = DATA_DIR / "<example>.pdb"

pytestmark = pytest.mark.fc


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url("<svc>-server", start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        yield c


# ----- Smoke -----

def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "<svc>"
    assert "version" in body


def test_manifest_lists_endpoints(client: httpx.Client) -> None:
    r = client.get("/api/manifest")
    r.raise_for_status()
    paths = {e["path"] for e in r.json()["endpoints"]}
    assert paths == {"/api/<endpoint>", ...}  # 每个 service 自己列


def test_openapi_served(client: httpx.Client) -> None:
    client.get("/openapi.json").raise_for_status()


def test_unknown_job_returns_404(client: httpx.Client) -> None:
    assert client.get("/api/jobs/missing-job-id").status_code == 404


# ----- Inference: 每个 endpoint 至少 1 个最小 job -----

def test_<endpoint>_minimal_job(client: httpx.Client, base_url: str) -> None:
    with open(TEST_PDB, "rb") as fh:
        r = client.post(
            "/api/<endpoint>",
            files={"pdb": (TEST_PDB.name, fh, "chemical/x-pdb")},
            data={"name": "fc_smoke", "<smallest-param>": "..."},
        )
    r.raise_for_status()
    final = poll_job(client, base_url, r.json()["job_id"])
    assert final["status"] == "completed", final
    files = client.get(f"/api/jobs/{final['job_id']}/files").json()["files"]
    assert files
```

`conftest.py` 需要做两件事：(1) 把 `services/<svc>/` 注册为 `server` 包——这样 `from server.app import app` 在测试中可用（Dockerfile 中它被 COPY 成 `/opt/.../server/`，但本地开发没有这层目录重映射）；(2) 注册 `fc` marker。

```python
"""Register `server` package alias + `fc` marker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent

if "server" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "server",
        SERVICE_DIR / "__init__.py",
        submodule_search_locations=[str(SERVICE_DIR)],
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["server"] = module
        spec.loader.exec_module(module)

from bioq_service.fc_testing import (  # noqa: E402
    register_fc_marker,
    skip_fc_tests_unless_enabled,
)


def pytest_configure(config):
    register_fc_marker(config)


def pytest_collection_modifyitems(config, items):
    skip_fc_tests_unless_enabled(config, items)
```

> **为什么需要 `importlib.util`？** 离线测试直接 `uv run pytest services/<svc>/tests/` 运行——Python 搜索路径上没有 `server` 包。这段代码把 `services/<svc>/` 目录注册为 `server` 模块，之后 `from server.settings import ...` 就能正常解析。大部分现有 service 已经用这个 pattern。

详细测试流程 / MCP 协议层 / 失败模式速查见 testing-fc-services.md。

### 12b. `services/<svc>/tests/test_fc_task.py`

FC **异步任务模式** 集成测试。测试目标：验证 `/api/tasks/<name>` 端点在
`X-Fc-Invocation-Type: Async` 下端到端跑通 —— submit 立即 202、job 最终 completed、
生命周期端点可读、平台层 dedup 生效。同样是 `@pytest.mark.fc`（复用 Section 12 的
`conftest.py`，不需要额外注册）；默认 skip，跑法：

```bash
RUN_FC_TESTS=1 \
uv run python -m pytest -m fc services/<svc>-server/tests/test_fc_task.py -v
```

为什么额外写这个文件而不是并到 `test_fc.py`？
- 两种调用模式（sync submit/poll vs. async task）的 header / 断言 / 生命周期语义差别足够大，
  塞一起 fixture 会打架；分文件更清晰。
- 异步任务模式是长跑 GPU 服务的**推荐**入口（无 HTTP gateway 30 s 回收 + 平台层 dedup +
  排队而非 429）。sync 保留是 legacy 兼容 —— 未来 pipeline 迁到 K8s / 别的平台时首先
  切换的就是它。异步先跑通、sync 兜底。
- Task 模式下 FC 事件 payload 有 **128 KiB 上限**（`EntityTooLarge` 400 otherwise），
  大 PDB / 大 zip 必须走 URI fallback，需要单独的 fixture 组织。

#### 骨架

```python
"""FC async task mode tests for <svc>-server (opt-in).

Run with::

    RUN_FC_TESTS=1 \\
    uv run python -m pytest -m fc services/<svc>-server/tests/test_fc_task.py -v

Validates the /api/tasks/<name> endpoints end-to-end under FC async task mode
(``X-Fc-Invocation-Type: Async``).  Async task mode pins the FC instance
for the whole job (no 30 s HTTP-gateway recycle risk) and dedups by
``X-Fc-Async-Task-Id`` at the platform layer.
"""

from __future__ import annotations

import io
import time
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest

from bioq_service.fc_testing import fc_url, poll_job

SERVICE = "<svc>-server"

# 每个 endpoint 需要多长；覆盖服务最慢的一次推理。
POLL_TIMEOUT_S = 1800
POLL_INTERVAL_S = 20
TIMEOUT = httpx.Timeout(connect=30, read=300, write=600, pool=30)


@pytest.fixture(scope="module")
def base_url() -> str:
    return fc_url(SERVICE, start=Path(__file__))


@pytest.fixture(scope="module")
def client(base_url: str):
    with httpx.Client(base_url=base_url, timeout=TIMEOUT) as c:
        yield c


# 每个 endpoint 一个 task_id，同一 module 内所有断言共享。
@pytest.fixture(scope="module")
def <endpoint>_task_id() -> str:
    return f"fc-async-<endpoint>-{int(time.time())}-{uuid.uuid4().hex[:6]}"


# ---- 关键 headers ----
def _async_headers(task_id: str) -> dict[str, str]:
    return {
        "X-Fc-Invocation-Type": "Async",
        "X-Bioagent-Job-Id": task_id,
        "X-Fc-Async-Task-Id": task_id,
    }


# ---- 429 抖动兜底 GET（sync test_fc.py 也应该有一份） ----
def _get_with_retry(client, path, *, max_attempts=20, backoff_s=30):
    """FC 网关 429 时轮询重试。max_concurrent_jobs=1 的服务尤其需要。"""
    last = None
    for _ in range(max_attempts):
        last = client.get(path)
        if last.status_code != 429:
            return last
        time.sleep(backoff_s)
    return last


def _poll_to_completion(client, task_id: str) -> dict:
    # framework 默认 max_transient_errors=10 × interval=15s = 150s，
    # 不够顶 FC 4-7 min 的 429 window。这里 60 × 20s = 20 min buffer。
    final = poll_job(
        client, "", task_id,
        timeout_s=POLL_TIMEOUT_S, interval_s=POLL_INTERVAL_S,
        max_transient_errors=60,
    )
    assert final["status"] == "completed", f"task did not complete: {final}"
    return final


# ---- 每个 endpoint 一对 submit_response / task fixture ----
@pytest.fixture(scope="module")
def <endpoint>_submit_response(client, <endpoint>_task_id):
    return client.post(
        "/api/tasks/<endpoint>",
        data={"<param>": "<smallest-valid>"},
        # 若上传超过 128 KiB，改用 URI fallback（见下文「大文件」段）
        headers=_async_headers(<endpoint>_task_id),
    )


@pytest.fixture(scope="module")
def <endpoint>_task(client, <endpoint>_task_id, <endpoint>_submit_response) -> dict:
    assert <endpoint>_submit_response.status_code == 202, (
        f"async <endpoint> submit returned {<endpoint>_submit_response.status_code}: "
        f"{<endpoint>_submit_response.text!r}"
    )
    return _poll_to_completion(client, <endpoint>_task_id)


# ===================================================================
# Section 1: submit 语义 + OpenAPI
# ===================================================================


@pytest.mark.fc
class TestAsyncSubmit:
    def test_<endpoint>_returns_202(self, <endpoint>_submit_response):
        assert <endpoint>_submit_response.status_code == 202

    def test_task_endpoints_registered_in_openapi(self, client):
        r = _get_with_retry(client, "/openapi.json")
        assert r.status_code == 200
        expected = {"/api/tasks/<endpoint>", ...}
        missing = expected - set(r.json()["paths"])
        assert not missing, (
            f"task endpoints missing: {missing}; "
            f"settings.task_endpoints_enabled 可能是 False"
        )


# ===================================================================
# Section 2: 每个 endpoint 的 completion + output
# ===================================================================


def _assert_completed_with_output(task, task_id, client, expected_name,
                                  *, min_duration_s=3.0):
    """job 完成 + 有 expected_name 产物。min_duration_s 用作 subprocess 真跑过的兜底。"""
    assert task["status"] == "completed"
    assert task["job_id"] == task_id
    d = task.get("duration_seconds")
    assert d is not None and d > min_duration_s, (
        f"duration {d}s too short (min {min_duration_s}s) — subprocess 可能没真跑"
    )
    assert task.get("output_count", 0) > 0
    r = _get_with_retry(client, f"/api/jobs/{task_id}/files")
    assert r.status_code == 200
    assert any(expected_name in f for f in r.json()["files"])


@pytest.mark.fc
class TestAsync<Endpoint>:
    def test_completed(self, <endpoint>_task, <endpoint>_task_id, client):
        _assert_completed_with_output(
            <endpoint>_task, <endpoint>_task_id, client, "<expected-out>",
            min_duration_s=60,  # 快 endpoint 传更小的数
        )

    def test_input_params_echoed(self, <endpoint>_task):
        params = <endpoint>_task.get("input_params") or {}
        assert params.get("<param>") == <expected-value>

    def test_output_downloadable(self, client, <endpoint>_task_id, <endpoint>_task):
        files = _get_with_retry(
            client, f"/api/jobs/{<endpoint>_task_id}/files"
        ).json()["files"]
        f = next(x for x in files if "<expected-out>" in x)
        r = _get_with_retry(client, f"/api/jobs/{<endpoint>_task_id}/file/{f}")
        assert r.status_code == 200
        assert len(r.content) > 100


# ===================================================================
# Section 3: 生命周期（用最便宜的 endpoint fixture）
# ===================================================================


@pytest.mark.fc
class TestJobLifecycle:
    def test_status_endpoint(self, client, <endpoint>_task_id, <endpoint>_task):
        body = _get_with_retry(client, f"/api/jobs/{<endpoint>_task_id}").json()
        assert body["status"] == "completed"

    def test_log_endpoint(self, client, <endpoint>_task_id, <endpoint>_task):
        r = _get_with_retry(client, f"/api/jobs/{<endpoint>_task_id}/log")
        assert r.status_code == 200
        assert len((r.json().get("log") or r.json().get("text") or "")) > 0

    def test_download_zip(self, client, <endpoint>_task_id, <endpoint>_task):
        r = _get_with_retry(client, f"/api/jobs/{<endpoint>_task_id}/download")
        assert r.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        assert any("<expected-out>" in n for n in zf.namelist())


# ===================================================================
# Section 4: 平台层 dedup — 同 task_id 重复提交
# ===================================================================


@pytest.mark.fc
class TestAsyncDuplicateDedup:
    """重复提交同 X-Fc-Async-Task-Id — FC 应 409（平台层）或 202（转发后
    framework execute_task 层返回已存在的 JobInfo）；无论哪种，都不能重跑。"""

    def test_duplicate_does_not_rerun(self, client, <endpoint>_task_id, <endpoint>_task):
        first_created = <endpoint>_task["created_at"]
        first_completed = <endpoint>_task["completed_at"]

        r2 = client.post(
            "/api/tasks/<endpoint>",
            data={"<param>": "<different-value>"},  # 故意改参数验证不生效
            headers=_async_headers(<endpoint>_task_id),
        )
        assert r2.status_code in (202, 409), (
            f"expected 409 或 202; got {r2.status_code} {r2.text!r}"
        )
        if r2.status_code == 202:
            time.sleep(15)

        re_query = _get_with_retry(client, f"/api/jobs/{<endpoint>_task_id}").json()
        assert re_query["status"] == "completed"
        assert re_query["created_at"] == first_created, "created_at 被重置"
        assert re_query["completed_at"] == first_completed, "任务被重跑了"
```

#### 大文件 / 大 zip：128 KiB event payload cap

FC 异步网关对 event body 有 **128 KiB 上限**（否则 400 `EntityTooLarge`）。
估算 payload 大小时把 form fields + 文件 base64 + multipart boundary 都算上；
文件超过 ~100 KiB 就要走 URI fallback。

**服务端准备**：为大文件字段增加 URI 版并做 `resolve_input(upload, uri, ...)`
分派。参考 `services/rfantibody-server/app.py` 的 `target_uri` / `framework_uri`
参数以及 `services/rfantibody-server/uris.py::resolve_input`。

**测试端 sync-bootstrap 模式**：只跑一次同步 POST 把大文件落到 NAS，异步测试通过
`file://<jobs_base>/<bootstrap_id>/input/<file>` URI 引用。参考
[services/rfantibody-server/tests/test_fc_task.py](../../services/rfantibody-server/tests/test_fc_task.py)
的 `staged_pdb_uris` fixture 与 `RFANTIBODY_TEST_*_NAS_PATH` env 覆盖入口。

**pipeline 链式调用**：后续步骤通过 `job://<prev_job_id>/<file>` URI 引用上一步产物，
不需要落盘再上传，参考 `TestAsyncProteinMPNN` / `TestAsyncRF2` 的 fixture。

#### `max_concurrent_jobs=1` 服务的 429 处理

如果 FC 控制台配置的 `max_concurrent_request` 很紧（多数 GPU 服务的默认值），
GET `/api/jobs/<id>` 会因平台层限流被 429 挡住，framework 的 `poll_job` 默认只
容忍 10 次连续错（`max_transient_errors=10` × `interval_s=15` = 2.5 min，不够 FC 
4-7 min 的 429 window）。在测试中必须显式 override：

```python
poll_job(client, "", task_id,
         timeout_s=1800, interval_s=20,
         max_transient_errors=60)  # 60 × 20s = 20 min buffer
```

同时所有非 poll_job 的 GET / DELETE 都要走 `_get_with_retry` 一样的包装。参考
[project 记忆 `project_fc_http_polling_unreliable_at_concurrency.md`]
 —— 高并发下 GET 端点必然会漏抖。

#### 参考实现

- [genie3-server/tests/test_fc_task.py](../../services/genie3-server/tests/test_fc_task.py) — 无
  文件上传的场景（unconditional）+ 小 zip 上传（motif/binder < 30 KB）+ 自定义 YAML。
- [rfantibody-server/tests/test_fc_task.py](../../services/rfantibody-server/tests/test_fc_task.py) — 大 PDB
  → sync bootstrap → `file://` URI，pipeline 链式 `job://` URI。

