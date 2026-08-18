# FC 集成测试

[English](fc-testing.md) | 中文

> **适用**：跑或写针对已部署 Function Compute（FC）服务的 `@pytest.mark.fc` 集成测试时；或是遇到「FC 起了一堆实例」、「403 AccessDenied」、冷启动超时这些问题时。
> **来源**：`services/<svc>-server/tests/test_fc*.py`、`framework/src/bioq_service/fc_testing.py`、`services/<svc>-server/deploy/fc.yaml`、FC 控制台实际配置。
> **刷新/删除条件**：FC 控制台的会话亲和 / trigger 配置变化，或 `fc_url()` 增加 base URL 的 env 覆盖之后。

FC 测试只对已部署环境有意义，永远不会在任何离线 run 里执行——它们标记为
`@pytest.mark.fc`，除非 `-m fc` / `RUN_FC_TESTS=1` 否则一律 skip（离线各层见
[testing.zh.md](testing.zh.md)）。

```bash
cd services/<svc>-server
RUN_FC_TESTS=1 uv run --group dev python -m pytest -m fc tests/ -v
```

## URL 解析与 VPC / 公网的区分

`base_url` fixture 通过 `bioq_service.fc_testing` 的 `fc_url(service_name)` 解析部署地址，
实际读的是 `services.yaml`（`services.<name>.url`）。这个条目是 **VPC HTTP trigger**
（`https://fc-...-vpc.fcapp.run`），gateway 及 VPC 内的一切都走它。

公网对应物是同主机名去掉 `-vpc` 的 `https://fc-...cn-hangzhou.fcapp.run`，能否访问由 trigger
的 `disableURLInternet` 开关决定（`deploy/fc.yaml` 默认 `true`）：

| 来源         | `-vpc.fcapp.run`                                  | 公网，`disableURLInternet=true`                                                          | 公网，`disableURLInternet=false` |
|--------------|---------------------------------------------------|------------------------------------------------------------------------------------------|----------------------------------|
| 阿里云 VPC 内 | 可用                                              | —                                                                                        | —                                |
| 外部机器      | `ConnectTimeout`（DNS 解析到内网 `100.x`，无路由） | `403 AccessDenied: "access denied due to function internet URL is disabled"`             | 可用（有冷启动）                 |

**要从外部机器测试**：先开启公网入口（控制台，或把 `deploy/fc.yaml` 的 `disableURLInternet`
改成 `false`），再把测试指向公网地址。`fc_url()` **没有 env 覆盖**，所以实用做法是临时把
`services.yaml` 里那一行 url 改成公网地址 → 跑测试 → 还原：

```bash
git checkout -- services.yaml   # 跑完还原；gateway 仍需 VPC 地址
```

## 会话亲和 ——「起了一堆实例」的根因

FC 配置了 **HeaderField 会话亲和**（`deploy/fc.yaml`）：

```yaml
sessionAffinity: HEADER_FIELD
affinityHeaderFieldName: bioagent-session-id
sessionConcurrencyPerInstance: 1
```

同一测试模块的**每个请求都必须带同一个 `bioagent-session-id` 值**，submit 与它后续的每次
poll 才会绑定到同一实例。不带时 FC 会把每次调用当成新会话，fa 开散到很多（常常是新建的）
实例上——这就是「打开控制台发现十几个实例」的失败模式。

框架对这套机制做了端到端支持：

- `_SessionAffinityMiddleware`（`framework/src/bioq_service/app.py`）在返回 `job_id` 的 `200`
  POST 响应里回显服务端分配的会话值（`job_id`）到 `bioagent-session-id` 响应头。它在
  `settings.session_header_name` 被设置时启用（env `<PREFIX>_SESSION_HEADER_NAME`，例如
  `PROTEINMPNN_SESSION_HEADER_NAME=bioagent-session-id`）。
- 测试侧必须在**每个**请求（submit **和** poll）发一致的值。仓库里的做法（见 alphafold /
  deeprank-ab / genie3 / … 的 `tests/test_fc.py`）是 module 级 fixture + client 级 header：

```python
SESSION_HEADER = "bioagent-session-id"

@pytest.fixture(scope="module")
def session_headers() -> dict[str, str]:
    return {SESSION_HEADER: f"test-{uuid.uuid4().hex[:12]}"}

@pytest.fixture(scope="module")
def client(base_url: str, session_headers: dict[str, str]) -> httpx.Client:
    # client 级 header 会 merge 进每个请求（包括 poll_job 里 headers={} 的 GET），
    # 所以 submit / poll / smoke / download 全都带上了。
    with httpx.Client(base_url=base_url, timeout=..., headers=session_headers) as c:
        yield c
```

等价做法：每个 `client.post(...)` 传 `headers=session_headers`、每个 `poll_job(...)` 传
`extra_headers=session_headers`。哪种都行，关键是跨请求一致。header 名是历史契约，必须保持
`bioagent-session-id`（见 [conventions.zh.md](conventions.zh.md)）。

## 冷启动与 429 重试

GPU 函数 scale-to-zero，首次请求冷启动约 12–40 s。给足超时（例如
`httpx.Timeout(connect=30, read=300, write=600, pool=30)`），否则第一次 submit/poll 会误报超时。

账号级 GPU 配额耗尽会在**任意**请求（包括便宜的 `/healthz`）上表现为 `429`。
`bioq_service.fc_testing` 提供两个工具：

- `make_retrying_client(base_url, *, timeout=120.0, max_retries=10, backoff_s=20.0)` —— 底层
  自动对 `429` 线性退避重试的 `httpx.Client`。
- `poll_job(..., max_transient_errors=..., interval_s=...)` —— 对 `max_concurrent_jobs=1` 的
  服务要调大 `max_transient_errors`（如 `60 × 20s`≈20 min 缓冲），因为 FC 的配额窗口比默认
  `10 × 15s` 长。

## 本地前置条件（为什么 `uv run --group dev` 会失败）

- 必须有 `[dependency-groups] dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]`，否则
  `uv run --group dev` 报 `Group "dev" is not defined`。历史 service 可能还残留旧的
  `[build-system]`+`[tool.setuptools]` 块而不是 `[tool.uv] package = false` —— 对齐成现行约定。
- `bioq_service` 必须可解析：`[tool.uv.sources] bioq-service-framework = { path = "../../framework", editable = true }`。
- `~/.cache/uv` 只读的沙箱里要设 `UV_CACHE_DIR=/tmp/...`（或 `UV_NO_CACHE=1`），否则 uv 还没解析就先报错。

## 离线 fixture 的坑（`helper_scripts` 500）

submit 端点会在 `runner.submit → build_argv` 里**同步**构造 argv，而 `prepare_inputs()` 又在
同一个请求里跑上游 `helper_scripts/*.py`。所以指向空 tmp 目录 `PROTEINMPNN_ROOT` 的离线
`test_app.py` `client` fixture 会拿到 `500 "Helper script not found"` 而不是 `200`。要在
fixture 里 stub 这些 helper 脚本（`parse_multiple_chains.py` 等）——见
`services/proteinmpnn-server/tests/test_app.py`。

## 已知缺口 / 检查清单

- 别把「永远 `pytest.skip` 的用例」当覆盖。`services/proteinmpnn-server/tests/test_fc.py::
  test_job_uri_cross_reference` 因为纯 design 只产出 FASTA 不产出 PDB 而永远 skip，`job://`
  PDB 跨任务引用这条路实际上从没被跑过——遇到这种 skip 要单独标注，而不是默认它已覆盖。
- 一次合格的通过应该是：smoke 全 `PASSED`、`422` 校验全 `PASSED`、每个 endpoint 至少一个
  inference job `PASSED` 且 `duration_seconds > 0`、输出非空。
- 跑完确认 `git status` 干净（`services.yaml` 的临时覆盖已还原）。