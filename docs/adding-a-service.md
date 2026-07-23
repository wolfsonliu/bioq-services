# 在 bioq-services 里新增一个服务

这是 **repo-local 落地指南**：在 `bioq-services` 仓库里把一个新算法/模型包成服务的具体步骤。

> **完整流程与设计原则（权威）在 `bioagent` monorepo 的
> `engineering/guides/adding-a-new-service/` cookbook** —— 骨架、Dockerfile（uv venv /
> conda 两套）、conda 踩坑、测试骨架、FC 部署、以及提交前的完整 checklist 都在那里。本页只讲
> 「代码落在本仓的哪里、怎么跑起来」，不重复 cookbook 的细节。

## 两仓分工（先搞清楚产物放哪）

服务**代码只在本仓（`bioq-services`）开发**；本仓无 `engineering/`，所以设计文档等留在 monorepo。

| 产物 | 仓库 |
|---|---|
| 设计文档 `engineering/decisions/YYYY-MM-DD-<svc>-server-design.md` + `index.md` 一行 | **`bioagent`（monorepo）** |
| 调用指南 base-URL 行 `engineering/guides/calling-bioagent-services.md` | **`bioagent`（monorepo）** |
| 服务代码 `services/<svc>-server/`（全套） | **本仓** |
| `services.yaml` 条目 | **本仓** |
| gateway e2e `gateway/tests/test_fc.py` 的 `TestEndToEnd<Svc>` | **本仓** |

**开工前先在 monorepo 写设计文档**（cookbook §0 必做），再回本仓写代码。

## 命名约定（本仓特有）

- import 名：**`bioq_service`**（`from bioq_service import ...`）——不是 `bioagent_service`。
- 分发名：**`bioq-service-framework`**——不是 `bioagent-service-framework`。
- HTTP header（`X-Bioagent-*`）、`bioagent-session-id` 等**不变**。

cookbook 子页的代码样例仍写 `bioagent_service`——照抄到本仓时统一改成 `bioq_service`。

## 必备文件（`services/<svc>-server/`）

```
services/<svc>-server/
├── __init__.py          # 空
├── settings.py          # <Svc>Settings(ServiceSettings)，env_prefix=<SVC>_
├── models.py            # 请求 pydantic models
├── tools.py             # argv builders（可选，视服务复杂度）
├── adapter.py           # <Svc>Adapter(JobAdapter)：name + detect_outputs + manifest_extras + endpoint_examples
├── app.py               # create_app + endpoints + /healthz/detail + task endpoints
├── __main__.py          # CLI 批处理入口（CLIEndpoint + create_cli）
├── Dockerfile
├── VERSION              # 如 v0.0.1（Makefile 读作镜像 tag）
├── README.md            # 架构图 + 每 endpoint curl + 配置表 + Weights 章节
├── pyproject.toml       # 离线测试/开发依赖（见下）
├── scripts/
│   ├── vendor.sh        # 必备：clone 上游到 upstream/ at pinned SHA
│   └── fetch_weights.sh # 可选：下载权重到 weights/ 或 NAS（WEIGHTS_DST 覆盖）
└── tests/
    ├── __init__.py
    ├── conftest.py      # server alias（importlib）+ fc marker
    ├── test_app.py      # 离线 TestClient 单测
    ├── test_cli.py      # CLI 批处理单测
    ├── test_fc.py       # FC sync 集成测试（@pytest.mark.fc，默认 skip）
    ├── test_fc_task.py  # FC 异步任务模式测试（默认 skip）
    └── data/            # 测试 fixture（小 PDB/JSON 等）
```

`upstream/`（vendor.sh 产物）与 `weights/`（fetch_weights.sh 产物）**不入库**——加进本仓
`.gitignore`（若尚无 per-service 条目，追加 `services/<svc>-server/upstream/` 与
`services/<svc>-server/weights/`）。

最省事的起点：把一个结构相近的现有服务整目录拷贝改名，再逐文件改。参考：
- `proteinmpnn-server` —— uv venv + 序列设计 + 权重外置的最简骨架；
- `dockq-server` / `diamond-server` —— CPU-only uv-venv slim；
- `deeprank-ab-server` / `pocketxmol-server` —— conda/micromamba 多阶段。

## pyproject.toml（离线测试环境）

只声明离线测试所需的轻依赖 + 框架 path source；运行时重依赖在 Dockerfile 里。

```toml
[project]
name = "<svc>-server"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = ["bioq-service-framework[mcp]"]

[tool.uv.sources]
bioq-service-framework = { path = "../../framework", editable = true }

[tool.uv]
package = false            # server 代码通过 Dockerfile COPY 注入，不 pip 装本包

[dependency-groups]
dev = ["pytest", "pytest-asyncio"]
```

## Dockerfile 约定（本仓特有的点）

- 装框架：`COPY framework /tmp/service-framework` +
  `uv pip install "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2`（**COPY，非 bind-mount**）。
- 模块路径用 `bioq_service`（若 Dockerfile 里有显式模块引用）。
- 上游源从 `services/<svc>-server/upstream/` COPY（**不在 image 内 git clone**）——先在 host 跑
  `vendor.sh`。
- 权重走 NAS `/data/models/<svc>/`，不烘焙进镜像。
- 构建上下文是**仓库根**：`docker build -f services/<svc>-server/Dockerfile .`（`.dockerignore` 已按此裁剪）。

Dockerfile 的两套骨架（uv venv / conda 多阶段）+ conda 踩坑，见 monorepo cookbook 的
`dockerfile.md` / `conda-pitfalls.md`。

## 验证（提交前跑一遍）

```bash
# 0. 先在 monorepo 写好设计文档（cookbook §0）

# 1. lint + 离线单测（本服务独立隔离；从服务目录跑）
cd services/<svc>-server
uvx ruff check .
uv run --group dev python -m pytest tests/test_app.py tests/test_cli.py -q

# 2. vendor 上游源（Docker build 前必跑）
./scripts/vendor.sh
ls upstream/ | head

# 3. (可选) 预下载权重到本地 stage
# ./scripts/fetch_weights.sh

# 4. 本地镜像构建（从仓库根）
cd ../..                       # 回仓库根
make build-<svc>-server        # Makefile 自动发现 services/*/Dockerfile

# 5. manifest / task 路由 sanity
docker run --rm -p 9000:9000 <svc>-server &
curl -s localhost:9000/api/manifest | jq .endpoints
curl -s localhost:9000/openapi.json | jq '.paths|keys[]' | grep /api/tasks/
kill %1
```

## 注册 + 网关联通

1. `services.yaml` 加一行 `<svc>-server:`（`url`、可选 `tier`/`function`/`gpu`；**有文件输入
   的服务加 `oss_mount: true`**）。未部署时可先加**注释条目**占位（照 `haddock3-server` / `lasermpnn-server`
   的写法）。
2. 经 gateway 调用的服务：在 `gateway/tests/test_fc.py` 加一个 `TestEndToEnd<Svc>` 类
   （照 `TestEndToEndProteinMPNN` 走 presign-file 输入 / `TestEndToEndMMseqs2` 走 inline 参数），断言
   download 是 302→OSS + `results.zip` 内含预期产物。

## 部署到 FC + 跑 FC 测试

部署步骤（FC 控制台异步任务模式 + OSS mount + NAS 权重挂载）见 monorepo cookbook 的 `deploy.md`。
部署后：

```bash
cd services/<svc>-server
RUN_FC_TESTS=1 uv run --group dev python -m pytest -m fc tests/test_fc.py -v
RUN_FC_TESTS=1 uv run --group dev python -m pytest -m fc tests/test_fc_task.py -v
```

## 提交清单

用 monorepo cookbook 页末的完整 checklist（设计文档 / vendor.sh / 权重外置 / healthz 探针 /
task endpoint / manifest_extras / services.yaml / gateway e2e 等）逐条勾。本仓这边只多两条：
import 名用 `bioq_service`、`upstream/` + `weights/` 已进 `.gitignore`。
