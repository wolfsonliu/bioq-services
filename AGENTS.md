# AGENTS.md — bioq-services

面向在本仓库内做功能开发的 agent。本文件自包含：覆盖仓库结构、核心心智模型、框架 API、
硬性约定、测试、构建、部署与新增服务的完整流程。**所有路径都相对本仓库根**，不要引用仓库
之外的相对路径。

---

## 1. 这是什么仓库

AI 药物研发（AIDD）**算法服务舰队 + 共享服务框架**。每个 `services/<name>-server/` 把一个
第三方生物信息 / AIDD 工具包成一张**双模 Docker 镜像**：

- **HTTP 模式**（默认）：`uvicorn server.app:app` —— FastAPI + 异步 job runner，部署到阿里云 FC；
- **CLI 批处理模式**：`python -m server <endpoint> ...` —— Slurm/sbatch 单次同步执行。

两种模式共用同一套 `tools.py`（argv 构造）、`adapter.py`（输出检测）、`settings.py`（配置）。
共享的通用层（HTTP / job 生命周期 / 错误处理 / 持久化 / manifest / CLI / 上传下载）全部由
`framework/` 提供，**不要在一个 service 里重新造这些轮子**。

发布物：每个 worker / gateway / edge 组件一张镜像；`framework/` 是库，无 Dockerfile，不发镜像。

---

## 2. 仓库布局（按职责分层）

```
framework/             — 共享服务框架（库，无 Dockerfile）。
│                        PyPI 分发名 bioq-service-framework，import 名 bioq_service。
│                        核心 API：JobAdapter / ServiceSettings / create_app / JobRunner /
│                        CLIEndpoint·create_cli / execute_task·register_task_endpoint /
│                        resolve_input·resolve_uri / model_form_depends 等。
gateway/               — 控制面：认证 / 上传协商（OSS presign）/ 异步任务调度 / 状态与下载代理
│                        （ECS，docker-compose / kind 部署）。自身也是被 `bioq_service` 支撑的
│                        服务（继承 ServiceSettings，dispatch backend 可切换）。
edge/                  — 非 worker 边缘组件
│   ├── jwt/           —   JWT 签发辅助
│   └── protein-design-mcp/ — MCP 协议适配器（SSE + Streamable HTTP）
services/              — 算力 worker（每个是一张 FC/OpenFaaS 函数镜像，保留 -server 后缀）
│   └── <name>-server/ — 见 §4 服务解剖学
deploy/                — 部署目标：ecs/（生产 ECS + FC）、compose/（本地全栈）、
│                        openfaas/（本地 kind + OpenFaaS）、config/（生成的共享非密配置）
docs/                  — adding-a-service.md（repo-local 落地指南）+ adding-a-new-service/
│                        （cookbook，按主题拆子页）+ specs/（设计文档）
services.yaml          — fleet registry（已部署服务的 url / tier / function / oss_mount 等）
Makefile               — 镜像构建/推送/bump/SIF + 本地 kind 联调（make local-*）
scripts/               — 杂项脚本（如 bench_concurrency.py）
```

**worker 清单（38 个）**，按命名：
alphafold / bindflow / boltz / boltzgen / chembounce / deeprank-ab / diamond / diffdock /
diffdock-pp / diffusion-hopping / dockq / drughive / ensemble / esmfold2 / flowmol / genie3 /
haddock3 / iggm / immunebuilder / lasermpnn / lightdock / megalodon / mmseqs2 / odesign /
openadmet / openbpmd / plip / pocketxmol / ppiflow / promera / proteinmpnn / qligfep / reinvent /
rfantibody / rfdiffusion / rfdiffusion2 / semlaflow / turbohopp。

已部署服务的权威清单在 `services.yaml`（含 `tier`: `hot` 常暖 / `warm` 缩容到零 / `cold` 批处理；
有文件输入的服务带 `oss_mount: true`）。未部署服务在 `services.yaml` 里以**注释条目**占位。

---

## 3. 心智模型

- **一个 service = 一张镜像 + 一组 HTTP endpoint + 一个 CLI 批处理入口。**
- 每个 service 的**运行时重依赖（torch/cuda/conda/rdkit 等）在各自 Dockerfile 里**，互斥、
  无法共用一个环境 —— 这正是每个 service 单独 Dockerfile 的原因。`pyproject.toml` 只声明
  **离线测试**所需的轻量 pip 依赖 + 框架 path source。
- 每个 service 通过相对路径依赖框架、无需发包：
  ```toml
  [tool.uv.sources]
  bioq-service-framework = { path = "../../framework", editable = true }
  ```
  （`gateway/` 在顶层，用 `path = "../framework"`。）
- **job 目录是所有状态的单元**：一个 job 在 `<jobs_base_dir>/<job_id>/` 下，含
  `input/`（self-contained 输入）、`output/`（产出，`/download` 打包此目录）、
  `logs/run.log`（子进程日志）、`job.json`（状态 sidecar，用于重启恢复）。
- **两种调用范式**（同一镜像内并存）：
  1. **submit/poll**：POST 立即返回 `job_id`，后台 `ThreadPoolExecutor` 跑子进程，客户端轮询
     `GET /api/jobs/{id}`。适合无需长期占用实例的场景。
  2. **task endpoint**（`/api/tasks/<name>`）：POST **阻塞到子进程跑完**才返回，专为 FC
     异步任务模式（`X-Fc-Invocation-Type: Async`）设计，让 FC 实例在整段计算期间保持占用、
     不被空闲回收。现代 GPU 服务的首选入口。两者成对提供（见 §5 约定）。

---

## 4. 服务解剖学（`services/<svc>-server/`）

必备文件：

```
services/<svc>-server/
├── __init__.py          # 包标记，通常空
├── app.py               # create_app + 服务专属 endpoint + /healthz/detail + task endpoint
├── __main__.py          # CLI 批处理入口（python -m server <endpoint>）
├── adapter.py           # JobAdapter 子类（name + detect_outputs + manifest_extras + endpoint_examples）
├── settings.py          # ServiceSettings 子类（env_prefix=<SVC>_）；weights_dir 默认 /data/models/<svc>/
├── models.py            # 请求 pydantic models
├── tools.py             # argv builders（可选，视复杂度）
├── Dockerfile           # COPY framework + services/<svc>/upstream/ + 算法栈
├── pyproject.toml       # 离线测试/开发依赖（仅 uv venv 骨架需要；部分服务无此文件）
├── README.md            # endpoint / 配置 / 部署说明 + «Weights» 章节
├── VERSION              # 镜像 tag（Makefile 读取，如 "v0.0.1"）
├── scripts/
│   ├── vendor.sh        # 必备：clone 上游源码到 upstream/ at pinned SHA
│   └── fetch_weights.sh # 可选：预下载权重到 weights/ 或直接 NAS（WEIGHTS_DST 覆盖）
└── tests/
    ├── __init__.py / conftest.py   # server 模块注册（importlib）+ fc marker
    ├── test_app.py      # 离线 TestClient 单测
    ├── test_cli.py      # CLI 批处理单测
    ├── test_fc.py       # FC sync 集成测试（@pytest.mark.fc，默认 skip）
    ├── test_fc_task.py  # FC 异步 task 模式测试（默认 skip）
    └── data/            # 测试 fixture（小 PDB/JSON 等）
```

`upstream/`（vendor.sh 产物）与 `weights/`（fetch_weights.sh 产物）**不入库**（.gitignore）。

**最省事的起点：把一个结构相近的现有服务整目录拷贝改名，再逐文件改。** 参考实现：

| 场景 | 参考服务 |
|---|---|
| uv venv + 序列设计 + 权重外置的最简骨架；vendor.sh 单上游标准示范 | `services/proteinmpnn-server/` |
| CPU-only uv-venv slim | `services/dockq-server/`、`services/diamond-server/` |
| conda/micromamba 多阶段 | `services/deeprank-ab-server/`、`services/pocketxmol-server/` |
| manifest_extras + endpoint_examples 完整示范 | `services/rfantibody-server/`、`services/genie3-server/` |
| 多 endpoint + config YAML 驱动 | `services/genie3-server/`、`services/drughive-server/` |
| vendor.sh 多 upstream / symlink 权重 | `services/promera-server/` |
| fetch_weights.sh + 大镜像瘦身示范 | `services/boltzgen-server/` |

---

## 5. 框架 API 速查（`framework/src/bioq_service/`）

import 名固定为 `bioq_service`（`from bioq_service import ...`），分发名 `bioq-service-framework`。

### JobAdapter（`adapter.py`）

一个 service 一个 `JobAdapter` 子类，承载**全 service 一致**的策略（文件布局、输出检测、日志、
子进程 env/cwd、重启恢复）；它不感知具体请求 shape / argv（那是 per-endpoint 的事）。可覆写钩子：

| 钩子 | 默认 | 用途 |
|---|---|---|
| `name: str` | （必填） | service 名，`JobInfo.service` 用它 |
| `job_dir(job_id)` | `<jobs_base_dir>/<job_id>` | job 工作目录 |
| `output_dir(job_dir)` | `<job_dir>/output` | `/download` 打包、`/files` 列出的目录 |
| `log_path(job_dir)` | `<job_dir>/logs/run.log` | 子进程 stdout+stderr tee 位置 |
| `detect_outputs(job_dir)` | `output/` 非空 | rc==0 但这里返回 False → 标记 FAILED（failure_kind=NO_OUTPUTS）。多端点服务应覆写以识别各工具产物 |
| `subprocess_env()` | `{}` | 注入子进程的额外环境变量 |
| `subprocess_cwd()` | None | 子进程工作目录 |
| `infer_job_from_dir(job_dir)` | 有产出→COMPLETED | 无 `job.json` 的遗留目录的恢复启发式 |
| `manifest_extras()` | `{}` | **强烈建议覆写**：至少给 `tool_outputs` + `input_uri_schemes`（输出文件命名约定、支持的 URI scheme、chaining 提示、config 坑） |
| `endpoint_examples()` | `{}` | **强烈建议覆写**：每个 endpoint 至少一个可复制即跑的 curl（最好加 python snippet） |

### ServiceSettings（`settings.py`）

所有运行时配置走 `pydantic_settings.BaseSettings` 子类；**框架和 adapter 里禁止 `os.getenv`**。
每个 service 设自己的 `env_prefix`：

```python
class DockQSettings(ServiceSettings):
    model_config = SettingsConfigDict(env_prefix="DOCKQ_", env_file=".env", extra="ignore")
```

基类关键字段（env `<PREFIX>_*`，默认值见 `framework/src/bioq_service/settings.py`）：
`jobs_base_dir`（默认 `/data/jobs`）、`uploads_base_dir`（`/data/uploads`）、`oss_output_mount`
（`/mnt/oss`）、`oss_region`（`cn-hangzhou`）、`disk_limit_mb`（`8000`，超过则驱逐已结束 job）、
`port`（`9000`）、`max_concurrent_jobs`（`2`，超出返回 503）、`keep_alive_sec`、
`keepalive_interval_s` / `keepalive_url`（FC 自保活）、`session_header_name`（FC 会话亲和）、
`task_endpoints_enabled`（默认 True）、`task_job_id_header`（默认 `X-Bioagent-Job-Id`）。

### 组集 FastAPI app（`app.py`）

```python
settings = DockQSettings()
adapter = DockQAdapter(settings=settings)
app = create_app(adapter, settings, title="...", version=read_version_file(__file__))
```
`create_app` 自动挂上通用路由（healthz / manifest / openapi / jobs 系列），并在 `app.state` 暴露
`adapter` / `settings` / `job_store` / `runner`。service 侧只需加自己的 POST 端点，端点里：
- 收表单参数用 `params: Model = Depends(model_form_depends(Model))`；
- 触发 submit/poll 用 `app.state.runner.submit(build_argv=_build, label="...", input_params=...)`；
- 最后（在所有 POST 路由注册之后）调用 `attach_mcp(app)` 把 HTTP 面镜像到 `/mcp`（可选，
  `bioq-service-framework[mcp]` extra）。

`read_version_file(__file__, default=...)` 读同目录 `VERSION`、去掉前导 `v`，保证 HTTP version
与镜像 tag 不漂移。

### Task endpoint（`task_endpoint.py`）

- `resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)` —— 取 header 优先、否则 UUID。
- `execute_task(request, *, job_id, label, params, build_argv, save_inputs=None, oss_prefix=None)`
  —— 同步跑完整管线（幂等去重、磁盘驱逐、PENDING→RUNNING→finalize、失败 200+FAILED、成功/失败都
  尝试 OSS output-sink）。有 `UploadFile` / 自定义 Form 的端点用它并自写 handler。
- `register_task_endpoint(app, *, path, label, request_model, build_argv, save_inputs=None)`
  —— 无上传场景的便捷封装。

**注意**：该模块**故意**不用 `from __future__ import annotations`（PEP 563 字符串注解会破坏
FastAPI 对运行时类的 `get_type_hints`）。改这个文件时不要补这个 future import。

### CLI 批处理（`cli.py`）

每个 service 的 `__main__.py`：
```python
endpoints = {"score": CLIEndpoint(name="score", help="...", request_model=ScoreRequest,
                                  build_argv=_score_build, inputs={"model": ("help", True)})}
create_cli(adapter, settings, endpoints, version="0.0.1")
```
`CLIEndpoint.inputs` 的每个键变成一个 `--<name>` 本地文件路径旗标；`build_argv(req, inputs, job_dir, settings)`
是薄薄一层到 `tools.py` 的回调。`create_cli` 从 pydantic model 自动生成 argparse flags、解析
输入文件、跑子进程、按 rc + `detect_outputs` 决定退出码。

### 输入解析（`uris.py`）

所有 service 统一一套 scheme（agent 学一次即可）：
| scheme | 语义 |
|---|---|
| multipart upload | 客户端直接传文件 |
| `job://<id>/<file>` | 从**同一 NAS** 上前一个 job 的 `output/` 拉取（chaining） |
| `file:///abs/path` 或裸 `/abs/path` | 复制 NAS 本地路径 |
| `oss://<bucket>/<key>` | 走 OSS SDK 下载（需 `alibabacloud-oss-v2`） |
| `http(s)://...` | 流式拉远程 URL（含 OSS 签名 URL） |

API：`resolve_input(upload, input_uri, dest, settings, field_name=None)`（两者都给则 URI 优先；
都缺 → 422；`field_name` 用在 422 详情里定位哪个字段缺）、`maybe_resolve_input(...)`（都缺返回
None，给有内联 SMILES/序列等替代表示的场景）、`resolve_uri(uri, dest, settings)`、
`save_upload(upload, dest)`。

### 通用 endpoint（框架自动注册，见 `framework/README.md`）

`GET /healthz`、`GET /healthz/detail`、`GET /api/manifest`、`GET /openapi.json`、
`GET /api/jobs/{id}`、`GET /api/jobs/{id}/files`、`GET /api/jobs/{id}/log`、
`GET /api/jobs/{id}/download`、`GET /api/jobs/{id}/file/{path}`、`DELETE /api/jobs/{id}`。

---

## 6. 硬性约定（改 service 时必须遵守）

- **命名**：import 名 `bioq_service`；分发名 `bioq-service-framework`；console script
  `bioq-service-mcp-stdio`。HTTP header（`X-Bioagent-*`）、`X-Bioagent-Job-Id`、
  `X-Bioagent-Oss-Prefix` 等**历史契约保持不变**（framework task endpoint 默认读它们）。
- **类型**：所有请求/响应类型都是 `pydantic.BaseModel`；**禁止 `dict[str, Any]`** 作为请求体。
- **配置**：所有运行时配置走 `pydantic-settings`；`settings.py` / 框架 / adapter 里**无 `os.getenv`**。
- **manifest**：`manifest_extras()` 至少提供 `tool_outputs` + `input_uri_schemes`；
  `endpoint_examples()` 每个 endpoint 至少一个即跑的 curl。
- **task endpoint 成对**：每个 submit/poll endpoint 都要有对应 `/api/tasks/<name>` task endpoint
  （无上传用 `register_task_endpoint`，有上传用 `execute_task`），且自定义 task 端点要塞在
  `if settings.task_endpoints_enabled:` 守卫内。
- **URI 字段命名（高频坑）**：客户端 `--file <field>=<path>` 约定把上传字段映射为请求体里的
  `<field>_uri`。因此「文件上传 / URI 二选一」的输入，其 URI 表单字段名**必须严格等于
  `<upload_field_name>_uri`**（例如上传字段 `model` → URI 字段 `model_uri`；`input_pdb` →
  `input_pdb_uri`；`custom_ff_zip` → `custom_ff_zip_uri`）。命名错配会让客户端生成的字段被
  FastAPI 丢弃 → 两侧 `upload=None, uri=None` → 422。排查/回归见
  [`docs/specs/2026-08-18-cross-service-uri-field-naming-design.md`](docs/specs/2026-08-18-cross-service-uri-field-naming-design.md)。
- **权重外置**：权重放 NAS `/data/models/<svc>/`，不烘焙进镜像（确需烘焙小权重 < 100 MB 要在注释
  说明理由）。`settings.py` 的 `weights_dir` default 指向该 NAS 路径；`app.py` 自定义
  `/healthz/detail` 带 `weights_loaded` / `weights_missing` 字段，缺权重时 HTTP 200 +
  `weights_loaded=false`，**不在 import 期 raise**。
- **vendor 上游**：Dockerfile 从 `services/<svc>-server/upstream/` COPY 上游源码（先在 host 跑
  `scripts/vendor.sh`，`git clone ... at pinned SHA` + 重试 + 校验），**不在 image 内 git clone**、
  **无 `COPY opensource/`**。装框架用 `COPY framework /tmp/service-framework`（**COPY，非
  bind-mount**，否则 output-sink 修复进不去镜像）+ `pip install`（或 `uv pip install`）。
- **版本**：每 service 独立发版，tag 来自各自 `VERSION`（`make bump-<svc>`），从不全局协调。
- **endpoint 收参**：endpoint 用 `Depends(model_form_depends(Model))` 接收表单参数，不是裸
  `params: Model`。

完整逐条 checklist 见 [`docs/adding-a-new-service/index.md`](docs/adding-a-new-service/index.md#提交清单在-pr-描述里勾掉)
（conda 服务的 LANG / yaml 剥离 / dead-import stub 另见
[`docs/adding-a-new-service/conda-pitfalls.md`](docs/adding-a-new-service/conda-pitfalls.md)）。

---

## 7. gateway（控制面）

`gateway/` 是常驻 API gateway（ECS / compose / kind），前接下游 FC/HTTP/OpenFaaS 服务。入口
`python -m server`（容器内）/ `python -m gateway`（仓库内），FastAPI app 在 `gateway/app.py`，
配置 schema 单一来源 `gateway/settings.py`（`GatewaySettings`，`env_prefix="GATEWAY_"`，
嵌套用 `GATEWAY_AUTH__...`）。

**端点（`/v1/*`，见 `gateway/README.md`）**：`GET /v1/services`、`GET /v1/services/{svc}`、
`POST /v1/run/{svc}/{endpoint}`、`GET /v1/jobs/{job_id}`、`GET /v1/jobs/{job_id}/download`、
`POST /v1/jobs/{job_id}/cancel`、`POST /v1/uploads/prepare`、`GET /healthz`。

**认证**：两层 —— VPC bypass（localhost / 内网 break-glass）→ OIDC/JWT（`Authorization: Bearer`）。
api key 已退役；人类走 OIDC device flow / SSO，机器走 OIDC client-credentials。用户 JIT 落库，
role 由 token 的 groups claim 推导（`GATEWAY_AUTH__JWT_ADMIN_GROUP`，默认 `bioq-admins` → admin）。
生产必须设 `GATEWAY_AUTH__JWT_ISSUER` 并保持 `bypass_vpc=false`。管理控制台在 `/admin`
（SSO 登录，CSRF 防护）。

**调度后端**：`GATEWAY_DISPATCH_BACKEND` = `fc`（阿里云 FC 异步任务模式，经 FC OpenAPI
`GetAsyncTask` 轮询，AK/SK 来自 `ALI_AK`/`ALI_SK`）/ `http`（compose，直接 submit/poll 每个
service 的 in-process runner）/ `openfaas`（kind + OpenFaaS）。dispatcher 协议在
`gateway/dispatchers/`（`base.py` + `fc.py` / `local.py` / `openfaas.py`）。

**存储**：`GATEWAY_STORAGE_BACKEND` = `oss`（presign 直传对象，bucket 默认 `bioagent-inputs`）
或 `file`（`/v1/files` 走共享卷 `GATEWAY_FILE_BASE_DIR`）。

**数据库 / 迁移**：用户+job 存储走 SQLAlchemy，schema 由 **Alembic** 管（非 `create_all()`），
容器 entrypoint 先 `alembic upgrade head` 再 uvicorn。本地 sqlite
（`GATEWAY_DB_URL=sqlite:///<path>`），生产 Postgres
（`postgresql+psycopg://...`）。加 schema 变更：
```bash
cd gateway && GATEWAY_DB_URL=sqlite:///$PWD/gw.db uv run alembic revision --autogenerate -m "<change>"
```

**部署配置**：非密拓扑在 checked-in 的 `deploy/config/gateway.<target>.env`（`make gen-config`
从 schema 生成；`make check-config` 做 CI drift 门禁），secrets 只在各 target 的 gitignored `.env`。
gateway 启动时校验配置，致命误配拒绝 boot。

---

## 8. 测试

每个 service 独立隔离。**先读各 service 的 `README.md` 与测试用例**再动手。命令（从对应目录跑）：

```bash
# 单个 service 的离线单测（service 目录内）
cd services/<svc>-server
uv run --group dev python -m pytest tests/ -q

# FC 集成测试（需已部署；默认 skip，用 env 或 -m fc 打开）
RUN_FC_TESTS=1 uv run --group dev python -m pytest -m fc tests/ -v

# framework 自身
cd framework && uv run --extra dev python -m pytest tests/ -q

# gateway（仓库根）
uv run python -m pytest gateway/tests/ -v

# lint
uvx ruff check services/<svc>/
```

少数服务的测试会读取 vendored 的 `upstream/`（git-ignore，未入库）——这类服务需先跑各自的
`scripts/vendor.sh`，否则相关测试缺文件失败。

---

## 9. 构建 / 部署镜像

`Makefile` 跨层自动发现（`services/*/Dockerfile` + `gateway/Dockerfile` + `edge/*/Dockerfile`；
`framework/` 无 Dockerfile 故不发现）。镜像名 = 目录末段名；构建上下文是**仓库根**
（`docker build -f <svc-dir>/Dockerfile .`）。

```bash
make list                     # 列出被发现的服务
make build-<service>          # 用其 VERSION 构建
make build-<svc> TAG=v0.0.5   # 覆盖 tag
make push-<service>           # 构建 + tag + push 到 harbor（REGISTRY 可覆盖）
make bump-<service>           # patch 版本 +1
make sif-<service>            # Docker → Apptainer SIF（HPC/Slurm）
```

**新增/改动 service 后的验证与部署**：照
[`docs/adding-a-new-service/index.md`](docs/adding-a-new-service/index.md) 的「验证检查清单」跑
（vendor → docker build → `/api/manifest` sanity → `python -m server --help` → task 路由 sanity），
部署到 FC 后跑 test_fc / test_fc_task。

---

## 10. 本地联调（kind + OpenFaaS）

一条命令把控制面 + 选定 worker 拉起在本地 kind 集群（前置只需 `docker`；`kind`/`kubectl`/`helm`
自动下载到 `$BIOQ_WORKDIR/bin`，默认 `~/.cache/bioq-local`）：

```bash
make local-up                                   # 默认 dockq-server
make local-up LOCAL_SERVICES="dockq-server plip-server"
make local-status / local-logs / local-info / local-test
make local-user ACCOUNT=alice PASSWORD=pw [ADMIN=1]   # Keycloak 建用户
make local-down / local-purge
```
gateway 端口转发到 `http://127.0.0.1:9000`（localhost = VPC bypass 免凭据）。认证走内置 Keycloak
（`http://localhost:8081`，realm `bioq`），api key 已退役。**`make local-up` 不重建已存在镜像**；
改了代码要 `make local-up BIOQ_BUILD=always`，或只重建 gateway（见仓库 `README.md`）。

可调 env：`BIOQ_WORKDIR` / `BIOQ_GATEWAY_PORT` / `BIOQ_CLUSTER` / `BIOQ_BUILD` /
`BIOQ_DB_BACKEND` / `BIOQ_KEYCLOAK` / `BIOQ_DOCKERHUB_MIRROR` 等，细节见
`deploy/openfaas/local-up.sh` 头部注释。

---

## 11. 新增一个服务（流程）

1. **先写设计文档**（开工前必做）：一份 `YYYY-MM-DD-<svc>-server-design.md`，必备章节见
   [`docs/adding-a-new-service/index.md`](docs/adding-a-new-service/index.md) §0（概述 / 设计目标 /
   Endpoint 拓扑 / 请求 Schema / 输出树 / 实现要点 / 配置 / 部署目标 / 测试策略 / 风险 / Sources）。
2. **照 cookbook 起骨架**：子页
   [`skeleton.md`](docs/adding-a-new-service/skeleton.md)、[`dockerfile.md`](docs/adding-a-new-service/dockerfile.md)、
   [`conda-pitfalls.md`](docs/adding-a-new-service/conda-pitfalls.md)、[`testing.md`](docs/adding-a-new-service/testing.md)、
   [`deploy.md`](docs/adding-a-new-service/deploy.md)；
repo-local 落地步骤（命名 / 必备文件 / pyproject / Dockerfile 约定 / 注册 / 提交清单）见
[`docs/adding-a-service.md`](docs/adding-a-service.md)。
3. **注册 + 联调**：`services.yaml` 加 `<svc>-server:` 条目（有文件输入加 `oss_mount: true`）；
经 gateway 调用的服务在 `gateway/tests/test_fc.py` 加 `TestEndToEnd<Svc>` e2e 类。
4. 过一遍 §6 的硬性约定与提交 checklist。

---

## 12. 其它规范

- 文档/注释用中文（与仓库现有 README / docs 一致）；代码、标识符、commit message 用英文。
- git commit 不要加 `Co-Authored-By` 之类 AI co-author trailer。
- 改动影响协议（端点签名 / 上传字段 / manifest）时，同步更新该 service 的 `README.md` 与
  `endpoint_examples()`，并视情况补一条 manifest 回归测试（范例见 rfantibody 的
  `test_quiver_uri_field_matches_upload_field` 思路）。
- 需要看框架完整行为时，直接读 `framework/src/bioq_service/`（每个模块 docstring 都详尽）与
  相应 `framework/tests/`。