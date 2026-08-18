# bioq-services

> English: [README.md](README.md)

AI 药物研发（AIDD）算法服务舰队 + 共享服务框架。每个 `services/<name>-server/` 把一个第三方
生物信息 / AIDD 工具包成**双模 Docker 镜像**：

- **HTTP 模式**（默认）：`uvicorn server.app:app` —— FastAPI + 异步 job runner，部署到阿里云 FC
- **CLI 批处理模式**：`python -m server <endpoint> ...` —— Slurm/sbatch 单次同步执行

## 结构

按**职责**分层：

```
framework/             — 共享服务框架（库，非服务，无 Dockerfile），PyPI 分发名 bioq-service-framework，
│                        import 名 bioq_service（JobAdapter / SubprocessRunner / CLIEndpoint / uris 等）
gateway/               — 控制面：认证 / 上传协商 / FC 异步调度（ECS，docker-compose 部署，见 deploy/ecs/）
edge/                  — 非 worker 边缘组件
├── jwt/               — JWT 签发辅助
└── protein-design-mcp/— MCP 协议适配器
services/              — 算力 worker（每个是一个 FC/OpenFaaS 函数镜像，保留 -server 后缀）
└── <name>-server/
    ├── server 包（app.py / models.py / tools.py / adapter.py / settings.py ...）
    ├── __main__.py    — CLI 批处理入口
    ├── tests/         — 离线单测（mock subprocess）+ FC 集成测试（@pytest.mark.fc，opt-in）
    ├── pyproject.toml — 离线测试/开发环境声明（见下）
    ├── Dockerfile / VERSION
    └── deploy/        — 平台部署描述子（fc.yaml；将来 openfaas.yaml / k8s.yaml）
services.yaml          — fleet registry（各服务 url / tier / function / 分类 tag）
```

各服务的**运行时重依赖（torch/cuda/conda/rdkit 等）在各自 Dockerfile 里**——它们互斥，故无法共用
一个环境（这正是每个服务单独 Dockerfile 的原因）。`pyproject.toml` 只声明**离线测试**所需的
pip 依赖 + 框架路径依赖。

## 依赖框架

每个服务通过相对路径依赖框架，无需发包：

```toml
[tool.uv.sources]
bioq-service-framework = { path = "../../framework", editable = true }
```

（`gateway/` 在顶层，用 `path = "../framework"`。）

## 测试（离线）

每个服务独立隔离运行，互不干扰：

```bash
cd services/<name>-server
uv run --group dev python -m pytest tests/ -q          # 离线单测
# FC 集成测试（需已部署的服务）：
RUN_FC_TESTS=1 uv run --group dev python -m pytest -m fc tests/ -v
```

框架自身：

```bash
cd framework && uv run --extra dev python -m pytest tests/ -q
```

> 少数服务的测试会读取 vendored 的 `upstream/`（git-ignore，未入库）。这类服务需先跑各自的
> `vendor.sh` 拉取上游源码，否则相关测试会因缺文件失败。

## 构建 / 推送镜像

`Makefile` 跨层自动发现（`services/*/Dockerfile` + `gateway/Dockerfile` + `edge/*/Dockerfile`；`framework/`
无 Dockerfile 故不入发现），每个服务用自己的 `VERSION` 作镜像 tag（各自独立发版）。镜像名 = 目录末段名
（worker 保留 `-server`，线上 FC 引用不变）：

```bash
make list                       # 列出发现的服务
make build-<service>            # 构建单个镜像（用其 VERSION）
make build-<svc> TAG=v0.0.5     # 覆盖 tag
make push-<service>             # 构建 + tag + 推送到 harbor（REGISTRY 可覆盖）
make bump-<service>             # patch 版本 +1
make sif-<service>             # Docker → Apptainer SIF（HPC/Slurm）
```

镜像构建上下文是仓库根（`docker build -f <svc-dir>/Dockerfile .`，`<svc-dir>` 由 Makefile 按名跨层解析），
`.dockerignore` 已按此裁剪。

## 本地启动（kind + OpenFaaS）

一条命令把整套控制面 + 选定的算力 worker 拉起在本地 kind 集群里，用于端到端联调。

**前置**：只需 `docker`；`kind` / `kubectl` / `helm` 会自动下载到 `$BIOQ_WORKDIR/bin`
（默认 `~/.cache/bioq-local`）。所有状态（kubeconfig、下载的工具、共享卷里的 `gateway.db` +
job 目录）都在 `BIOQ_WORKDIR` 下。

```bash
make local-up                                   # 起默认服务（dockq-server）
make local-up LOCAL_SERVICES="dockq-server plip-server"   # 指定要起的 worker
```

`local-up` 幂等地拉起：kind 集群 → OpenFaaS → bundled PostgreSQL → bioq gateway → 选定 worker，
并把 gateway 端口转发到 **`http://127.0.0.1:9000`**（默认 API key `bioq-local-secret`）。

### 常用命令

```bash
make local-status              # 看 pods / services
make local-logs LOCAL_SVC=gateway   # 跟踪日志（LOCAL_SVC=gateway 看 gateway，缺省看 dockq-server）
make local-info                # 打印 gateway URL / API key / kubeconfig / 共享目录
make local-forward             # 端口转发断了重新建立
make local-test                # 跑 dockq 功能测试打一遍本地部署
make local-user ACCOUNT=alice PASSWORD=pw          # 在 Keycloak 建普通用户
make local-user ACCOUNT=root  PASSWORD=pw ADMIN=1  # 建管理员（加入 bioq-admins 组）
make local-users               # 列出 Keycloak 用户 + bioq-admins 成员
make local-svc CLIENT=ci [ADMIN=1]  # 建/轮换机器账号（client-credentials，secret 默认 <client>-secret）
make local-svcs                # 列出 service-account clients
make local-down                # 拆掉部署（保留 BIOQ_WORKDIR 状态）
make local-purge               # 拆掉并清空 BIOQ_WORKDIR
```

### 认证（Keycloak / OIDC）

本地部署内置 **Keycloak**（realm `bioq`），鉴权走 OIDC/JWT（api key 已退役）。`BYPASS_VPC=true`：
localhost 免凭据（break-glass），非 localhost Host 需 OIDC token。
- **Keycloak**：`http://localhost:8081`（master 控制台 `admin`/`admin`；bootstrap 应用管理员 `admin`/`admin` 在 realm `bioq`、组 `bioq-admins`）。
- **用户/权限**：`make local-user ... [ADMIN=1]` 经 kcadm 建用户;角色由**组**决定（`bioq-admins` → gateway 里 `role=admin`，首次登录 JIT 落库）。
- **管理控制台 SSO**：开 `http://127.0.0.1:9000/admin/login` → 「Sign in with SSO」→ Keycloak 登录（admin/admin）→ 回控制台（localhost 也可免登录直接进）。
- **bioq CLI**（人类，device flow）：
  ```bash
  bioq --gateway-url http://127.0.0.1:9000 login --oidc \
       --issuer http://localhost:8081/realms/bioq --client-id bioq-cli
  bioq services       # 带 Bearer JWT → gateway 验（JWKS 集群内 / issuer=localhost:8081）
  ```
- **机器/CI**（client-credentials, service account）：`make local-svc CLIENT=ci [ADMIN=1]` 建一个 confidential client，然后
  ```bash
  export BIOQ_OIDC_CLIENT_SECRET=ci-secret        # 默认 <client>-secret，勿写进配置文件
  bioq --gateway-url http://127.0.0.1:9000 login --client-credentials \
       --issuer http://localhost:8081/realms/bioq --client-id ci
  bioq services       # 每次用 client_id+secret 现换 token（无人值守）
  ```
  realm 自带示例 `bioq-svc`（普通权限，secret `bioq-svc-secret`）可直接用。
- `BIOQ_KEYCLOAK=0` 关掉 Keycloak（则仅 localhost VPC bypass 可用，无 OIDC）。

> 关键机制:Keycloak 用 `KC_HOSTNAME=http://localhost:8081`(浏览器/bioq 可达的 frontend issuer)
> + `KC_HOSTNAME_BACKCHANNEL_DYNAMIC=true`(gateway pod 用集群 DNS 取 token/jwks);一个 issuer、两条可达路径。

### 改了代码后重新部署

`make local-up` **不会重建已存在的镜像**（复用本地 `<svc>:latest` / `gateway:latest`）。改了代码要生效：

```bash
make local-up BIOQ_BUILD=always            # 强制重建所有镜像（worker + gateway）再重部署，较慢

# 只更新 gateway（改了 gateway/ 代码时更快）：重建 → load 进 kind → 重启
make build-gateway
export KUBECONFIG=$HOME/.cache/bioq-local/kubeconfig PATH="$HOME/.cache/bioq-local/bin:$PATH"
kind load docker-image gateway:latest --name bioq
kubectl -n bioq rollout restart deploy/bioq-gateway
```

### 可调环境变量

`BIOQ_WORKDIR`（状态目录）、`BIOQ_API_KEY`、`BIOQ_GATEWAY_PORT`、`BIOQ_CLUSTER`（kind 集群名，默认 `bioq`）、
`BIOQ_BUILD`（`auto|always|never`）、`BIOQ_DB_BACKEND`（`postgres|sqlite`）、
`BIOQ_KEYCLOAK`（`1` 默认内置 Keycloak；`0` 关闭退回 api-key）、`BIOQ_KC_PORT`（默认 8081）、
`BIOQ_DOCKERHUB_MIRROR`（Docker Hub 镜像加速）。细节见 `deploy/openfaas/local-up.sh` 头部注释。

## 添加新服务

**先读 [`docs/adding-a-service.md`](docs/adding-a-service.md)** —— repo-local 落地指南（代码放哪、
`bioq_service` 命名、vendor/测试/构建/注册的具体命令）。

完整流程与设计原则（骨架 / Dockerfile 两套 / conda 踩坑 / 测试骨架 / FC 部署 / 提交 checklist）在
`bioagent` 仓库的 `engineering/guides/adding-a-new-service/` cookbook + `engineering/CONVENTIONS.md`
（跨仓库文档）。

> 分工：服务**代码**只在本仓开发；**设计文档**（`engineering/decisions/...-design.md`）留在
> `bioagent` monorepo。详见 [`docs/adding-a-service.md`](docs/adding-a-service.md)。

## 相关仓库

- **`bioq`** —— 瘦客户端 CLI（gateway REST 客户端）
- **`bioagent`** —— 研究知识库（`wiki/`）、pipeline 编排（`pipelines/`）、工程文档（`engineering/`）