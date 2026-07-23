# bioq-services

AI 药物研发（AIDD）算法服务舰队 + 共享服务框架。每个 `services/<name>-server/` 把一个第三方
生物信息 / AIDD 工具包成**双模 Docker 镜像**：

- **HTTP 模式**（默认）：`uvicorn server.app:app` —— FastAPI + 异步 job runner，部署到阿里云 FC
- **CLI 批处理模式**：`python -m server <endpoint> ...` —— Slurm/sbatch 单次同步执行

> 本仓库由 `bioagent` monorepo 拆分独立而来（`git filter-repo --path services/`）。瘦客户端
> `bioq` CLI 在 **`bioq`** 仓库；研究知识库 + pipeline 编排在 **`bioagent`** 仓库。

## 结构

```
services/
├── _framework/        — 共享服务框架，PyPI 分发名 bioq-service-framework，import 名 bioq_service
│                        （JobAdapter / SubprocessRunner / CLIEndpoint / create_cli / uris 等）
└── <name>-server/     — 单个服务（gateway / dockq / boltz / rfdiffusion / ...）
    ├── server 包（app.py / models.py / tools.py / adapter.py / settings.py ...）
    ├── __main__.py    — CLI 批处理入口
    ├── tests/         — 离线单测（mock subprocess）+ FC 集成测试（@pytest.mark.fc，opt-in）
    ├── pyproject.toml — 离线测试/开发环境声明（见下）
    ├── Dockerfile / VERSION
    └── fc_<svc>.yaml  — 阿里云 FC 部署配置
```

各服务的**运行时重依赖（torch/cuda/conda/rdkit 等）在各自 Dockerfile 里**——它们互斥，故无法共用
一个环境（这正是每个服务单独 Dockerfile 的原因）。`pyproject.toml` 只声明**离线测试**所需的
pip 依赖 + 框架路径依赖。

## 依赖框架

每个服务通过相对路径依赖框架，无需发包：

```toml
[tool.uv.sources]
bioq-service-framework = { path = "../_framework", editable = true }
```

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
cd services/_framework && uv run --extra dev python -m pytest tests/ -q
```

> 少数服务的测试会读取 vendored 的 `upstream/`（git-ignore，未入库）。这类服务需先跑各自的
> `vendor.sh` 拉取上游源码，否则相关测试会因缺文件失败。

## 构建 / 推送镜像

`Makefile` 自动发现 `services/*/Dockerfile`，每个服务用自己的 `VERSION` 作镜像 tag（各自独立发版）：

```bash
make list                       # 列出发现的服务
make build-<service>            # 构建单个镜像（用其 VERSION）
make build-<svc> TAG=v0.0.5     # 覆盖 tag
make push-<service>             # 构建 + tag + 推送到 harbor（REGISTRY 可覆盖）
make bump-<service>             # patch 版本 +1
make sif-<service>             # Docker → Apptainer SIF（HPC/Slurm）
```

镜像构建上下文是仓库根（`docker build -f services/<svc>/Dockerfile .`），`.dockerignore` 已按此裁剪。

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
