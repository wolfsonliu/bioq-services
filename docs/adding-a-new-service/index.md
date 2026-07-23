# 新增 service（cookbook）

日期: 2026-05-29（持续更新；2026-07-14 按主题拆分为子页；2026-07-22 随 services 迁入 bioq-services 并自包含化）
适用: 把一个新的开源算法 / 模型包装成 FC 上跑的 HTTP 服务，同时支持 Slurm sbatch 批处理
相关: 见页末「相关」

新 service = 一个 Docker 镜像 + 一组 HTTP endpoint + CLI 批处理入口。
部署到 FC 上由 agent / pipeline 通过 HTTP 调用；在 Slurm 集群上通过 `python -m server` 作为 sbatch 任务执行。
**强制基于 [services/_framework](../../services/_framework/) 构建** —— 不要再造 HTTP /
job 生命周期 / 错误处理 / 持久化 / manifest / CLI 这些通用层。

本页是单一权威索引；如果 `_framework/README.md`、设计文档、调用指南之间有冲突，以本 cookbook 为准。

## 关于本指南

本指南是在 **本仓库（`bioq-services`）** 里新增一个服务的权威流程文档：从骨架 → Dockerfile →
测试 → 部署。下文所有 `services/<svc>/` 路径都是**本仓库根**下的树。

**命名约定：** 框架 import 名 **`bioq_service`**、分发名 **`bioq-service-framework`**、console script
`bioq-service-mcp-stdio`。HTTP header（`X-Bioagent-*`）、FC session header `bioagent-session-id` 等**保持
原样**（历史契约，不改）。

**设计文档（§0，开工前必做）** 是独立于代码的产物。本指南只规定它**必须包含的章节**（见 §0），
不规定存放位置——按你团队的 decisions/ 流程归档。

## 子页导航

本 cookbook 按主题拆成子页，本页保留总览（设计文档先行 + 必备文件清单 + 验证/提交清单）：

| 子页 | 内容 |
|------|------|
| [skeleton](./skeleton.md) | 5 分钟 echo skeleton：`__init__` / `settings` / `models` / `adapter` / `app.py`（含 task endpoint）/ `__main__` / `pyproject` / `VERSION` / `README` |
| [dockerfile](./dockerfile.md) | Dockerfile：`vendor.sh` / `fetch_weights.sh` / uv + conda 骨架 / wrapper vs patch 决策 |
| [conda-pitfalls](./conda-pitfalls.md) | 包装 conda-based upstream 的常见陷阱（LANG / yaml override / dead-import stub / uv git+ 等） |
| [testing](./testing.md) | `test_app` / `test_cli` / `test_fc` / `test_fc_task` 测试骨架 |
| [deploy](./deploy.md) | 部署到 FC + 控制台配置（异步任务模式 + OSS mount）+ 并发能力测试 |

## 0. 先写设计文档（**开工前必做**）

新 service 落地**先出一份 `YYYY-MM-DD-<svc>-server-design.md` 设计文档**（归档到你团队的 decisions/
流程），然后再动代码。这不是仪式感——本 guide 是"HOW 起骨架"，设计文档是"WHY 这么设计 + 与项目其它
service 的边界"，两者互补缺一不可。

**设计文档写作规范**：H1 标题 + 元数据行（日期 / 状态 / 适用 / 相关）+ 下表的必备章节；写完后在你的
decisions 索引里加一行链接。

新 service 设计文档的**必备章节**（可从最近的样板 clone 后改）：

| 章节 | 关键内容 |
|---|---|
| 概述 | 上游模型/算法定位、在 bioagent pipeline 中扮演什么角色、与已有 service 的边界 |
| 设计目标 | 3-6 条要落地的原则（单/多 endpoint、包装 vs patch、权重外置策略、conda vs uv、CLI 双模式对齐等） |
| Endpoint 拓扑 | 每个 endpoint 一句话；**v0.0.1 明确不做的**要列出（避免后续 scope creep） |
| 请求 Schema | pydantic 字段表格：字段/类型/默认/约束/说明；文件上传字段单列（不放 model） |
| 输出 | `<jobs_base_dir>/<job_id>/` 树形结构 + 每个文件的语义（agent 消费时需要知道） |
| 实现要点 | 包装/patch 决策（用下文表格）+ 数据布局 shim（如需要）+ argv 构造 + `detect_outputs()` 判据 + conda env pin + `/healthz/detail` 探针项 |
| 配置 | `env_prefix` + `Settings` 字段表 |
| 部署目标 | FC 实例规格 / 超时 / 内存 / NAS 挂载 / 是否启用异步任务模式 |
| 测试策略 | offline HTTP / offline CLI / FC sync / FC async 四张表 + fixture 来源 |
| 风险 / 限制 | build-time gotcha、依赖 pin 敏感度、GPU 世代兼容性、与其它 service 的功能重叠 |
| Sources | 论文引用、上游 URL + pinned SHA、fixture 出处、参考的其它服务设计 |

**样板文档**（按结构相似度选一个 clone）：

| 场景 | 参考 |
|---|---|
| pytorch + PyG + conda 骨架（GPU 扩散/等变网络） | diffusion-hopping-server 设计 |
| conda + ESM/AF cache + monkey-patch upstream | deeprank-ab-server 设计 |
| uv venv + 单/多变体 endpoint | proteinmpnn-server 设计 |
| 多 endpoint + config YAML 驱动 | genie3-server / drughive-server 设计 |
| CPU 服务 + 大型外部资源（DB） | chembounce-server 设计 |

写完设计文档并同步 index 后，再回到本 guide 从"必备文件清单"起做实现。

## 必备文件清单

```
services/<svc>/
├── __init__.py              # 包标记，通常空
├── app.py                   # create_app + 服务专属 endpoints + /healthz/detail 权重探针（如走 NAS）
├── __main__.py              # CLI 批处理入口（python -m server <endpoint>）
├── adapter.py               # JobAdapter 子类（name + manifest_extras + endpoint_examples）
├── settings.py              # ServiceSettings 子类（env_prefix=<SVC>_）；weights_dir 默认指向 /data/models/<svc>/
├── models.py                # 请求 pydantic models
├── Dockerfile               # COPY services/_framework + services/<svc>/upstream/ + 算法栈
├── pyproject.toml           # 包依赖（可选——仅 uv venv 骨架 + pip install -e . 时需要）
├── README.md                # endpoint / 配置 / 部署说明 + Weights 章节（NAS / SIF --bind）
├── VERSION                  # 镜像 tag，Makefile 读取（如 "v0.0.1"）
├── scripts/
│   ├── vendor.sh            # 必备：clone 上游源码到 upstream/ at pinned SHA（详见下文）
│   └── fetch_weights.sh     # 可选：下载模型权重到 weights/ 或直接 NAS（支持 WEIGHTS_DST env 覆盖）
├── upstream/                # gitignored：vendor.sh 产物，Docker build 从这里 COPY 上游源
├── weights/                 # gitignored：fetch_weights.sh 产物，stage 用，正式部署上传到 NAS
└── tests/                   # pytest 测试
    ├── __init__.py
    ├── conftest.py          # server 模块注册（importlib.util）+ fc marker
    ├── test_app.py          # 离线 TestClient 单测（health / manifest / 一个端点）
    ├── test_cli.py          # CLI 批处理单测（endpoint 注册 / argv builder / create_cli）
    ├── test_fc.py           # sync submit/poll 路径的 FC 集成测试（marker=fc，默认 skip）
    ├── test_fc_task.py      # /api/tasks/<name> 异步任务模式集成测试（marker=fc，默认 skip）
    └── data/                # test_fc / test_fc_task / test_cli 用到的 PDB / JSON / 等 fixture
```

服务专属（按需添加，不强制）：

| 文件 | 用途 | 例子 |
|---|---|---|
| `tools.py` / `configs.py` | argv builder / 配置 YAML 构造函数 | [rfantibody-server/tools.py](../../services/rfantibody-server/tools.py), [genie3-server/configs.py](../../services/genie3-server/configs.py) |
| `datasets.py` | dataset zip 解压 + 路径重写 | [genie3-server/datasets.py](../../services/genie3-server/datasets.py) |
| `uris.py` | URI scheme 解析（job:// / oss:// / file:// / http(s):// 等）——**现由框架统一提供** | [_framework uris.py](../../services/_framework/src/bioq_service/uris.py) |
| `patches/` | 上游源码补丁 + 应用规则（vendor.sh 后 Dockerfile 中 `patch -p1` 应用） | [genie3-server/patches/](../../services/genie3-server/patches/) |
| `scripts/vendor.sh` | **强制**：clone 上游源到 `upstream/` at pinned SHA + 重试 + 校验。Build 期 Dockerfile 从 `upstream/` COPY，不在 image 内 `git clone` | [proteinmpnn-server/scripts/vendor.sh](../../services/proteinmpnn-server/scripts/vendor.sh)（单 upstream），[promera-server/scripts/vendor.sh](../../services/promera-server/scripts/vendor.sh)（多 upstream） |
| `scripts/fetch_weights.sh` | 预下载模型权重到 `weights/`，**支持 `WEIGHTS_DST=` 直接下到 NAS**。Docker image 不再 COPY 权重 | [boltzgen-server/scripts/fetch_weights.sh](../../services/boltzgen-server/scripts/fetch_weights.sh)，[immunebuilder-server/scripts/fetch_weights.sh](../../services/immunebuilder-server/scripts/fetch_weights.sh) |
| `upstream/` | gitignored 上游源 vendor 目录（vendor.sh 的产物）；Docker build 从这里 COPY | proteinmpnn-server/upstream/ |
| `weights/` | gitignored 本地权重 stage 目录；正式部署 rsync 到 NAS | boltz-server/weights/ |

## 验证检查清单

新 service 提交前跑一遍：

```bash
# 1. lint
uvx ruff check services/<svc>/

# 2. unit tests — HTTP (test_app) + CLI (test_cli)
uv run python -m pytest services/<svc>/tests/test_app.py -v
uv run python -m pytest services/<svc>/tests/test_cli.py -v

# 3. import smoke (catch typos in adapter / settings)
python -c "from server.app import app; print(app.title)"

# 3.5. vendor 上游源（必备，Docker build 前运行）
./services/<svc>/scripts/vendor.sh
# 验证产物
ls services/<svc>/upstream/ | head

# 3.6. (如有) 预下载权重 — 仅本地 SIF 测试需要；FC 部署直接上传到 NAS
# ./services/<svc>/scripts/fetch_weights.sh

# 4. local docker build (验证 Dockerfile 没漏 COPY)
make build-<svc>

# 5. manifest payload sanity (确认 endpoints_examples 都填了)
docker run --rm -p 9000:9000 <svc>-server &
curl http://localhost:9000/api/manifest | jq .endpoints
kill %1

# 5.5. CLI smoke — 验证 __main__.py 能解析参数（不需要真跑算法）
docker run --rm <svc>-server .venv/bin/python -m server --help
docker run --rm <svc>-server .venv/bin/python -m server generate --help

# 5.7. task endpoint 路由 sanity（验证 task endpoints 已注册）
docker run --rm -p 9000:9000 <svc>-server &
sleep 5
curl http://localhost:9000/openapi.json | jq '.paths | keys | .[]' | grep "/api/tasks/"
# 应该看到 /api/tasks/<name>... 与 /api/<name>... 一一对应
kill %1

# 6. 部署到 FC 后跑 FC 测试（先 smoke，再 inference）
pytest -m fc -k "not minimal_job" services/<svc>/tests/test_fc.py
pytest -m fc services/<svc>/tests/test_fc.py

# 7. 部署 + 控制台开启异步任务模式后跑 task 测试（推荐：现代 GPU 服务的首选入口）
pytest -m fc services/<svc>/tests/test_fc_task.py
```

## 提交清单（在 PR 描述里勾掉）

- [ ] **设计文档已就位**：`YYYY-MM-DD-<svc>-server-design.md` 含 §0 列出的必备章节，且已在 decisions 索引加一行链接（按团队 decisions/ 流程归档）
- [ ] 必备文件（含 VERSION + `__main__.py` + scripts/vendor.sh + tests/{conftest,test_app,test_cli,test_fc,test_fc_task,data} + README；pyproject.toml 按需）
- [ ] `scripts/vendor.sh` 存在且可跑：固定上游 SHA、5 次重试、SHA 校验、rsync 到 `upstream/`
- [ ] `services/<svc>/upstream/` 加进项目根 `.gitignore`
- [ ] **Dockerfile 中无 `git clone`、无 `COPY opensource/`**：上游一律从 `services/<svc>/upstream/` COPY；apt 不装 git（除非有其它真实用途）
- [ ] **Dockerfile 中无 `COPY services/<svc>/weights/`**：权重默认外置到 NAS；如确需烘焙小权重（< 100 MB），注释里写明理由
- [ ] `settings.py` 的 `weights_dir` default 指向 `/data/models/<svc>/...`
- [ ] `app.py` 实现自定义 `/healthz/detail`（带 `_strip_route` + `weights_loaded` / `weights_missing` 字段），权重缺失时 HTTP 200 + `weights_loaded=false`，不在 import 期 raise
- [ ] `services/<svc>/README.md` 含 `## Weights` 章节：NAS 布局树 / pre-stage 命令 / FC verify / SIF `--bind` 示例
- [ ] `scripts/fetch_weights.sh` 支持 `WEIGHTS_DST=` env 覆盖直接下到 NAS（如服务需要本地权重）
- [ ] 权重已上传到 NAS `/data/models/<svc>/`（FC 部署前置）
- [ ] `__main__.py` 注册所有 endpoint 的 `CLIEndpoint` 描述符
- [ ] `JobAdapter.manifest_extras()` 至少有 `tool_outputs` + `input_uri_schemes`
- [ ] `JobAdapter.endpoint_examples()` 每个 endpoint 至少 1 个 curl 示例
- [ ] endpoint 用 `Depends(model_form_depends(Model))` 接收表单参数（见 app.py 注意事项）
- [ ] 每个 submit/poll endpoint 都有对应的 task endpoint（`/api/tasks/<name>`），用 `register_task_endpoint`（无上传）或 `execute_task`（有上传）
- [ ] task endpoint 注册放在 `if settings.task_endpoints_enabled:` 守卫内（自定义 endpoint 时）
- [ ] FC 部署后控制台开启「异步任务模式」+ 清空 keepalive URL + 确认 NAS 挂载 `/data/models/<svc>` 可读
- [ ] **经 gateway 调用的服务**：FC 控制台把数据面 OSS bucket `bioagent-inputs` 挂到 `/mnt/oss`（RW）；Dockerfile runtime 有 `ENV <SVC>_OSS_OUTPUT_MOUNT=/mnt/oss`；framework 用 `COPY services/_framework`（非 bind-mount，否则 output-sink 修复进不去镜像）；有文件输入的服务其 `uris.py` 必须支持裸 `/` 绝对路径（gateway 把 `oss://` 改写成 `/mnt/oss/...`，下游用 `shutil.copy2` 直读）。详见 迁移到 OSS mount
- [ ] FC 部署后 `curl /healthz/detail` 验证 `weights_loaded: true`
- [ ] Dockerfile 用 uv venv 骨架 或 conda/micromamba 多阶段构建
- [ ] **conda 服务**：Dockerfile runtime 阶段有 `ENV LANG=C.UTF-8` + `ENV LC_ALL=C.UTF-8`（关闭 upstream `open()` ASCII locale 陷阱，见 [conda-pitfalls.md](./conda-pitfalls.md)）
- [ ] **conda 服务**：bundled upstream config yaml 已 strip 掉所有 wrapper 通过 CLI 控制的键（`data_file`, `data_path`, `num_samples`, `seed`, `mode`, `checkpoint_path`, etc.）；wrapper 里加了 `assert upstream_args.data_file == csv_path` 类防御断言
- [ ] **conda 服务**：Dockerfile 有 filesystem sanity check（`[ -f "$f" ] || exit 1` for each critical upstream .py）+ 完整 upstream module-chain import smoke（含 wandb/matplotlib stub 注入，如需）
- [ ] **conda 服务**：wrapper 里所有 top-level 死代码 import（wandb / matplotlib / etc.）用 `sys.modules.setdefault(name, ModuleType(name))` stub，**不装真包**
- [ ] 改过 `scripts/vendor.sh` 排除规则的：**先 `rm -rf services/<svc>/upstream/` 再重跑 vendor.sh**，避免 Docker COPY 复用陈旧 vendor 产物
- [ ] 如有 `pyproject.toml`，需依赖 `bioq-service-framework`（部分 service 不需要 pyproject.toml——server 代码通过 COPY 注入而非 pip install）
- [ ] settings.py 没有 `os.getenv` 调用（全走 pydantic-settings）
- [ ] `uvx ruff check` 通过
- [ ] `pytest tests/test_app.py tests/test_cli.py` 通过（offline）
- [ ] 部署后 `pytest -m fc services/<svc>/tests/test_fc.py` 通过（每个 endpoint 至少 1 个 inference job 跑过）
- [ ] 控制台开启异步任务模式后 `pytest -m fc services/<svc>/tests/test_fc_task.py` 通过（覆盖 `/api/tasks/<name>` 的 submit=202 / 完成 / 生命周期 / 平台层 dedup）
- [ ] [services/services.yaml](../../services/services.yaml) 加一个条目 `<svc>-server:` + `url: https://...`（可选 `tier` / `region` / `function` / `gpu`；**有文件输入的服务加 `oss_mount: true`** 让 gateway 把 `oss://` 输入改写成 `/mnt/oss/...` 供下游凭证-free 直读）
- [ ] **经 gateway 调用的服务**：在 [services/gateway-server/tests/test_fc.py](../../services/gateway-server/tests/test_fc.py) 加一个 `TestEndToEnd<Svc>` e2e 类（照 `TestEndToEndProteinMPNN` 走 presign-file 输入 / `TestEndToEndMMseqs2` 走 inline 参数），断言 download 是 302→OSS + `results.zip` 内含预期产物。**嵌套 endpoint（如 `generate/motif`）需 gateway ≥ v0.0.2**（`{endpoint:path}` 路由）
- [ ] 在消费端调用指南（base-URL 表格）追加一行 base URL + source 链接（该表随消费端文档，按团队流程维护）

## 相关

- [services/_framework/](../../services/_framework/) —— 框架包源码 + 单元测试可作参考
- Service 框架抽象设计 —— 设计动机 + 内部协议
- CLI 批处理模式设计 —— 同一镜像 HTTP + CLI 双模式架构、`CLIEndpoint` 描述符、pydantic→argparse 转换规则
- FC 异步任务模式设计 —— task endpoint + execute_task / register_task_endpoint 框架
- gateway OSS output-sink 设计 + 迁移到 OSS mount（cookbook） —— 经 gateway 调用时的输入直读（`oss://→/mnt/oss`）+ 结果回传 OSS；新服务照 cookbook 的 checklist 接入
- Service 权重 NAS 外置化设计 —— 权重 NAS 挂载 + vendor.sh + healthz 探针的完整背景
- FC GPU 实例 keepalive —— legacy keepalive 与 task endpoint 的对比
- [services/_framework/scripts/probe_fc_concurrency.py](../../services/_framework/scripts/probe_fc_concurrency.py) —— FC 异步任务模式并发探测工具
- 调用 bioagent service —— agent / client 端怎么消费这些服务
- 参考实现：
  - [proteinmpnn-server](../../services/proteinmpnn-server/) —— **vendor.sh 标准示范**：单上游 + cherry-pick + 测试通过的最简骨架
  - [boltzgen-server](../../services/boltzgen-server/) —— vendor.sh + 权重外置 + healthz 探针 + 大镜像（~15 GB → ~2.5 GB）的对照
  - [promera-server](../../services/promera-server/) —— vendor.sh **多上游**（tinyprot + promera + LigandMPNN）+ symlink 模式权重
  - [genie3-server](../../services/genie3-server/) —— vendor.sh + patches 应用 + symlink 模式权重
  - [rfantibody-server](../../services/rfantibody-server/) / [boltz-server](../../services/boltz-server/) —— uv venv 骨架，已外置权重
  - [deeprank-ab-server](../../services/deeprank-ab-server/) —— conda/micromamba 多阶段骨架，已外置权重
