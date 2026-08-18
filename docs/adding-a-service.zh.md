# 在 bioq-services 里新增一个服务

[English](adding-a-service.md) | 中文

> 完整流程（设计文档先行 → 骨架 → Dockerfile → 测试 → 部署 → 提交清单）在权威 cookbook
> —— [`docs/adding-a-new-service/index.zh.md`](adding-a-new-service/index.zh.md)。本页只讲「代码落在本仓
> 哪里、怎么跑起来」的 repo-local 要点，不重复 cookbook 的细节。

一个服务 = `services/<svc>-server/` 目录下的**双模 Docker 镜像**：

- **HTTP 模式**（默认）：`uvicorn server.app:app` —— FastAPI + 异步 job runner，部署到阿里云 FC；
- **CLI 批处理模式**：`python -m server <endpoint> ...` —— Slurm/sbatch 单次同步执行。

**强制基于 [`framework/`](../framework/) 构建** —— 不要再造 HTTP / job 生命周期 / 持久化 / manifest /
CLI / 上传下载这些通用层。

## 命名约定（固定，不要改）

- import 名：**`bioq_service`**（`from bioq_service import ...`）
- 分发名：**`bioq-service-framework`**
- legacy HTTP header：**`X-Bioagent-*`**、`bioagent-session-id` 等**保持不变**（历史契约）

## 起步

1. **先写设计文档**（必备章节见 [cookbook §0](adding-a-new-service/index.zh.md#0-先写设计文档开工前必做)），
   再动代码。
2. 照 cookbook 起骨架：必备文件清单与 5 分钟 echo skeleton 见
   [skeleton.zh.md](adding-a-new-service/skeleton.zh.md)。
3. 最省事的起点：把一个结构相近的现有服务整目录拷贝改名，再逐文件改。参考：
   - `proteinmpnn-server` —— uv venv + 序列设计 + 权重外置的最简骨架；
   - `dockq-server` / `diamond-server` —— CPU-only uv-venv slim；
   - `deeprank-ab-server` / `pocketxmol-server` —— conda/micromamba 多阶段。

## repo-local 验证（提交前跑一遍）

```bash
# 1. lint + 离线单测（本服务独立隔离；从服务目录跑）
cd services/<svc>-server
uvx ruff check .
uv run --group dev python -m pytest tests/test_app.py tests/test_cli.py -q

# 2. vendor 上游源码（Docker build 前必跑）
./scripts/vendor.sh
ls upstream/ | head

# 3. (可选) 预下载权重到本地 stage
# ./scripts/fetch_weights.sh

# 4. 本地镜像构建（从仓库根；Makefile 自动发现 services/*/Dockerfile）
cd ../..
make build-<svc>-server

# 5. manifest / task 路由 sanity
docker run --rm -p 9000:9000 <svc>-server &
curl -s localhost:9000/api/manifest | jq .endpoints
curl -s localhost:9000/openapi.json | jq '.paths|keys[]' | grep /api/tasks/
kill %1
```

`upstream/`（vendor.sh 产物）与 `weights/`（fetch_weights.sh 产物）**不入库**——加进本仓
`.gitignore`（若尚无 per-service 条目，追加 `services/<svc>-server/upstream/` 与
`services/<svc>-server/weights/`）。

## 注册 + 网关联通

1. `services.yaml` 加一行 `<svc>-server:`（`url`、可选 `tier`/`function`/`gpu`；**有文件输入的服务加
   `oss_mount: true`**）。未部署时可先加**注释条目**占位。
2. 经 gateway 调用的服务：在 `gateway/tests/test_fc.py` 加一个 `TestEndToEnd<Svc>` 类（照
   `TestEndToEndProteinMPNN` 走 presign-file 输入 / `TestEndToEndMMseqs2` 走 inline 参数），断言
   download 是 302→OSS + `results.zip` 内含预期产物。

## 提交清单

以 [cookbook 页末的完整 checklist](adding-a-new-service/index.zh.md#提交清单在-pr-描述里勾掉) 为准，逐条勾。
本仓侧最容易漏的两条：import 名用 `bioq_service`；`upstream/` + `weights/` 已进 `.gitignore`。

## 相关

- [新增 service cookbook 总览](adding-a-new-service/index.zh.md) —— 权威流程与子页导航
- [framework/](../framework/) —— 共享框架包源码 + 单元测试可作参考