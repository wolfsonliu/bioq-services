# 仓库布局

[English](repository-layout.md) | 中文

> **适用**：需要目录地图、各层职责，或 `services.yaml` 字段含义时。
> **来源**：由仓库实际顶层目录结构与 `services.yaml` 归纳（二者可直接读取；本表若过时请按实际树重写）。
> **刷新/删除条件**：目录结构或 `services.yaml` schema 变化、导致本表误导时。

## 顶层分层

```
framework/   — 共享服务框架（库，无 Dockerfile，不发镜像）。
│              分发名 bioq-service-framework；import 名 bioq_service。
gateway/     — 控制面：认证 / 上传协商（OSS presign）/ 异步调度 / 状态与下载代理
│              （ECS，docker-compose / kind 部署）。
edge/        — 非 worker 边缘组件：jwt/（JWT 签发）、protein-design-mcp/（MCP 适配器）。
services/    — 算力 worker（每张一张 FC/OpenFaaS 函数镜像，保留 -server 后缀）。
deploy/      — 部署目标：ecs/（生产 ECS+FC）、compose/（本地全栈）、
│              openfaas/（本地 kind+OpenFaaS）、config/（生成的共享非密配置）。
docs/        — adding-a-new-service/（cookbook）+ specs/（设计文档）
│              + topics/（本套双语话题文档）。
services.yaml   — fleet registry（已部署服务的 url / tier / function / oss_mount）。
Makefile        — 镜像构建/推送/bump/SIF + 本地 kind 联调（make local-*）。
scripts/        — 杂项脚本（如 bench_concurrency.py）。
```

## 文档组织

| 目录 | 用途 |
|---|---|
| `docs/topics/` | 平铺话题集 —— 一个话题一对双语文件（`<topic>.md` + `<topic>.zh.md`） |
| `docs/adding-a-new-service/` | 多页 cookbook —— `index.md` + 子页 |
| `docs/specs/` | 带日期的 `YYYY-MM-DD-*` 设计文档 |
| `docs/plans/` | 带日期的计划 / 决策说明 |

**平铺 vs 子目录：** 话题默认在 `docs/topics/` 平铺；只有当其真正拆成多个子页时，才升级为
`docs/<area>/` 子目录（带 `index.md`，每页 `.md` + `.zh.md`）。当前唯一例是
`docs/adding-a-new-service/`。

## Worker（38 个）

alphafold / bindflow / boltz / boltzgen / chembounce / deeprank-ab / diamond / diffdock /
diffdock-pp / diffusion-hopping / dockq / drughive / ensemble / esmfold2 / flowmol / genie3 /
haddock3 / iggm / immunebuilder / lasermpnn / lightdock / megalodon / mmseqs2 / odesign /
openadmet / openbpmd / plip / pocketxmol / ppiflow / promera / proteinmpnn / qligfep / reinvent /
rfantibody / rfdiffusion / rfdiffusion2 / semlaflow / turbohopp。

## services.yaml 语义

已部署服务的权威清单。每项只有 `url` 必填：

| 字段 | 含义 |
|---|---|
| `url` | VPC HTTP 触发器 URL |
| `region` | 阿里云区域（默认 `cn-hangzhou`） |
| `tier` | `hot` 常暖 / `warm` 缩容到零（默认）/ `cold` 仅批处理 |
| `function` | gateway 做异步状态轮询用的 FC 函数名 |
| `gpu` | GPU 卡型（可选，如 `fc.gpu.tesla.1`） |
| `oss_mount: true` | 服务有文件输入 → gateway 把 `oss://` 输入改写为 `/mnt/oss/...` |

未部署服务以**注释条目**占位。

## 下一步

- 与目录无关的概念：[mental-model.md](./mental-model.md)
- 单个 service 的文件布局：[service-anatomy.md](./service-anatomy.md)
- 控制面内部：[gateway.md](./gateway.md)
