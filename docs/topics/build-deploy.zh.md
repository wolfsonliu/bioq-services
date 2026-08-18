# 构建 / 部署镜像

[English](build-deploy.md) | 中文

> **适用**：构建 / 打 tag / 推送 / bump / SIF 镜像，或切版本时。
> **来源**：`Makefile`（自动发现 + 每服务版本化）与 `docs/adding-a-new-service/` 验证流程。
> **刷新/删除条件**：Makefile 目标或版本策略变化时。

`Makefile` 跨层自动发现可构建镜像（`services/*/Dockerfile` + `gateway/Dockerfile` +
`edge/*/Dockerfile`；`framework/` 无 Dockerfile 故跳过）。镜像名 = 目录末段名（worker 保留
`-server`）。构建上下文是**仓库根**（`docker build -f <svc-dir>/Dockerfile .`）。

```bash
make help                     # 每个目标，带说明
make list                     # 列出被发现的服务（镜像名）
make version                  # 打印每个服务的当前 tag
make build-<service>          # 按其 VERSION 构建
make build-<svc> TAG=v0.0.5   # 覆盖单次构建的 tag
make push-<service>           # 构建 + tag + 推送 harbor（REGISTRY 可覆盖）
make bump-<service>           # patch 版本 +1（v0.0.5 → v0.0.6）
make sif-<service>            # Docker → Apptainer SIF（HPC/Slurm）
make login-harbor             # 首次推送前 docker login harbor.ruosheng.bio
make clean-<service>          # 删除单个本地镜像
make clean                    # 删除全部本地服务镜像
```

## 版本化

每个 service 独立发版。tag 优先级：

1. `TAG=vX.Y.Z` CLI 覆盖（优先于一切）
2. `<svc-dir>/VERSION` 文件（常规情况）
3. `git describe --tags --always --dirty`（仅无版本本地构建）

从不全局协调 tag。

## SIF

`make sif-<service>` 需要 PATH 上有 `apptainer`；输出到 `SIF_DIR`（默认 `sif/`）。用
`make clean-sif-<service>` / `make clean-sif` 清理。

## 新增/改动 service 之后

照 [`../adding-a-new-service/index.zh.md`](../adding-a-new-service/index.zh.md) 的验证清单跑（vendor →
本地 docker build → `/api/manifest` sanity → `python -m server --help` → task 路由 sanity），FC
部署后再跑 `test_fc` / `test_fc_task`（见 [testing.md](./testing.md)）。

## 相关

- 本地 kind 部署：[local-dev.md](./local-dev.md)
- 完整新服务流程：[新增服务 cookbook](../adding-a-new-service/index.zh.md)
