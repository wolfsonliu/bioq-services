# 新增一个服务

[English](adding-a-service.md) | 中文

> **适用**：在本仓端到端新增一个服务时。
> **来源**：指向权威 cookbook（目前为中文）——[`../adding-a-new-service/index.zh.md`](../adding-a-new-service/index.zh.md) 及其子页。
> **刷新/删除条件**：cookbook 结构变化时。

## 流程（总览）

1. **先写设计文档（开工前必做）**：`YYYY-MM-DD-<svc>-server-design.md` 归档到
   [`../specs/`](../specs/)，必备章节见
   [`../adding-a-new-service/index.zh.md`](../adding-a-new-service/index.zh.md#0-先写设计文档开工前必做)。
2. **照 cookbook 起骨架**：子页
   [`skeleton`](../adding-a-new-service/skeleton.zh.md) ·
   [`dockerfile`](../adding-a-new-service/dockerfile.zh.md) ·
   [`conda-pitfalls`](../adding-a-new-service/conda-pitfalls.zh.md) ·
   [`testing`](../adding-a-new-service/testing.zh.md) ·
   [`deploy`](../adding-a-new-service/deploy.zh.md)；
   命名 / 必备文件 / 验证 / 注册 / 提交清单都在 [`index.zh.md`](../adding-a-new-service/index.zh.md)。
   文件布局 + 起步参考：[service-anatomy.md](./service-anatomy.md)。
3. **注册 + 网关联通**：`services.yaml` 加 `<svc>-server:` 条目（有文件输入加 `oss_mount: true`）；
   经 gateway 调用的服务在 `gateway/tests/test_fc.py` 加 `TestEndToEnd<Svc>` e2e 类。
4. **过一遍硬约束**（`AGENTS.md`）与
   [`../adding-a-new-service/index.zh.md`](../adding-a-new-service/index.zh.md) 的提交清单。

## 提交前快速验证

```bash
cd services/<svc>-server && uvx ruff check . && uv run --group dev python -m pytest tests/test_app.py tests/test_cli.py -q
./scripts/vendor.sh && ls upstream/ | head
cd ../.. && make build-<svc>-server
```

## 相关

- 约束依据：[conventions.md](./conventions.md)
- 构建/验证循环：[build-deploy.md](./build-deploy.md) · [testing.md](./testing.md)