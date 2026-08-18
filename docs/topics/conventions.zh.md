# 约定与约束的依据

[English](conventions.md) | 中文

> **适用**：想知道某条硬约束背后的“为什么”，或提交 service 变更前要过完整清单时。
> **来源**：沉淀自跨服务踩坑（URI 字段 422、bind-mount 的 output-sink 修复）与 FC/NAS 部署模型；条目编号对应 `AGENTS.md` 的 15 条硬约束。
> **刷新/删除条件**：某条规则依赖的框架/契约变化时——删掉它，别让它残留。

简版清单见 [`../../AGENTS.md`](../../AGENTS.md#hard-constraints)。完整变更清单见
[`../adding-a-new-service/index.zh.md`](../adding-a-new-service/index.zh.md)。

## 每条规则的 为什么 / 何时适用 / 何时可删

1. **只用 pydantic 类型，禁止 `dict[str, Any]` 请求体。**
   *为什么*：框架从 pydantic schema 派生 manifest、CLI argparse flags 与 OpenAPI；`dict[str, Any]`
   绕过校验与自动生成的 flags。*何时*：定义任何请求/响应模型。*何时可删*：框架改用其它 schema 系统。

2. **配置走 `pydantic-settings`，禁 `os.getenv`。**
   *为什么*：单一可校验的配置源；散落的 `os.getenv` 隐藏配置、破坏测试。*何时*：增读运行时配置。
   *何时可删*：框架弃用 pydantic-settings。

3. **固定命名：`bioq_service` / `bioq-service-framework` / `X-Bioagent-*` header。**
   *为什么*：import/分发名在服务迁入本仓时已固定；header 是客户端与 task endpoint 读取的跨服务历史契约。
   *何时*：写 import、`pyproject.toml`、或碰 HTTP header。*何时可删*：全仓协调的重命名落地。

4. **基于 `framework/`，不要重造通用层。**
   *为什么*：HTTP/job 生命周期/持久化/manifest/CLI/上传下载只解一次；分叉它们破坏一致性。*何时*：起新服务。
   *何时可删*：被新共享框架取代。

5. **task endpoint 成对，且塞在 `if settings.task_endpoints_enabled:` 守卫内。**
   *为什么*：FC 异步任务模式需要阻塞端点保持实例占用；守卫让同一镜像也能服务非 FC 部署。*何时*：加任何
   submit/poll 端点。*何时可删*：FC 异步任务模式不再是目标。

6. **URI 字段名严格等于 `<upload_field>_uri`。**
   *为什么*：客户端 `--file <field>=<path>` 把上传字段映射为 `<field>_uri`；命名错配会让 FastAPI 丢弃
   字段 → `upload=None, uri=None` → 422。*何时*：定义文件/URI 二选一输入。*何时可删*：映射约定变化（见
   `../specs/2026-08-18-cross-service-uri-field-naming-design.md`）。

7. **权重外置 NAS；`/healthz/detail` 报 `weights_loaded`，import 期不 raise。**
   *为什么*：权重大且共享——烘焙进镜像膨胀体积；import 期 raise 会杀死探针。*何时*：添加/迁移模型权重。
   *何时可删*：确需 <100 MB 小权重进镜像（注释说明理由）。

8. **上游固定 SHA vendor；不在镜像内 `git clone`、无 `COPY opensource/`。**
   *为什么*：可复现、离线构建；`COPY opensource/` 是历史路径。*何时*：写 `vendor.sh` / `Dockerfile`。
   *何时可删*：构建弃用 Docker 或 vendor 方案变化。

9. **用 `COPY framework /tmp/service-framework` + `pip install` 装框架。**
   *为什么*：bind-mount 不进镜像，生产会缺运行时修复（如 output-sink）。*何时*：在 Dockerfile 装框架。
   *何时可删*：框架改为固定版本发布到包注册中心。

10. **每个 service 独立 `VERSION`、独立发版。**
    *为什么*：服务按各自节奏演进；全局 tag 强加耦合。*何时*：切版本。*何时可删*：发版策略全局变更。

11. **`manifest_extras`（`tool_outputs` + `input_uri_schemes`）；`endpoint_examples` ≥1 curl。**
    *为什么*：agent/客户端只靠 manifest 调用服务，无需读源码。*何时*：完成服务或改端点。
    *何时可删*：manifest 消费者消失。

12. **端点用 `Depends(model_form_depends(Model))`。**
    *为什么*：multipart 表单解析需要该依赖；裸 `params: Model` 破坏文件/表单解析。*何时*：加任何带表单
    参数的 POST 端点。*何时可删*：FastAPI 表单处理变化。

13. **`framework/src/bioq_service/task_endpoint.py` 不要加 `from __future__ import annotations`。**
    *为什么*：PEP 563 字符串注解破坏 FastAPI 对运行时类的 `get_type_hints`。*何时*：编辑该文件。
    *何时可删*：FastAPI 支持 PEP 563。

14. **`upstream/` 与 `weights/` 入 .gitignore。**
    *为什么*：它们是 vendor/下载构建产物。*何时*：任何新服务。*何时可删*：永不。

15. **只用仓库根相对路径。**
    *为什么*：仓库被检出到任何位置都不会断链。*何时*：写文档/配置。*何时可删*：永不。

## 其它规范

- **文档/注释用中文；代码/标识符/commit 用英文**（与现有 README/docs 一致）。`AGENTS.md` 与英文
  `docs/topics/*.md` 是刻意例外，并有对应 `*.zh.md`。
- **commit 不加 AI co-author trailer**（如 `Co-Authored-By`）。
- **协议变更自包含**：改动端点签名 / 上传字段 / manifest 时，同一变更里更新该 service 的 `README.md`
  与 `endpoint_examples()`，并按需补 manifest 回归测试（见 rfantibody 的
  `test_quiver_uri_field_matches_upload_field`）。
- **想看框架完整行为？** 读 `framework/src/bioq_service/`（docstring 详尽）与对应 `framework/tests/`。
