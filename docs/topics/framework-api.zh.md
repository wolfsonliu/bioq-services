# 框架 API

[English](framework-api.md) | 中文

> **适用**：写 endpoint / adapter / settings / CLI 代码、或需要某个钩子的精确签名时。
> **来源**：`framework/src/bioq_service/`——每个模块的 docstring 与 `framework/tests/` 是权威细节；本页只是快速索引。
> **刷新/删除条件**：框架 API 变化时；本页应继续瘦身而非膨胀（优先指回源码）。

import 名固定为 **`bioq_service`**（`from bioq_service import ...`）；分发名 `bioq-service-framework`。

## JobAdapter（`adapter.py`）

一个 service 一个 `JobAdapter` 子类，承载全 service 一致的策略（文件布局、输出检测、日志、子进程
env/cwd、重启恢复）；它不感知具体请求 shape / argv（那是 per-endpoint 的事）。可覆写钩子：

| 钩子 | 默认 | 用途 |
|---|---|---|
| `name: str` | 必填 | service 名，`JobInfo.service` 用它 |
| `job_dir(job_id)` | `<jobs_base_dir>/<job_id>` | job 工作目录 |
| `output_dir(job_dir)` | `<job_dir>/output` | `/download` 打包、`/files` 列出的目录 |
| `log_path(job_dir)` | `<job_dir>/logs/run.log` | 子进程 stdout+stderr tee 位置 |
| `detect_outputs(job_dir)` | `output/` 非空 | rc==0 但返回 False → FAILED（`failure_kind=NO_OUTPUTS`）。多端点服务应覆写识别各工具产物 |
| `subprocess_env()` | `{}` | 注入子进程的额外环境变量 |
| `subprocess_cwd()` | None | 子进程工作目录 |
| `infer_job_from_dir(job_dir)` | 有产出→COMPLETED | 无 `job.json` 的遗留目录恢复启发式 |
| `manifest_extras()` | `{}` | **强烈建议覆写**：至少 `tool_outputs` + `input_uri_schemes` |
| `endpoint_examples()` | `{}` | **强烈建议覆写**：每个 endpoint ≥1 个可复制 curl（python snippet 更好） |

## ServiceSettings（`settings.py`）

所有运行时配置走 `pydantic_settings.BaseSettings` 子类——框架与 adapter 里**禁止 `os.getenv`**。
每个 service 设自己的 `env_prefix`：

```python
class DockQSettings(ServiceSettings):
    model_config = SettingsConfigDict(env_prefix="DOCKQ_", env_file=".env", extra="ignore")
```

基类字段（env `<PREFIX>_*`；默认见 `framework/src/bioq_service/settings.py`）：`jobs_base_dir`
（`/data/jobs`）、`uploads_base_dir`（`/data/uploads`）、`oss_output_mount`（`/mnt/oss`）、
`oss_region`（`cn-hangzhou`）、`disk_limit_mb`（`8000`，超过驱逐已结束 job）、`port`（`9000`）、
`max_concurrent_jobs`（`2`，超出 503）、`keep_alive_sec`、`keepalive_interval_s` / `keepalive_url`
（FC 自保活）、`session_header_name`（FC 会话亲和）、`task_endpoints_enabled`（True）、
`task_job_id_header`（`X-Bioagent-Job-Id`）。

## 组集 app（`app.py`）

```python
settings = DockQSettings()
adapter = DockQAdapter(settings=settings)
app = create_app(adapter, settings, title="...", version=read_version_file(__file__))
```

`create_app` 自动挂通用路由（healthz / manifest / openapi / jobs 系列），并在 `app.state` 暴露
`adapter` / `settings` / `job_store` / `runner`。service 侧只加自己的 POST 端点：

- 收表单参数用 `params: Model = Depends(model_form_depends(Model))`；
- 触发 submit/poll 用 `app.state.runner.submit(build_argv=_build, label="...", input_params=...)`；
- 可选：所有 POST 路由之后调 `attach_mcp(app)` 把 HTTP 面镜像到 `/mcp`（需 `bioq-service-framework[mcp]`）。

`read_version_file(__file__, default=...)` 读同目录 `VERSION`、去前导 `v`，保证 HTTP version 与
镜像 tag 不漂移。

## Task endpoint（`task_endpoint.py`）

- `resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)` — header 优先，否则 UUID。
- `execute_task(request, *, job_id, label, params, build_argv, save_inputs=None, oss_prefix=None)`
  — 同步跑完整管线（幂等去重、磁盘驱逐、PENDING→RUNNING→finalize、失败=200+FAILED、成败都尝试
  OSS output-sink）。有 `UploadFile` / 自定义 Form 的端点用它并自写 handler。
- `register_task_endpoint(app, *, path, label, request_model, build_argv, save_inputs=None)` — 无上传场景的便捷封装。

**不要**给该模块补 `from __future__ import annotations`（PEP 563 字符串注解破坏 FastAPI 的
`get_type_hints`）；改文件时保持原样。

## CLI 批处理（`cli.py`）

每个 service 的 `__main__.py`：

```python
endpoints = {"score": CLIEndpoint(name="score", help="...", request_model=ScoreRequest,
                                  build_argv=_score_build, inputs={"model": ("help", True)})}
create_cli(adapter, settings, endpoints, version="0.0.1")
```

`CLIEndpoint.inputs` 每个键变成一个 `--<name>` 本地文件旗标；`build_argv(req, inputs, job_dir,
settings)` 是薄薄一层到 `tools.py`。`create_cli` 从 pydantic model 自动生成 argparse flags、解析
输入、跑子进程、按 rc + `detect_outputs` 决定退出码。

## 输入解析（`uris.py`）

| scheme | 语义 |
|---|---|
| multipart 上传 | 客户端直接传文件 |
| `job://<id>/<file>` | 从同一 NAS 上前一个 job 的 `output/` 拉取（chaining） |
| `file:///abs/path` 或裸 `/abs/path` | 复制 NAS 本地路径 |
| `oss://<bucket>/<key>` | 走 OSS SDK 下载（需 `alibabacloud-oss-v2`） |
| `http(s)://...` | 流式拉远程 URL（含 OSS 签名 URL） |

API：`resolve_input(upload, input_uri, dest, settings, field_name=None)`（两者都给则 URI 优先；
都缺 → 422；`field_name` 用于 422 详情定位缺字段）、`maybe_resolve_input(...)`（都缺返回 None，给
内联 SMILES/序列场景）、`resolve_uri(uri, dest, settings)`、`save_upload(upload, dest)`。

## 通用 endpoint（自动注册）

`GET /healthz`、`GET /healthz/detail`、`GET /api/manifest`、`GET /openapi.json`、
`GET /api/jobs/{id}`、`GET /api/jobs/{id}/files`、`GET /api/jobs/{id}/log`、
`GET /api/jobs/{id}/download`、`GET /api/jobs/{id}/file/{path}`、`DELETE /api/jobs/{id}`。
