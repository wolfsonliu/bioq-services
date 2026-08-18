# 心智模型

[English](mental-model.md) | 中文

> **适用**：写 service 代码前，需要各 service 共用的核心概念（job 目录 = 状态单元、submit/poll vs task endpoint、双模）。
> **来源**：提炼自 `framework/src/bioq_service/`（job runner、task endpoint、uris）与各 service Dockerfile；存疑时以框架源码为准。
> **刷新/删除条件**：框架的 job 生命周期或输入解析契约变化时。

## 一个 service = 一张镜像 + HTTP endpoint + CLI 入口

- 每个 service 是双模 Docker 镜像：默认 HTTP（FastAPI，FC），另有
  `python -m server <endpoint>` 供 Slurm/sbatch 单次执行。
- 两种模式共用 `tools.py`（argv 构造）、`adapter.py`（输出检测）、`settings.py`（配置）。

## 重依赖只活在 Dockerfile 里

torch/cuda/conda/rdkit 互斥、无法共用一个环境——这正是每个 service 单独 Dockerfile 的原因。
`pyproject.toml` 只声明离线测试所需的轻量依赖 + 框架 path source：

```toml
[tool.uv.sources]
bioq-service-framework = { path = "../../framework", editable = true }
```
（`gateway/` 在顶层，用 `path = "../framework"`。）

## job 目录是所有状态的单元

一个 job 在 `<jobs_base_dir>/<job_id>/` 下：

- `input/` — 自包含输入
- `output/` — 产出；`/download` 打包此目录
- `logs/run.log` — 子进程 stdout+stderr
- `job.json` — 状态 sidecar，用于重启恢复

## 两种调用范式（同一镜像内并存）

1. **submit/poll** — POST 立即返回 `job_id`；后台 `ThreadPoolExecutor` 跑子进程；客户端轮询
   `GET /api/jobs/{id}`。无需长期占用实例时用。
2. **task endpoint**（`/api/tasks/<name>`）— POST **阻塞到子进程跑完**。为 FC 异步任务模式
   （`X-Fc-Invocation-Type: Async`）设计，让实例在整段计算期间保持占用。现代 GPU 服务首选。
   两者总是成对提供。

## 统一的输入解析

所有 service 共用一套 URI scheme（细节见 [framework-api.md](./framework-api.md)）：
multipart 上传 · `job://<id>/<file>`（从上一 job 的 output chaining）· `file:///abs/path` 或裸
`/abs/path` · `oss://<bucket>/<key>` · `http(s)://...`。

## 职责怎么拆

- **adapter** 拥有全 service 一致的策略（文件布局、输出检测、日志）；它不感知请求 shape 与 argv。
- 每个 **endpoint** 拥有自己的请求模型 + argv 构造（经 `tools.py`）。
- **framework** 拥有两者都接入的 HTTP/job/CLI 管道。

正因为这样拆，同一个 adapter 就能让一张镜像同时服务于 HTTP 与 CLI 两种模式。
