# Skeleton — 服务源码文件

[English](skeleton.md) | 中文

> ← 返回 [新增 service cookbook 总览](./index.zh.md)

本页覆盖服务的源码文件骨架：`__init__.py` / `settings.py` / `models.py` / `adapter.py` /
`app.py`（含 task endpoint）/ `__main__.py` / `pyproject.toml` / `VERSION` / `README.md`。
Dockerfile 见 [dockerfile](./dockerfile.zh.md)；测试见 [testing](./testing.zh.md)。

## 5 分钟 echo skeleton

下面是最小可跑的新 service 起点。把所有 `<svc>` 替换为你的服务名（小写，连字符），所有 `<Svc>` 换成驼峰。

### 1. `services/<svc>/__init__.py`

空文件。

### 2. `services/<svc>/settings.py`

```python
"""Env-driven config for <svc>. All values via pydantic-settings (no os.getenv)."""

from pathlib import Path
from bioq_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class <Svc>Settings(ServiceSettings):
    model_config = SettingsConfigDict(env_prefix="<SVC>_", env_file=".env", extra="ignore")

    # Override the framework default so FC deployment can use NAS layout.
    jobs_base_dir: Path = Field(default=Path("/data/<svc>_jobs"))

    # Service-specific knobs (env: <SVC>_ROOT, <SVC>_WEIGHTS_DIR, ...)
    root: Path = Field(default=Path("/opt/<svc>"))

    # 权重目录 —— **统一约定**默认指向 NAS 挂载点 /data/models/<svc>/。
    # FC 自动挂载；SIF / 本地 docker 需 --bind /scratch/models/<svc>:/data/models/<svc>。
    # 不在 image 内烘焙权重。
    weights_dir: Path = Field(default=Path("/data/models/<svc>"))
    # ... add what your tool needs
```

### 3. `services/<svc>/models.py`

```python
"""Per-endpoint pydantic request models. Re-export framework's JobInfo for compat."""

from bioq_service import JobInfo, JobStatus, FailureKind  # noqa: F401
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    n_samples: int = Field(default=4, ge=1, le=10000)
    seed: int | None = Field(default=None)
```

### 4. `services/<svc>/adapter.py`

```python
"""Service-wide policy: name + output detection + manifest_extras + endpoint_examples."""

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter, JobInfo, JobStatus

from .settings import <Svc>Settings


class <Svc>Adapter(JobAdapter):
    name = "<svc>"

    settings: <Svc>Settings  # narrow for IDEs

    def __init__(self, settings: <Svc>Settings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        """Tighten the default `output_dir non-empty` check to your real artifact."""
        out = self.output_dir(job_dir) / "result.txt"
        return out.exists() and out.stat().st_size > 0

    def subprocess_cwd(self) -> Path | None:
        return self.settings.root

    def manifest_extras(self) -> dict:
        """Service-specific protocol knowledge an agent needs to call this service."""
        return {
            "tool_outputs": {"generate": "output/result.txt"},
            "input_uri_schemes": {"upload": "multipart/form-data"},
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        """One copy-pasteable curl per endpoint (mandatory)."""
        return {
            "/api/generate": [
                EndpointExample(
                    title="basic generation",
                    curl="curl -X POST $URL/api/generate -F n_samples=4",
                    notes="Smallest call. See request_fields in manifest for full params.",
                ),
            ],
        }
```

### 5. `services/<svc>/app.py`

```python
"""FastAPI app + service-specific POST routes. Framework provides /healthz / /api/jobs / /api/manifest."""

from pathlib import Path
from typing import Optional

from bioq_service import JobInfo, attach_mcp, create_app, model_form_depends, read_version_file
from fastapi import Depends, File, Form, Request, UploadFile

from .adapter import <Svc>Adapter
from .models import GenerateRequest
from .settings import <Svc>Settings

settings = <Svc>Settings()
adapter = <Svc>Adapter(settings=settings)

app = create_app(
    adapter, settings,
    title="<Svc> Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---- /healthz/detail override：报告 NAS 权重是否就位 ----
# 走 NAS 权重的服务**必须**自定义 /healthz/detail，让 agent 能在第一次推理
# crash 之前发现挂载缺失。框架默认提供的 /healthz/detail 是泛用磁盘信息，
# FastAPI 用 first-match 路由，所以要先 strip 掉框架的再注册自己的。
# FastAPI >=0.115 把 included router 包了一层 _IncludedRouter，递归剥离。

def _strip_route(router, path: str, method: str) -> None:
    router.routes = [
        r for r in router.routes
        if not (getattr(r, "path", None) == path
                and method in getattr(r, "methods", set()))
    ]
    for r in router.routes:
        inner = getattr(r, "original_router", None)
        if inner is not None:
            _strip_route(inner, path, method)


_strip_route(app.router, "/healthz/detail", "GET")


@app.get("/healthz/detail")
def healthz_detail(request: Request) -> dict:
    """权重就位探针：列出 weights_dir 下期望的关键文件，缺失时返回 weights_loaded=false。

    NAS 未挂载时不 raise（让服务能启动），把状态暴露给 /healthz/detail。
    """
    expected = {
        "main_ckpt": settings.weights_dir / "main.ckpt",
        # ... 列出本服务的关键权重文件
    }
    missing = {k: str(p) for k, p in expected.items() if not p.exists()}
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "weights_dir": str(settings.weights_dir),
        "weights_loaded": not missing,
        "weights_missing": missing,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


@app.post("/api/generate", response_model=JobInfo)
def generate(
    params: GenerateRequest = Depends(model_form_depends(GenerateRequest)),
    input_pdb: Optional[UploadFile] = File(None),
    input_pdb_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Basic generation endpoint."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        out = job_dir / "output"
        out.mkdir(exist_ok=True)
        return ["bash", "-c", f"echo 'generated {params.n_samples} samples' > {out}/result.txt"]

    return app.state.runner.submit(
        build_argv=_build, label="generate",
        input_params=params.model_dump(mode="json"),
    )


# Must come AFTER all POST routes so MCP auto-discovery sees the full surface.
attach_mcp(app)
```

> **注意**：当 endpoint 同时接收 pydantic model（form fields）和 `UploadFile` 时，
> **必须**用 `Depends(model_form_depends(Model))` 而非 `Annotated[Model, Form()]`。
> 后者在 FastAPI 中不会正确展开 model 的字段，导致上传请求解析失败。
> 如果 endpoint 不需要文件上传，`Annotated[Model, Form()]` 也能用。

#### Task endpoint（同步阻塞，FC 异步任务模式专用）

每个 submit/poll endpoint 都应该有一个对应的 `/api/tasks/<name>` task endpoint。
两者共享 `argv builder` 和 `save_inputs` 逻辑，差别只在执行模型：

| | submit/poll `/api/<name>` | task endpoint `/api/tasks/<name>` |
|---|---|---|
| HTTP 响应 | 立即返回 `JobInfo(status=pending)` | 跑完才返回 `JobInfo(status=completed/failed)` |
| 适用 | 本地 / Slurm / 客户端要立即轮询 | FC 异步任务模式（Async Task Mode） |
| 实例占用 | 子进程在后台跑，HTTP 已结束 → FC 可能回收实例 | HTTP 请求与计算同生死 → 不会回收 |
| 并发控制 | 客户端控制 | FC 平台层管理 |

详细配置见 [deploy.zh.md](./deploy.zh.md)（FC 异步任务模式控制台配置）。

**两种注册方式：**

**(a) 无文件上传**：直接用 `register_task_endpoint` 帮手（参考 `services/immunebuilder-server/app.py`）：

```python
from bioq_service import register_task_endpoint

def _generate_build(req, _job_id, job_dir):
    return generate_argv(req, job_dir, settings)

register_task_endpoint(
    app,
    path="/api/tasks/generate",
    label="generate",
    request_model=GenerateRequest,
    build_argv=_generate_build,
)
```

`register_task_endpoint` 内部检查 `settings.task_endpoints_enabled`（默认 True），关闭时是 no-op。

**(b) 有文件上传**（多数 GPU service 的情况）：自己定义 endpoint，调用 `execute_task`，并加 `if settings.task_endpoints_enabled:` 守卫（参考 `services/boltzgen-server/app.py`）：

```python
from bioq_service import execute_task, resolve_task_id
from fastapi import Header, Request

if settings.task_endpoints_enabled:

    @app.post("/api/tasks/generate", response_model=JobInfo)
    def generate_task(
        request: Request,
        params: GenerateRequest = Depends(model_form_depends(GenerateRequest)),
        input_pdb: Optional[UploadFile] = File(None),
        input_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Generate as a single atomic task. Blocks until pipeline completion."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        # closure-shared dict bridges upload persistence in _save to argv build in _build
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["pdb"] = resolve_input(input_pdb, input_uri, input_dir / "input.pdb", settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return generate_argv(req, paths["pdb"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="generate", params=params,
            build_argv=_build, save_inputs=_save,
        )
```

**关键模式与约定：**

| 项 | 约定 |
|---|---|
| Endpoint 路径 | `/api/tasks/<same-name-as-submit-poll>` |
| 双 header | 接 `X-Bioagent-Job-Id`（业务）+ `X-Fc-Async-Task-Id`（FC 平台）；优先级 `X-Bioagent-Job-Id > X-Fc-Async-Task-Id > UUID` |
| job_id 解析 | 调 `resolve_task_id(...)` 统一处理 |
| 闭包共享上传路径 | 用 dict / list 而非 magic string——`_save` 写入 `paths["..."]`，`_build` 读 |
| Settings guard | 必须用 `if settings.task_endpoints_enabled:` 包裹（自定义 endpoint 时）；用 `register_task_endpoint` 时自带 |
| `attach_mcp(app)` | 任何 task endpoint 注册之后、文件末尾——MCP 才能发现新 endpoint |
| 异常语义 | `build_argv` / `save_inputs` 抛异常 → 框架 cleanup_job + 5xx；子进程非零 rc → 200 + `status=failed` |

**Idempotency（去重）双层 defense：**

- **FC 平台层**：同 `X-Fc-Async-Task-Id` 重复 invoke 返回 HTTP 409 Conflict，**请求不进函数**（控制台异步任务模式自带）
- **框架层**：`execute_task` 进入时检查 `JobStore.get(job_id)`，存在则直接返回 existing JobInfo（用于 LocalDispatcher / curl 等不走 FC 平台的调用路径）

两层不冲突，都保留。

### 6. `services/<svc>/__main__.py`（CLI 批处理入口）

每个服务需要一个 `__main__.py`，让同一个 Docker 镜像支持 `python -m server <endpoint> ...` 一次性批处理模式。CLI 模式复用 `tools.py` 的 argv builder、`adapter.py` 的输出检测、`settings.py` 的配置——不启动 FastAPI / uvicorn。

详细钩子签名见 [framework-api.zh.md](../topics/framework-api.zh.md)（CLI 批处理）。

```python
"""CLI batch-mode entry point for <svc>-server.

Usage::

    python -m server generate \
        --input-pdb /data/input.pdb \
        --output-dir /scratch/results/
"""

from __future__ import annotations

from pathlib import Path

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import <Svc>Adapter
from .models import GenerateRequest
from .settings import <Svc>Settings
from .tools import generate_argv

settings = <Svc>Settings()
adapter = <Svc>Adapter(settings=settings)


def _generate_build(req, inputs, job_dir, settings):
    return generate_argv(
        req,
        job_dir=job_dir,
        input_pdb=inputs["input_pdb"],
        settings=settings,
    )


endpoints = {
    "generate": CLIEndpoint(
        name="generate",
        help="Run generation on an input PDB",
        request_model=GenerateRequest,
        build_argv=_generate_build,
        inputs={"input_pdb": ("Input PDB file", True)},
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")
```

**关键点**：

- **`CLIEndpoint.inputs`** 声明需要的输入文件。每个 entry 变成 `--<name>` CLI flag，框架自动校验文件存在、解析绝对路径后传入 `build_argv` 的 `inputs` 字典
- **`build_argv` 回调** 签名是 `(request, inputs, job_dir, settings) → list[str]`，和 `app.py` 里调 `tools.py` 的逻辑一样，只是套了 `CLIEndpoint` 的壳
- **无文件输入的 endpoint**（如 immunebuilder-server 接收序列参数）：`inputs={}` 留空即可，参数全从 pydantic model 自动生成 argparse flags
- **复杂类型字段**（`dict`、`list[Model]` 等无法映射为 argparse flag 的类型）：用 `--params-json '{"key": value}'` 传入。CLI flags 优先级高于 `--params-json`
- **`inputs` 和 model 字段同名冲突**：框架自动跳过已在 `inputs` 中声明的字段，不会产生 argparse 重复定义

调用方式：

```bash
# Docker — 覆盖 CMD 执行 CLI 模式
docker run --rm -v /data:/data <svc>-server \
    .venv/bin/python -m server generate \
    --input-pdb /data/input.pdb \
    --output-dir /data/results/

# Singularity / Apptainer (sbatch)
apptainer exec --nv <svc>-server.sif \
    .venv/bin/python -m server generate \
    --input-pdb /data/input.pdb \
    --output-dir /scratch/$SLURM_JOB_ID/

# --params-json 适合脚本化调用 / 复杂参数
python -m server generate \
    --input-pdb input.pdb \
    --params-json '{"n_samples": 10, "seed": 42}' \
    --output-dir ./output/
```

### 7. `services/<svc>/pyproject.toml`（可选）

`pyproject.toml` 只声明**离线测试**所需的 pip 依赖 + 框架路径依赖——运行时重依赖（torch/cuda/conda/rdkit 等）在各自 Dockerfile 里。每个服务通过相对路径 editable 依赖框架，无需发包：

```toml
[project]
name = "<svc>-server"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = [
    "bioq-service-framework",
    # 离线测试所需的轻量 pip 依赖（非算法重依赖）
]

# 相对路径引用框架，无需发布到 PyPI
[tool.uv.sources]
bioq-service-framework = { path = "../../framework", editable = true }

# 仅作为离线测试环境，不构建本服务包
[tool.uv]
package = false

[dependency-groups]
dev = ["pytest", "pytest-asyncio"]
```

跑离线单测：`cd services/<svc> && uv run --group dev python -m pytest tests/ -q`。

**例外——服务端代码需 `pip install -e .` 时**（uv venv 骨架中 Dockerfile 用 `uv pip install -e .`）：去掉 `package = false`，加回构建后端声明——

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
```

**如果 server 代码只通过 `COPY` 注入且算法依赖全部在 conda env 或单独 pip install 中解决，可以不写 pyproject.toml**（deeprank-ab-server、jwt 等即如此）。


### 9. `services/<svc>/VERSION`

```
v0.0.1
```

`Makefile` 读取这个文件作为镜像 tag。发新版时 `echo v0.0.2 > services/<svc>/VERSION`。


### 13. `services/<svc>/README.md`

至少含：
- 顶部架构图（client → FastAPI + framework → subprocess → NAS）
- 每个 endpoint 一段 curl 示例（与 `endpoint_examples()` 同步）
- 配置表（所有 env vars + 默认值）
- 本地开发命令 + Docker 构建 + FC 部署 sub-section

参考 [services/rfantibody-server/README.md](../../services/rfantibody-server/README.md) 或
[services/genie3-server/README.md](../../services/genie3-server/README.md)。

