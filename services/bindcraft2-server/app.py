"""bindcraft2-server 的 FastAPI app。

暴露三组 endpoint（design / rank / filter），每组都有 `/api/tasks/*` 孪生。
框架负责 /healthz、/api/jobs/{id}、/files、/log、/download、/file/{path}、
DELETE /api/jobs/{id}、/api/manifest —— 这里只注册服务专属路由和自定义
/healthz/detail。

FC 部署要点：
  - 监听 0.0.0.0:9000（CAPort）
  - 启动 120 s 内必须能应答 /healthz
  - keep-alive >= 15 min；campaign 是小时级，必须走异步任务模式
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

from bioq_service import (
    JobInfo,
    attach_mcp,
    create_app,
    execute_task,
    model_form_depends,
    read_version_file,
    resolve_task_id,
)
from bioq_service.uris import resolve_input
from fastapi import Depends, File, Form, Header, HTTPException, Request, UploadFile

from .adapter import Bindcraft2Adapter
from .campaigns import resolve_campaign_dir
from .models import (
    DesignRequest,
    FilterRequest,
    RankRequest,
    validate_target_selection,
)
from .settings import Bindcraft2Settings
from .tools import (
    CAMPAIGN_MODELS,
    design_argv,
    filter_argv,
    missing_alphafold_params,
    missing_proteinmpnn_weights,
    prepare_design,
    rank_argv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = Bindcraft2Settings()
adapter = Bindcraft2Adapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="BindCraft2 Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---------------------------------------------------------------------------
# 自定义 /healthz/detail
# ---------------------------------------------------------------------------

_KNOWN_SUFFIXES = (".mmcif", ".fasta", ".pdb", ".cif", ".fas", ".fa")

# GPU 探针在子进程里跑：绝不能在 HTTP 进程内 import jax——那会初始化 CUDA
# context 并占掉 campaign 需要的显存。
_GPU_PROBE_CODE = (
    "import jax; print(jax.default_backend()); "
    "print(','.join(sorted({d.platform for d in jax.devices()})))"
)
_probe_cache: dict[str, object] = {"at": 0.0, "value": ("", "")}


def _gpu_probe(settings: Bindcraft2Settings) -> tuple[str, str]:
    """返回 (backend, devices)；结果按 gpu_probe_ttl_seconds 缓存。"""
    now = time.monotonic()
    cached = _probe_cache["value"]
    if isinstance(cached, tuple) and cached[0]:
        if now - float(_probe_cache["at"]) < settings.gpu_probe_ttl_seconds:
            return cached  # type: ignore[return-value]
    backend, devices = "probe_failed", ""
    # 探针只做 `import jax`，与上游 CLI 无关，绝不能拼 `-m <module>`：生产里
    # module="bindcraft.cli"，`python -m bindcraft.cli -c <code>` 会把 `-c <code>`
    # 当成 CLI 的普通参数，代码永不执行（CLI 报 usage 并退出 2），于是
    # gpu_backend 永远停在 "probe_failed"——唯一能发现静默回退 CPU 的信号被毁掉。
    argv = [settings.python, "-c", _GPU_PROBE_CODE]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            # 探针卡死不能拖住健康检查；30 s 足够解释器 import jax。
            timeout=30,
            cwd=str(settings.root) if settings.root.is_dir() else None,
        )
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if proc.returncode == 0 and lines:
            backend = lines[0]
            devices = lines[1] if len(lines) > 1 else ""
        else:
            backend = "probe_failed"
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("GPU probe failed: %s", exc)
    _probe_cache["at"] = now
    _probe_cache["value"] = (backend, devices)
    return backend, devices


# 摘掉框架的通用 /healthz/detail，让服务专属探针接管。
def _strip_route(router, path: str, method: str) -> None:
    router.routes = [
        r
        for r in router.routes
        if not (getattr(r, "path", None) == path and method in getattr(r, "methods", set()))
    ]
    # FastAPI 的 included-router 层按 route-version 计数器缓存有效候选；直接赋值
    # routes 不会 bump 计数器，缓存已热时这次剥离会被静默忽略（框架的通用
    # /healthz/detail 先匹配，还不报错）。有该方法就防御性地 bump 一次。
    mark = getattr(router, "_mark_routes_changed", None)
    if callable(mark):
        mark()
    for r in router.routes:
        inner = getattr(r, "original_router", None)
        if inner is not None:
            _strip_route(inner, path, method)


_strip_route(app.router, "/healthz/detail", "GET")


@app.get("/healthz/detail")
def healthz_detail(request: Request) -> dict:
    """扩展健康检查：权重是否就位 + JAX 是否真的拿到了 GPU。

    权重缺失时返回 HTTP 200 + `weights_loaded=false`，不在 import 期 raise。
    `gpu_backend != "gpu"` 是唯一能暴露"静默跑 CPU"的信号——镜像内没有
    nvidia-smi，JAX 取不到卡时只警告一次然后慢两个数量级。

    并发字段的范围要读准：`active_jobs` 只统计 submit/poll（框架 JobRunner 的
    计数器）。`/api/tasks/*` 走框架 `execute_task`，在请求线程里同步阻塞、框架里
    没有任何活动计数，task 路径的在飞任务**无法**从这里观测；它的并发只能靠 FC 的
    `instanceConcurrency` / `sessionConcurrencyPerInstance`（deploy/fc.yaml 均为 1）
    兜底。`active_jobs_scope` 与 `concurrency_note` 就是为了不让调用方把它误读成
    "整实例在飞工作总量"。
    """
    params_dir = settings.alphafold_params_dir
    missing = missing_alphafold_params(params_dir)
    mpnn_missing = missing_proteinmpnn_weights(settings.shipped_weights_dir)
    backend, devices = _gpu_probe(settings)

    body = {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "weights_dir": str(params_dir),
        "weights_loaded": not missing and not mpnn_missing,
        "weights_missing": missing,
        "proteinmpnn_weights_loaded": not mpnn_missing,
        "proteinmpnn_weights_missing": mpnn_missing,
        "expected_alphafold_models": [f"params_{m}.npz" for m in CAMPAIGN_MODELS],
        "gpu_backend": backend,
        "gpu_devices": devices,
        "active_jobs": request.app.state.runner.active_job_count,
        # 明确标注 active_jobs 的口径：只覆盖 submit/poll，不含 /api/tasks/*。
        "active_jobs_scope": "submit_poll_only",
        "max_concurrent_jobs": settings.max_concurrent_jobs,
        "concurrency_note": (
            "active_jobs counts only submit/poll jobs "
            "(framework JobRunner.active_job_count); /api/tasks/* jobs run "
            "synchronously inside the request thread and are not counted by the "
            f"framework. BINDCRAFT2_MAX_CONCURRENT_JOBS={settings.max_concurrent_jobs} "
            "caps submit/poll jobs; task-path concurrency is capped by FC "
            "instanceConcurrency=1 and sessionConcurrencyPerInstance=1."
        ),
    }
    if backend != "gpu":
        body["warning"] = (
            "JAX did not report a GPU backend; a campaign would run on CPU "
            "(orders of magnitude slower). Check the FC GPU allocation."
        )
    return body


# ---------------------------------------------------------------------------
# 服务专属 endpoint
# ---------------------------------------------------------------------------


def _input_suffix(filename: Optional[str]) -> str:
    """按文件名/URI 后缀决定落盘扩展名；上游按后缀识别 PDB / mmCIF / FASTA。"""
    if filename:
        lower = filename.lower()
        for suffix in _KNOWN_SUFFIXES:
            if lower.endswith(suffix):
                return suffix
    return ".pdb"


def _resolve_target_input(
    target: Optional[UploadFile],
    target_uri: Optional[str],
    dest: Path,
    settings: Bindcraft2Settings,
) -> Path:
    """`resolve_input` 的包装：客户端输入错误一律 422，不要漏成 500。

    submit/poll 与 FC 异步任务两条入口共用（孪生端点若各写一份，很容易只修一处）。
    """
    try:
        return resolve_input(target, target_uri, dest, settings, field_name="target")
    except HTTPException:
        # 框架已给出合理的 4xx/502，原样透传。
        raise
    except Exception as exc:
        # 其余都是客户端输入错误（目录、坏 URI、缺凭证……），映射成 422，
        # 否则 FastAPI 会把它当服务端故障返回 500。
        raise HTTPException(
            status_code=422, detail=f"Could not resolve target: {exc}"
        ) from exc


@app.post("/api/design", response_model=JobInfo)
def run_design(
    target: Optional[UploadFile] = File(
        None, description="目标结构（PDB/mmCIF/FASTA）。与 `target_name`、`target_uri` 三选一。"
    ),
    target_uri: Optional[str] = Form(
        None,
        description=(
            "目标结构的 URI 代替上传。schemes: job://<id>/<file>, file:///path, "
            "oss://<bucket>/<key>, http(s)://..."
        ),
    ),
    params: DesignRequest = Depends(model_form_depends(DesignRequest)),
) -> JobInfo:
    """提交一场 BindCraft2 campaign（submit/poll 模式）。"""

    def _build(job_id: str, job_dir: Path) -> list[str]:
        validate_target_selection(params.target_name, target is not None, target_uri)
        target_path = None
        if not params.target_name:
            dest = job_dir / "input" / f"target{_input_suffix(target_uri or getattr(target, 'filename', None))}"
            target_path = _resolve_target_input(target, target_uri, dest, settings)
        campaign_file = prepare_design(
            params, job_dir=job_dir, target_path=target_path, settings=settings
        )
        return design_argv(campaign_file, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="design", input_params=params.model_dump(mode="json")
    )


@app.post("/api/rank", response_model=JobInfo)
def run_rank(
    params: RankRequest = Depends(model_form_depends(RankRequest)),
) -> JobInfo:
    """对已有 campaign 目录按指定指标重排（submit/poll 模式）。"""

    def _build(job_id: str, job_dir: Path) -> list[str]:
        campaign_dir = resolve_campaign_dir(params.campaign_uri, settings)
        return rank_argv(params, campaign_dir, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="rank", input_params=params.model_dump(mode="json")
    )


@app.post("/api/filter", response_model=JobInfo)
def run_filter(
    params: FilterRequest = Depends(model_form_depends(FilterRequest)),
) -> JobInfo:
    """对已有 campaign 目录重放 / 覆盖 acceptance 阈值（submit/poll 模式）。"""

    def _build(job_id: str, job_dir: Path) -> list[str]:
        campaign_dir = resolve_campaign_dir(params.campaign_uri, settings)
        return filter_argv(params, campaign_dir, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="filter", input_params=params.model_dump(mode="json")
    )


# ---------------------------------------------------------------------------
# FC 异步任务模式孪生
# ---------------------------------------------------------------------------

if settings.task_endpoints_enabled:

    @app.post("/api/tasks/design", response_model=JobInfo)
    def run_design_task(
        request: Request,
        target: Optional[UploadFile] = File(None),
        target_uri: Optional[str] = Form(None),
        params: DesignRequest = Depends(model_form_depends(DesignRequest)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """campaign 作为单个原子任务执行（小时内阻塞，需 FC 异步任务模式）。"""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            validate_target_selection(params.target_name, target is not None, target_uri)
            if params.target_name:
                return
            dest = input_dir / f"target{_input_suffix(target_uri or getattr(target, 'filename', None))}"
            paths["target"] = _resolve_target_input(target, target_uri, dest, settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            campaign_file = prepare_design(
                req, job_dir=job_dir, target_path=paths.get("target"), settings=settings
            )
            return design_argv(campaign_file, job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="design", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/rank", response_model=JobInfo)
    def run_rank_task(
        request: Request,
        params: RankRequest = Depends(model_form_depends(RankRequest)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, _input_dir: Path) -> None:
            paths["campaign"] = resolve_campaign_dir(params.campaign_uri, settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return rank_argv(req, paths["campaign"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="rank", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/filter", response_model=JobInfo)
    def run_filter_task(
        request: Request,
        params: FilterRequest = Depends(model_form_depends(FilterRequest)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, _input_dir: Path) -> None:
            paths["campaign"] = resolve_campaign_dir(params.campaign_uri, settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return filter_argv(req, paths["campaign"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="filter", params=params,
            build_argv=_build, save_inputs=_save,
        )


# MCP server 必须在所有 POST 路由注册之后挂载（自动发现会遍历全部路由）。
attach_mcp(app)
