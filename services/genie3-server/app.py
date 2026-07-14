"""FastAPI app for genie3-server.

Exposes four POST endpoints:

  * `/api/generate/unconditional` — no dataset, configurable length range
  * `/api/generate/motif`         — motif scaffolding (zip with `motifs/`)
  * `/api/generate/binder`        — binder design (zip with `targets/`)
  * `/api/generate`               — freeform YAML for advanced configs (iterative,
                                    custom `cond_strategy`, etc.)

Lifecycle (status / log / download / single-file / delete / manifest / OpenAPI)
is contributed by `bioagent_service.create_app`.

FC deployment:
  - 0.0.0.0:CAPort (default 9000)
  - /healthz must respond within 120 s of start
  - keep-alive >= 15 min so long-running generations don't get cut off
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any, Optional

import yaml
from bioagent_service import (
    JobInfo,
    attach_mcp,
    create_app,
    execute_task,
    maybe_resolve_input,
    model_form_depends,
    read_version_file,
    resolve_input,
    resolve_task_id,
)
from fastapi import Depends, File, Form, Header, HTTPException, Request, UploadFile

from .adapter import Genie3Adapter
from .configs import (
    build_binder_config,
    build_motif_config,
    build_unconditional_config,
    rewrite_custom_paths,
)
from .datasets import extract_dataset
from .models import BinderRequest, MotifRequest, UnconditionalRequest
from .settings import Genie3Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = Genie3Settings()
adapter = Genie3Adapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="Genie3 Server",
    version=read_version_file(__file__, default="0.2.0"),
)


# Remove framework's generic /healthz/detail so our genie3-specific weights
# probe takes over. FastAPI >=0.115 wraps included routers in
# `_IncludedRouter`; descend to find the framework route.
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
    """Extended health: report whether NAS-mounted weights are reachable.

    Pretrained checkpoints live on NAS at `GENIE3_PRETRAINED_DIR` (default
    `/data/models/genie3/pretrained/v1/`).  The image contains a symlink at
    /opt/genie3/pretrained → /data/models/genie3/pretrained so the genie3
    CLI's relative lookup succeeds; if NAS is unmounted the symlink target
    is missing.  `weights_loaded=false` lets the agent detect that early.
    """
    pdir = settings.pretrained_dir
    if not pdir.exists():
        weights_loaded = False
        files_found = 0
    else:
        files_found = sum(1 for p in pdir.rglob("*") if p.is_file())
        weights_loaded = files_found > 0
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "pretrained_dir": str(pdir),
        "weights_loaded": weights_loaded,
        "files_found": files_found,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_upload(upload: UploadFile, dest: Path) -> Path:
    """Stream UploadFile to disk in chunks (multi-GB datasets allowed)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return dest


def _write_yaml(config: dict[str, Any], job_dir: Path) -> Path:
    """Persist the experiment config so it's reproducible from the job dir alone."""
    path = job_dir / "input" / "experiment.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _genie3_argv(config_path: Path, num_devices: Optional[int]) -> list[str]:
    """`genie3 generate -c <yaml> [--num-devices N]`."""
    cmd = [settings.bin, "generate", "-c", str(config_path)]
    if num_devices is not None:
        cmd.extend(["--num-devices", str(num_devices)])
    return cmd


# ---------------------------------------------------------------------------
# Structured endpoints
# ---------------------------------------------------------------------------


@app.post("/api/generate/unconditional", response_model=JobInfo)
def generate_unconditional(
    params: UnconditionalRequest = Depends(model_form_depends(UnconditionalRequest)),
) -> JobInfo:
    """Unconditional protein backbone generation (no input structure)."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        config = build_unconditional_config(rootdir=job_dir / "output", req=params)
        config_path = _write_yaml(config, job_dir)
        return _genie3_argv(config_path, params.num_devices)

    return app.state.runner.submit(
        build_argv=_build, label="unconditional",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/generate/motif", response_model=JobInfo)
def generate_motif(
    dataset: UploadFile = File(..., description="Zip with problems/ + motifs/."),
    params: MotifRequest = Depends(model_form_depends(MotifRequest)),
) -> JobInfo:
    """Motif scaffolding generation. Dataset zip must contain `problems/` + `motifs/`."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        zip_path = _save_upload(dataset, job_dir / "input" / "dataset.zip")
        try:
            dataset_root = extract_dataset(zip_path, job_dir / "input" / "dataset")
        except (zipfile.BadZipFile, ValueError) as e:
            raise HTTPException(status_code=422, detail=f"Invalid dataset zip: {e}") from e
        config = build_motif_config(
            rootdir=job_dir / "output", dataset_root=dataset_root, req=params,
        )
        config_path = _write_yaml(config, job_dir)
        return _genie3_argv(config_path, params.num_devices)

    return app.state.runner.submit(
        build_argv=_build, label="motif",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/generate/binder", response_model=JobInfo)
def generate_binder(
    dataset: UploadFile = File(..., description="Zip with problems/ + targets/."),
    params: BinderRequest = Depends(model_form_depends(BinderRequest)),
) -> JobInfo:
    """Binder design generation. Dataset zip must contain `problems/` + `targets/`."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        zip_path = _save_upload(dataset, job_dir / "input" / "dataset.zip")
        try:
            dataset_root = extract_dataset(zip_path, job_dir / "input" / "dataset")
        except (zipfile.BadZipFile, ValueError) as e:
            raise HTTPException(status_code=422, detail=f"Invalid dataset zip: {e}") from e
        config = build_binder_config(
            rootdir=job_dir / "output", dataset_root=dataset_root, req=params,
        )
        config_path = _write_yaml(config, job_dir)
        return _genie3_argv(config_path, params.num_devices)

    return app.state.runner.submit(
        build_argv=_build, label="binder",
        input_params=params.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# Freeform YAML
# ---------------------------------------------------------------------------


@app.post("/api/generate", response_model=JobInfo)
def generate_custom(
    config_yaml: str = Form(..., description="Full experiment YAML as a string."),
    dataset: Optional[UploadFile] = File(
        None, description="Optional dataset zip; extracted to <job>/input/dataset/.",
    ),
    num_devices: Optional[int] = Form(None, description="Override GPU auto-detect."),
) -> JobInfo:
    """Run `genie3 generate` with a fully custom YAML config.

    `paths.rootdir` and (if a dataset zip is provided) `paths.dataset` are
    rewritten so the YAML you supply doesn't need to know its job_dir.
    """

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        try:
            user_config = yaml.safe_load(config_yaml)
        except yaml.YAMLError as e:
            raise HTTPException(status_code=422, detail=f"Invalid YAML: {e}") from e
        if not isinstance(user_config, dict):
            raise HTTPException(
                status_code=422, detail="config_yaml must be a mapping at the top level",
            )

        dataset_root: Optional[Path] = None
        if dataset is not None:
            zip_path = _save_upload(dataset, job_dir / "input" / "dataset.zip")
            try:
                dataset_root = extract_dataset(zip_path, job_dir / "input" / "dataset")
            except (zipfile.BadZipFile, ValueError) as e:
                raise HTTPException(status_code=422, detail=f"Invalid dataset zip: {e}") from e

        config = rewrite_custom_paths(
            user_config, rootdir=job_dir / "output", dataset_root=dataset_root,
        )
        config_path = _write_yaml(config, job_dir)
        return _genie3_argv(config_path, num_devices)

    return app.state.runner.submit(
        build_argv=_build, label="custom",
        input_params={"config_yaml": "(user-supplied)", "num_devices": num_devices},
    )


# ---------------------------------------------------------------------------
# Task endpoints (atomic; for FC Async Task Mode)
# ---------------------------------------------------------------------------


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/generate/unconditional", response_model=JobInfo)
    def generate_unconditional_task(
        request: Request,
        params: UnconditionalRequest = Depends(model_form_depends(UnconditionalRequest)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Unconditional protein backbone generation as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            config = build_unconditional_config(rootdir=job_dir / "output", req=req)
            config_path = _write_yaml(config, job_dir)
            return _genie3_argv(config_path, req.num_devices)

        return execute_task(
            request,
            job_id=job_id,
            label="unconditional",
            params=params,
            build_argv=_build,
        )

    @app.post("/api/tasks/generate/motif", response_model=JobInfo)
    def generate_motif_task(
        request: Request,
        dataset: Optional[UploadFile] = File(None, description="Zip with problems/ + motifs/."),
        dataset_uri: Optional[str] = Form(
            None,
            description="URI to the dataset zip (oss://, file://, job://, http(s)://) "
                        "as an alternative to a multipart upload — used by the gateway.",
        ),
        params: MotifRequest = Depends(model_form_depends(MotifRequest)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Motif scaffolding generation as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        dataset_state: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            zip_path = resolve_input(
                dataset, dataset_uri, input_dir / "dataset.zip", settings, "dataset",
            )
            try:
                dataset_state["root"] = extract_dataset(zip_path, input_dir / "dataset")
            except (zipfile.BadZipFile, ValueError) as e:
                raise HTTPException(status_code=422, detail=f"Invalid dataset zip: {e}") from e

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            config = build_motif_config(
                rootdir=job_dir / "output", dataset_root=dataset_state["root"], req=req,
            )
            config_path = _write_yaml(config, job_dir)
            return _genie3_argv(config_path, req.num_devices)

        return execute_task(
            request,
            job_id=job_id,
            label="motif",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )

    @app.post("/api/tasks/generate/binder", response_model=JobInfo)
    def generate_binder_task(
        request: Request,
        dataset: Optional[UploadFile] = File(None, description="Zip with problems/ + targets/."),
        dataset_uri: Optional[str] = Form(
            None,
            description="URI to the dataset zip (oss://, file://, job://, http(s)://) "
                        "as an alternative to a multipart upload — used by the gateway.",
        ),
        params: BinderRequest = Depends(model_form_depends(BinderRequest)),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Binder design generation as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        dataset_state: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            zip_path = resolve_input(
                dataset, dataset_uri, input_dir / "dataset.zip", settings, "dataset",
            )
            try:
                dataset_state["root"] = extract_dataset(zip_path, input_dir / "dataset")
            except (zipfile.BadZipFile, ValueError) as e:
                raise HTTPException(status_code=422, detail=f"Invalid dataset zip: {e}") from e

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            config = build_binder_config(
                rootdir=job_dir / "output", dataset_root=dataset_state["root"], req=req,
            )
            config_path = _write_yaml(config, job_dir)
            return _genie3_argv(config_path, req.num_devices)

        return execute_task(
            request,
            job_id=job_id,
            label="binder",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )

    @app.post("/api/tasks/generate", response_model=JobInfo)
    def generate_custom_task(
        request: Request,
        config_yaml: str = Form(..., description="Full experiment YAML as a string."),
        dataset: Optional[UploadFile] = File(None),
        dataset_uri: Optional[str] = Form(
            None,
            description="URI to an optional dataset zip (oss://, file://, job://, "
                        "http(s)://) as an alternative to a multipart upload.",
        ),
        num_devices: Optional[int] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Custom YAML generation as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        # Parse + validate user YAML up-front so 422s never allocate a job.
        try:
            user_config = yaml.safe_load(config_yaml)
        except yaml.YAMLError as e:
            raise HTTPException(status_code=422, detail=f"Invalid YAML: {e}") from e
        if not isinstance(user_config, dict):
            raise HTTPException(
                status_code=422, detail="config_yaml must be a mapping at the top level",
            )

        dataset_state: dict[str, Optional[Path]] = {"root": None}

        # The custom endpoint takes no Pydantic request model — synthesize a tiny
        # echo model so execute_task can record `input_params` consistently.
        from pydantic import BaseModel as _BaseModel

        class _CustomEcho(_BaseModel):
            num_devices: Optional[int] = None
            config_yaml_summary: str = "(user-supplied)"

        params_echo = _CustomEcho(num_devices=num_devices)

        def _save(_req, input_dir: Path) -> None:
            zip_path = maybe_resolve_input(
                dataset, dataset_uri, input_dir / "dataset.zip", settings, "dataset",
            )
            if zip_path is not None:
                try:
                    dataset_state["root"] = extract_dataset(zip_path, input_dir / "dataset")
                except (zipfile.BadZipFile, ValueError) as e:
                    raise HTTPException(status_code=422, detail=f"Invalid dataset zip: {e}") from e

        def _build(_req, _job_id: str, job_dir: Path) -> list[str]:
            config = rewrite_custom_paths(
                user_config, rootdir=job_dir / "output", dataset_root=dataset_state["root"],
            )
            config_path = _write_yaml(config, job_dir)
            return _genie3_argv(config_path, num_devices)

        return execute_task(
            request,
            job_id=job_id,
            label="custom",
            params=params_echo,
            build_argv=_build,
            save_inputs=_save,
        )


# Mount MCP server — must be AFTER all POST routes are registered so the
# auto-discovery walk sees the full surface.
attach_mcp(app)
