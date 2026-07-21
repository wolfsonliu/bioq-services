"""FastAPI app for boltzgen-server.

Exposes /api/design and /api/inverse_fold for BoltzGen binder design pipeline.
Job lifecycle endpoints (/healthz, /api/jobs/*, /api/manifest, /openapi.json)
come from `bioq_service.create_app`.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import List, Optional

from bioq_service import (
    JobInfo,
    attach_mcp,
    create_app,
    execute_task,
    model_form_depends,
    read_version_file,
    resolve_task_id,
)
from fastapi import Depends, File, Form, Header, HTTPException, Request, UploadFile

from .adapter import BoltzGenAdapter
from .models import DesignRequest, InverseFoldRequest
from .settings import BoltzGenSettings
from .tools import design_argv, inverse_fold_argv
from bioq_service.uris import resolve_input, resolve_uri, save_upload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = BoltzGenSettings()
adapter = BoltzGenAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="BoltzGen Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# Remove framework's generic /healthz/detail so we can override it with
# boltzgen-specific signals (weights/moldir presence on NAS).  FastAPI uses
# first-match routing, so without this our handler below would be shadowed.
# FastAPI >=0.115 wraps included routers in `_IncludedRouter`; descend into
# them to find the framework's route.
def _strip_route(router, path: str, method: str) -> None:
    router.routes = [
        r
        for r in router.routes
        if not (
            getattr(r, "path", None) == path
            and method in getattr(r, "methods", set())
        )
    ]
    for r in router.routes:
        inner = getattr(r, "original_router", None)
        if inner is not None:
            _strip_route(inner, path, method)


_strip_route(app.router, "/healthz/detail", "GET")


@app.get("/healthz/detail")
def healthz_detail(request: Request) -> dict:
    """Extended health: surface whether NAS-mounted weights are reachable.

    Weights live on NAS at `BOLTZGEN_WEIGHTS_DIR` + `BOLTZGEN_MOLDIR`
    (default `/data/models/boltzgen/{weights,moldir}/`).  Reports missing
    paths via `weights_missing` so the agent can detect a misconfigured FC
    mount / unbound SIF without crashing the service.
    """
    expected = {
        "weights_dir": settings.weights_dir,
        "moldir": settings.moldir,
    }
    missing = {k: str(p) for k, p in expected.items() if not p.exists()}
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "weights_dir": str(settings.weights_dir),
        "moldir": str(settings.moldir),
        "weights_loaded": not missing,
        "weights_missing": missing,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


def _extract_ref_zip(zip_uri: str, input_dir: Path) -> None:
    """Resolve a zip of reference structures and extract it flat into input_dir.

    The gateway dispatches form fields only and cannot multipart-upload the
    `ref_files` list, so it passes a single zip via `ref_files_zip_uri` instead.
    Members are written by basename next to design_spec.yaml because boltzgen
    resolves `file: path:` relative to the spec's own directory (schema.py:
    base_file_path=path.parent). Flat extraction also neutralizes zip-slip.
    """
    zip_dest = input_dir / "_ref_files.zip"
    resolve_uri(zip_uri, zip_dest, settings)
    try:
        with zipfile.ZipFile(zip_dest, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                name = Path(member.filename).name
                if name:
                    (input_dir / name).write_bytes(zf.read(member))
    except zipfile.BadZipFile as exc:
        raise HTTPException(422, f"Invalid ref_files zip: {exc}") from exc
    finally:
        zip_dest.unlink(missing_ok=True)


def _save_inputs(
    design_yaml: Optional[UploadFile],
    design_yaml_uri: Optional[str],
    ref_files: List[UploadFile],
    ref_files_zip_uri: Optional[str],
    input_dir: Path,
) -> Path:
    """Save the design spec YAML and reference files to the job input dir."""
    input_dir.mkdir(parents=True, exist_ok=True)
    yaml_dest = input_dir / "design_spec.yaml"
    yaml_path = resolve_input(design_yaml, design_yaml_uri, yaml_dest, settings)

    for rf in ref_files:
        if rf.filename:
            save_upload(rf, input_dir / rf.filename)

    if ref_files_zip_uri:
        _extract_ref_zip(ref_files_zip_uri, input_dir)

    return yaml_path


@app.post("/api/design", response_model=JobInfo)
def post_design(
    params: DesignRequest = Depends(model_form_depends(DesignRequest)),
    design_yaml: Optional[UploadFile] = File(None),
    design_yaml_uri: Optional[str] = Form(None),
    ref_files: List[UploadFile] = File([]),
    ref_files_zip_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Run the full BoltzGen binder design pipeline.

    Requires a design specification YAML describing the target structure and
    designed region. Reference CIF/PDB files referenced in the YAML should be
    uploaded via `ref_files`, or bundled into a zip referenced by
    `ref_files_zip_uri` (oss://, file://, job://, http(s)://) — the gateway
    uses the zip form since it can't multipart-upload the `ref_files` list.

    Pipeline: design -> inverse_folding -> folding -> [design_folding] ->
    [affinity] -> analysis -> filtering.
    """

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        yaml_path = _save_inputs(
            design_yaml, design_yaml_uri, ref_files, ref_files_zip_uri, job_dir / "input"
        )
        return design_argv(params, job_dir=job_dir, yaml_path=yaml_path, settings=settings)

    return app.state.runner.submit(
        build_argv=_build,
        label="design",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/inverse_fold", response_model=JobInfo)
def post_inverse_fold(
    params: InverseFoldRequest = Depends(model_form_depends(InverseFoldRequest)),
    design_yaml: Optional[UploadFile] = File(None),
    design_yaml_uri: Optional[str] = Form(None),
    ref_files: List[UploadFile] = File([]),
    ref_files_zip_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Run BoltzGen in inverse-fold-only mode.

    Skips the design diffusion step; runs inverse_folding -> folding ->
    analysis -> filtering on a provided backbone structure. The backbone CIF/PDB
    referenced by the YAML comes via `ref_files` (upload) or `ref_files_zip_uri`
    (zip URI — used by the gateway).
    """

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        yaml_path = _save_inputs(
            design_yaml, design_yaml_uri, ref_files, ref_files_zip_uri, job_dir / "input"
        )
        return inverse_fold_argv(params, job_dir=job_dir, yaml_path=yaml_path, settings=settings)

    return app.state.runner.submit(
        build_argv=_build,
        label="inverse_fold",
        input_params=params.model_dump(mode="json"),
    )


if settings.task_endpoints_enabled:
    @app.post("/api/tasks/design", response_model=JobInfo)
    def post_design_task(
        request: Request,
        params: DesignRequest = Depends(model_form_depends(DesignRequest)),
        design_yaml: Optional[UploadFile] = File(None),
        design_yaml_uri: Optional[str] = Form(None),
        ref_files: List[UploadFile] = File([]),
        ref_files_zip_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Run the full BoltzGen design pipeline as a single atomic task.

        Blocks until pipeline completion.  Designed to be invoked via FC Async
        Task Mode (X-Fc-Invocation-Type: Async): FC enqueues and dispatches,
        the HTTP request stays active for the full subprocess lifetime so FC
        won't recycle the instance mid-run.

        For the legacy submit/poll interface, use POST /api/design instead.
        """
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        yaml_paths: list[Path] = []

        def _save(_req, input_dir: Path) -> None:
            yaml_paths.append(
                _save_inputs(design_yaml, design_yaml_uri, ref_files, ref_files_zip_uri, input_dir)
            )

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return design_argv(req, job_dir=job_dir, yaml_path=yaml_paths[0], settings=settings)

        return execute_task(
            request,
            job_id=job_id,
            label="design",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )

    @app.post("/api/tasks/inverse_fold", response_model=JobInfo)
    def post_inverse_fold_task(
        request: Request,
        params: InverseFoldRequest = Depends(model_form_depends(InverseFoldRequest)),
        design_yaml: Optional[UploadFile] = File(None),
        design_yaml_uri: Optional[str] = Form(None),
        ref_files: List[UploadFile] = File([]),
        ref_files_zip_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Run BoltzGen inverse-fold-only mode as a single atomic task.

        Same lifecycle as /api/tasks/design but skips the design diffusion
        step; runs inverse_folding -> folding -> analysis -> filtering.
        """
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        yaml_paths: list[Path] = []

        def _save(_req, input_dir: Path) -> None:
            yaml_paths.append(
                _save_inputs(design_yaml, design_yaml_uri, ref_files, ref_files_zip_uri, input_dir)
            )

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return inverse_fold_argv(req, job_dir=job_dir, yaml_path=yaml_paths[0], settings=settings)

        return execute_task(
            request,
            job_id=job_id,
            label="inverse_fold",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )


attach_mcp(app)
