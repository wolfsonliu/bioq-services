"""FastAPI app for boltzgen-server.

Exposes /api/design and /api/inverse_fold for BoltzGen binder design pipeline.
Job lifecycle endpoints (/healthz, /api/jobs/*, /api/manifest, /openapi.json)
come from `bioagent_service.create_app`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from bioagent_service import (
    JobInfo,
    attach_mcp,
    create_app,
    execute_task,
    model_form_depends,
    read_version_file,
    resolve_task_id,
)
from fastapi import Depends, File, Form, Header, Request, UploadFile

from .adapter import BoltzGenAdapter
from .models import DesignRequest, InverseFoldRequest
from .settings import BoltzGenSettings
from .tools import design_argv, inverse_fold_argv
from .uris import resolve_input, save_upload

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


def _save_inputs(
    design_yaml: Optional[UploadFile],
    design_yaml_uri: Optional[str],
    ref_files: List[UploadFile],
    input_dir: Path,
) -> Path:
    """Save the design spec YAML and reference files to the job input dir."""
    input_dir.mkdir(parents=True, exist_ok=True)
    yaml_dest = input_dir / "design_spec.yaml"
    yaml_path = resolve_input(design_yaml, design_yaml_uri, yaml_dest, settings)

    for rf in ref_files:
        if rf.filename:
            save_upload(rf, input_dir / rf.filename)

    return yaml_path


@app.post("/api/design", response_model=JobInfo)
def post_design(
    params: DesignRequest = Depends(model_form_depends(DesignRequest)),
    design_yaml: Optional[UploadFile] = File(None),
    design_yaml_uri: Optional[str] = Form(None),
    ref_files: List[UploadFile] = File([]),
) -> JobInfo:
    """Run the full BoltzGen binder design pipeline.

    Requires a design specification YAML describing the target structure and
    designed region. Reference CIF/PDB files referenced in the YAML should be
    uploaded via `ref_files`.

    Pipeline: design -> inverse_folding -> folding -> [design_folding] ->
    [affinity] -> analysis -> filtering.
    """

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        yaml_path = _save_inputs(design_yaml, design_yaml_uri, ref_files, job_dir / "input")
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
) -> JobInfo:
    """Run BoltzGen in inverse-fold-only mode.

    Skips the design diffusion step; runs inverse_folding -> folding ->
    analysis -> filtering on a provided backbone structure.
    """

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        yaml_path = _save_inputs(design_yaml, design_yaml_uri, ref_files, job_dir / "input")
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
            yaml_paths.append(_save_inputs(design_yaml, design_yaml_uri, ref_files, input_dir))

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
            yaml_paths.append(_save_inputs(design_yaml, design_yaml_uri, ref_files, input_dir))

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
