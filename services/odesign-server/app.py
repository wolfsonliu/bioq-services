"""FastAPI app for odesign-server.

Exposes /api/design for ODesign cross-modality biomolecular interaction design.
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

from .adapter import ODesignAdapter
from .models import DesignRequest
from .settings import ODesignSettings
from .tools import design_argv
from .uris import resolve_input, rewrite_ref_files, save_upload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = ODesignSettings()
adapter = ODesignAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="ODesign Server",
    version=read_version_file(__file__, default="0.0.1"),
)


def _save_inputs(
    input_json: Optional[UploadFile],
    input_json_uri: Optional[str],
    ref_files: List[UploadFile],
    input_dir: Path,
) -> Path:
    """Save the JSON spec and reference files to the job input dir."""
    input_dir.mkdir(parents=True, exist_ok=True)
    json_dest = input_dir / "input.json"
    json_path = resolve_input(input_json, input_json_uri, json_dest, settings)

    for rf in ref_files:
        if rf.filename:
            save_upload(rf, input_dir / rf.filename)

    rewrite_ref_files(json_path, input_dir)
    return json_path


@app.post("/api/design", response_model=JobInfo)
def post_design(
    params: DesignRequest = Depends(model_form_depends(DesignRequest)),
    input_json: Optional[UploadFile] = File(None),
    input_json_uri: Optional[str] = Form(None),
    ref_files: List[UploadFile] = File([]),
) -> JobInfo:
    """Run ODesign biomolecular interaction design.

    Requires a JSON specification describing the target structure and design
    task. Reference CIF/PDB files referenced in the JSON should be uploaded
    via `ref_files`.

    Pipeline: input parsing -> conditional diffusion (backbone generation) ->
    inverse folding (sequence design) -> CIF output.
    """

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        json_path = _save_inputs(input_json, input_json_uri, ref_files, job_dir / "input")
        return design_argv(params, job_dir=job_dir, json_path=json_path, settings=settings)

    return app.state.runner.submit(
        build_argv=_build,
        label="design",
        input_params=params.model_dump(mode="json"),
    )


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/design", response_model=JobInfo)
    def post_design_task(
        request: Request,
        params: DesignRequest = Depends(model_form_depends(DesignRequest)),
        input_json: Optional[UploadFile] = File(None),
        input_json_uri: Optional[str] = Form(None),
        ref_files: List[UploadFile] = File([]),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Run ODesign biomolecular interaction design as a single atomic task.

        Blocks until pipeline completion (input parsing -> conditional diffusion ->
        inverse folding -> CIF output).  Designed for FC Async Task Mode.  For the
        submit/poll interface, use POST /api/design.
        """
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        json_paths: list[Path] = []

        def _save(_req, input_dir: Path) -> None:
            json_paths.append(_save_inputs(input_json, input_json_uri, ref_files, input_dir))

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return design_argv(req, job_dir=job_dir, json_path=json_paths[0], settings=settings)

        return execute_task(
            request,
            job_id=job_id,
            label="design",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )


attach_mcp(app)
