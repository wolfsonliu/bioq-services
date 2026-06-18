"""FastAPI app + service-specific POST routes for promera-server.

Framework provides /healthz, /api/jobs/*, /api/manifest, /openapi.json.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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

from .adapter import PromeraAdapter
from .models import CofoldRequest, DesignRequest
from .settings import PromeraSettings
from .tools import (
    build_design_config,
    cofold_argv,
    design_argv,
    write_design_config,
)
from .uris import resolve_input, save_upload

settings = PromeraSettings()
adapter = PromeraAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="Promera Server",
    version=read_version_file(__file__, default="0.0.1"),
)


@app.post("/api/cofold", response_model=JobInfo)
def cofold(
    params: CofoldRequest = Depends(model_form_depends(CofoldRequest)),
    input_schema: Optional[UploadFile] = File(None),
    input_schema_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Structure prediction (cofolding) endpoint."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        schema_path = resolve_input(
            input_schema, input_schema_uri, input_dir / "input.json", settings
        )
        return cofold_argv(
            params, job_dir=job_dir, schema_path=schema_path, settings=settings
        )

    return app.state.runner.submit(
        build_argv=_build,
        label="cofold",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/design", response_model=JobInfo)
def design(
    params: DesignRequest = Depends(model_form_depends(DesignRequest)),
    target_schema: Optional[UploadFile] = File(None),
    target_schema_uri: Optional[str] = Form(None),
    target_template: Optional[UploadFile] = File(None),
) -> JobInfo:
    """De novo binder design endpoint."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        input_dir = job_dir / "input"
        target_dir = input_dir / "targets"
        target_dir.mkdir(parents=True, exist_ok=True)
        output_dir = job_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        resolve_input(
            target_schema, target_schema_uri, target_dir / "target.json", settings
        )

        template_path = None
        if target_template is not None:
            template_path = input_dir / "target_template.cif"
            save_upload(target_template, template_path)

        cfg = build_design_config(
            params,
            target_dir=target_dir,
            output_dir=output_dir,
            template_path=template_path,
            settings=settings,
        )
        config_path = write_design_config(cfg, input_dir / "task_config.yaml")

        return design_argv(
            params, job_dir=job_dir, config_path=config_path, settings=settings
        )

    return app.state.runner.submit(
        build_argv=_build,
        label="design",
        input_params=params.model_dump(mode="json"),
    )


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/cofold", response_model=JobInfo)
    def cofold_task(
        request: Request,
        params: CofoldRequest = Depends(model_form_depends(CofoldRequest)),
        input_schema: Optional[UploadFile] = File(None),
        input_schema_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Cofold as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["schema"] = resolve_input(
                input_schema, input_schema_uri, input_dir / "input.json", settings
            )

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return cofold_argv(req, job_dir=job_dir, schema_path=paths["schema"], settings=settings)

        return execute_task(
            request, job_id=job_id, label="cofold", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/design", response_model=JobInfo)
    def design_task(
        request: Request,
        params: DesignRequest = Depends(model_form_depends(DesignRequest)),
        target_schema: Optional[UploadFile] = File(None),
        target_schema_uri: Optional[str] = Form(None),
        target_template: Optional[UploadFile] = File(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """De novo design as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        state: dict[str, Optional[Path]] = {"template": None}

        def _save(_req, input_dir: Path) -> None:
            target_dir = input_dir / "targets"
            target_dir.mkdir(parents=True, exist_ok=True)
            resolve_input(
                target_schema, target_schema_uri, target_dir / "target.json", settings
            )
            if target_template is not None:
                state["template"] = input_dir / "target_template.cif"
                save_upload(target_template, state["template"])

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            input_dir = job_dir / "input"
            target_dir = input_dir / "targets"
            output_dir = job_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            cfg = build_design_config(
                req,
                target_dir=target_dir,
                output_dir=output_dir,
                template_path=state["template"],
                settings=settings,
            )
            config_path = write_design_config(cfg, input_dir / "task_config.yaml")

            return design_argv(req, job_dir=job_dir, config_path=config_path, settings=settings)

        return execute_task(
            request, job_id=job_id, label="design", params=params,
            build_argv=_build, save_inputs=_save,
        )


attach_mcp(app)
