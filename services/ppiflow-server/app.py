"""FastAPI app for ppiflow-server.

Exposes the five PPIFlow structure-generation endpoints. Sequence design,
side-chain packing, scoring, and Rosetta refinement live in their own
bioagent services; this server is intentionally scoped to PPIFlow's own
sampling capability.

Lifecycle (status / log / download / single-file / delete / manifest /
openapi) is contributed by `bioagent_service.create_app`. See
`engineering/guides/calling-bioagent-services.md` for the call protocol.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from bioagent_service import (
    JobInfo,
    attach_mcp,
    create_app,
    execute_task,
    model_form_depends,
    read_version_file,
    register_task_endpoint,
    resolve_task_id,
)
from fastapi import Depends, File, Form, Header, Request, UploadFile

from .adapter import PPIFlowAdapter
from .models import (
    AntibodyRequest,
    BinderRequest,
    MonomerRequest,
    NanobodyRequest,
    ScaffoldingRequest,
)
from .settings import PPIFlowSettings
from .tools import (
    antibody_argv,
    binder_argv,
    monomer_argv,
    nanobody_argv,
    scaffolding_argv,
)
from .uris import resolve_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = PPIFlowSettings()
adapter = PPIFlowAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="PPIFlow Server",
    version=read_version_file(__file__, default="0.0.1"),
)


@app.post("/api/sample/binder", response_model=JobInfo)
def sample_binder(
    params: BinderRequest = Depends(model_form_depends(BinderRequest)),
    target: Optional[UploadFile] = File(
        None, description="Target PDB. Mutually exclusive with `target_uri`.",
    ),
    target_uri: Optional[str] = Form(
        None,
        description="URI to fetch target instead of uploading (job:// / file:// / oss:// / http(s)://).",
    ),
) -> JobInfo:
    """PPI binder design against an uploaded (or URI-referenced) target PDB."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        target_pdb = resolve_input(target, target_uri, job_dir / "input" / "target.pdb", settings)
        return binder_argv(params, target_pdb, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="binder",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/sample/antibody", response_model=JobInfo)
def sample_antibody(
    params: AntibodyRequest = Depends(model_form_depends(AntibodyRequest)),
    antigen: Optional[UploadFile] = File(None),
    antigen_uri: Optional[str] = Form(None),
    framework: Optional[UploadFile] = File(None),
    framework_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Antibody (heavy + light) CDR design over an uploaded framework."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        antigen_pdb = resolve_input(antigen, antigen_uri, job_dir / "input" / "antigen.pdb", settings)
        framework_pdb = resolve_input(framework, framework_uri, job_dir / "input" / "framework.pdb", settings)
        return antibody_argv(params, antigen_pdb, framework_pdb, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="antibody",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/sample/nanobody", response_model=JobInfo)
def sample_nanobody(
    params: NanobodyRequest = Depends(model_form_depends(NanobodyRequest)),
    antigen: Optional[UploadFile] = File(None),
    antigen_uri: Optional[str] = Form(None),
    framework: Optional[UploadFile] = File(None),
    framework_uri: Optional[str] = Form(None),
) -> JobInfo:
    """VHH (heavy-only) CDR design over an uploaded nanobody framework."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        antigen_pdb = resolve_input(antigen, antigen_uri, job_dir / "input" / "antigen.pdb", settings)
        framework_pdb = resolve_input(framework, framework_uri, job_dir / "input" / "framework.pdb", settings)
        return nanobody_argv(params, antigen_pdb, framework_pdb, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="nanobody",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/sample/monomer", response_model=JobInfo)
def sample_monomer(
    params: MonomerRequest = Depends(model_form_depends(MonomerRequest)),
) -> JobInfo:
    """Unconditional monomer generation at the requested lengths."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        return monomer_argv(params, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="monomer",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/sample/scaffolding", response_model=JobInfo)
def sample_scaffolding(
    params: ScaffoldingRequest = Depends(model_form_depends(ScaffoldingRequest)),
    motif_csv: Optional[UploadFile] = File(
        None, description="Motif metadata CSV (target,length,contig,motif_path).",
    ),
    motif_csv_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Motif scaffolding from a CSV + motif PDB(s) (uses monomer.ckpt)."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        csv_path = resolve_input(motif_csv, motif_csv_uri, job_dir / "input" / "motif_metadata.csv", settings)
        return scaffolding_argv(params, csv_path, job_dir, settings)

    return app.state.runner.submit(
        build_argv=_build, label="scaffolding",
        input_params=params.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# Task endpoints (synchronous; FC Async Task Mode-friendly)
# ---------------------------------------------------------------------------

if settings.task_endpoints_enabled:

    @app.post("/api/tasks/sample/binder", response_model=JobInfo)
    def sample_binder_task(
        request: Request,
        params: BinderRequest = Depends(model_form_depends(BinderRequest)),
        target: Optional[UploadFile] = File(None),
        target_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """PPI binder design as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["target"] = resolve_input(target, target_uri, input_dir / "target.pdb", settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return binder_argv(req, paths["target"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="binder", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/sample/antibody", response_model=JobInfo)
    def sample_antibody_task(
        request: Request,
        params: AntibodyRequest = Depends(model_form_depends(AntibodyRequest)),
        antigen: Optional[UploadFile] = File(None),
        antigen_uri: Optional[str] = Form(None),
        framework: Optional[UploadFile] = File(None),
        framework_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Antibody CDR design as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["antigen"] = resolve_input(antigen, antigen_uri, input_dir / "antigen.pdb", settings)
            paths["framework"] = resolve_input(framework, framework_uri, input_dir / "framework.pdb", settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return antibody_argv(req, paths["antigen"], paths["framework"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="antibody", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/sample/nanobody", response_model=JobInfo)
    def sample_nanobody_task(
        request: Request,
        params: NanobodyRequest = Depends(model_form_depends(NanobodyRequest)),
        antigen: Optional[UploadFile] = File(None),
        antigen_uri: Optional[str] = Form(None),
        framework: Optional[UploadFile] = File(None),
        framework_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """VHH (nanobody) CDR design as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["antigen"] = resolve_input(antigen, antigen_uri, input_dir / "antigen.pdb", settings)
            paths["framework"] = resolve_input(framework, framework_uri, input_dir / "framework.pdb", settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return nanobody_argv(req, paths["antigen"], paths["framework"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="nanobody", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/sample/scaffolding", response_model=JobInfo)
    def sample_scaffolding_task(
        request: Request,
        params: ScaffoldingRequest = Depends(model_form_depends(ScaffoldingRequest)),
        motif_csv: Optional[UploadFile] = File(None),
        motif_csv_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Motif scaffolding as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["csv"] = resolve_input(motif_csv, motif_csv_uri, input_dir / "motif_metadata.csv", settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return scaffolding_argv(req, paths["csv"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="scaffolding", params=params,
            build_argv=_build, save_inputs=_save,
        )


# Monomer endpoint has no uploads → use the simpler register_task_endpoint helper.
# It internally honors settings.task_endpoints_enabled (no outer if guard needed).

def _monomer_build(req, _job_id: str, job_dir: Path) -> list[str]:
    return monomer_argv(req, job_dir, settings)


register_task_endpoint(
    app,
    path="/api/tasks/sample/monomer",
    label="monomer",
    request_model=MonomerRequest,
    build_argv=_monomer_build,
)


# Mount MCP server — must be AFTER all POST routes are registered so the
# auto-discovery walk sees the full surface.
attach_mcp(app)
