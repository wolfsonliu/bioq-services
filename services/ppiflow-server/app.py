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

from bioagent_service import JobInfo, attach_mcp, create_app, model_form_depends
from fastapi import Depends, File, Form, UploadFile

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
    version="0.0.1",
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

    return app.state.runner.submit(build_argv=_build, label="binder")


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

    return app.state.runner.submit(build_argv=_build, label="antibody")


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

    return app.state.runner.submit(build_argv=_build, label="nanobody")


@app.post("/api/sample/monomer", response_model=JobInfo)
def sample_monomer(
    params: MonomerRequest = Depends(model_form_depends(MonomerRequest)),
) -> JobInfo:
    """Unconditional monomer generation at the requested lengths."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        return monomer_argv(params, job_dir, settings)

    return app.state.runner.submit(build_argv=_build, label="monomer")


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

    return app.state.runner.submit(build_argv=_build, label="scaffolding")


# Mount MCP server — must be AFTER all POST routes are registered so the
# auto-discovery walk sees the full surface.
attach_mcp(app)
