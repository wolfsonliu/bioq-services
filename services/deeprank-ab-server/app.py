"""FastAPI app for deeprank-ab-server.

Exposes /api/score for scoring antibody-antigen docking complexes.
Job lifecycle endpoints (/healthz, /api/jobs/*, /api/manifest, /openapi.json)
come from `bioagent_service.create_app`.
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
    resolve_task_id,
)
from fastapi import Depends, File, Form, Header, Request, UploadFile

from .adapter import DeepRankAbAdapter
from .models import ScoreRequest
from .settings import DeepRankAbSettings
from .argv import score_argv
from .uris import resolve_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = DeepRankAbSettings()
adapter = DeepRankAbAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="DeepRank-Ab Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# Remove framework's generic /healthz/detail so our deeprank-ab-specific
# weights probe takes over. FastAPI >=0.115 wraps included routers in
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
    """Extended health: report whether NAS-mounted ESM-2 weights are reachable.

    Weights live on NAS at `DEEPRANK_AB_WEIGHTS_DIR` (default
    `/data/models/deeprank-ab/esm/`).  Probes the two expected files;
    `weights_loaded=false` lets the agent detect a misconfigured FC mount /
    unbound SIF without crashing the service.
    """
    expected = {
        "esm2_t33_650M_UR50D.pt":
            settings.weights_dir / "esm2_t33_650M_UR50D.pt",
        "esm2_t33_650M_UR50D-contact-regression.pt":
            settings.weights_dir / "esm2_t33_650M_UR50D-contact-regression.pt",
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


@app.post("/api/score", response_model=JobInfo)
def post_score(
    params: ScoreRequest = Depends(model_form_depends(ScoreRequest)),
    input_pdb: Optional[UploadFile] = File(None),
    input_pdb_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Score an antibody-antigen docking complex.

    Runs the DeepRank-Ab EGNN pipeline: PDB processing, ESM-2 embeddings,
    ANARCI CDR annotation, atom-level graph construction, MCL clustering,
    and EGNN inference. Returns predicted DockQ scores as a CSV.
    """

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        input_dir = job_dir / "input"
        pdb_path = resolve_input(input_pdb, input_pdb_uri, input_dir / "input.pdb", settings)
        return score_argv(
            params,
            job_dir=job_dir,
            pdb_path=pdb_path,
            settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build,
        label="score",
        input_params=params.model_dump(mode="json"),
    )


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/score", response_model=JobInfo)
    def post_score_task(
        request: Request,
        params: ScoreRequest = Depends(model_form_depends(ScoreRequest)),
        input_pdb: Optional[UploadFile] = File(None),
        input_pdb_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Score an antibody-antigen docking complex as a single atomic task.

        Blocks until the DeepRank-Ab pipeline (PDB processing → ESM-2 embeddings →
        ANARCI → graph → MCL → EGNN) completes.  Designed for FC Async Task Mode
        invocation; the submit/poll interface is at POST /api/score.
        """
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        pdb_paths: list[Path] = []

        def _save(_req, input_dir: Path) -> None:
            pdb_paths.append(
                resolve_input(input_pdb, input_pdb_uri, input_dir / "input.pdb", settings)
            )

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return score_argv(req, job_dir=job_dir, pdb_path=pdb_paths[0], settings=settings)

        return execute_task(
            request,
            job_id=job_id,
            label="score",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )


attach_mcp(app)
