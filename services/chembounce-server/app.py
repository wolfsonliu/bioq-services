"""FastAPI app for chembounce-server.

Exposes `/api/scaffold_hop` (submit/poll) + `/api/tasks/scaffold_hop`
(FC async task mode).  Job lifecycle endpoints (/healthz, /api/jobs/*,
/api/manifest, /openapi.json) come from `bioagent_service.create_app`.
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
from fastapi import Depends, Form, Header, Request

from .adapter import ChemBounceAdapter
from .models import ScaffoldHopRequest
from .settings import ChemBounceSettings
from .tools import scaffold_hop_argv
from .uris import resolve_smiles_uri

logger = logging.getLogger(__name__)

settings = ChemBounceSettings()
adapter = ChemBounceAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="ChemBounce Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---------------------------------------------------------------------------
# /healthz/detail override — surface NAS-mounted data DB presence.
# ---------------------------------------------------------------------------

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
    """Probe whether the scaffold + fingerprint DB files are reachable on NAS.

    Reports both 250mw and full as separate flags; service can run with just
    one of them (250mw is required by default), full is bonus.
    """
    db250 = {
        "scaffolds_250mw.txt": settings.scaffold_db_250mw,
        "scaffold_fingerprints_250mw.npz": settings.fingerprint_250mw,
    }
    db_full = {
        "scaffolds.txt": settings.scaffold_db_full,
        "scaffold_fingerprints.npz": settings.fingerprint_full,
    }
    missing_250mw = {k: str(p) for k, p in db250.items() if not p.exists()}
    missing_full = {k: str(p) for k, p in db_full.items() if not p.exists()}
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "weights_dir": str(settings.weights_dir),
        # Service usable iff at least 250mw is present (the default db).
        "weights_loaded": not missing_250mw,
        "weights_missing": {**missing_250mw, **missing_full},
        "database_status": {
            "250mw": not missing_250mw,
            "full": not missing_full,
        },
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_smiles(
    params: ScaffoldHopRequest,
    input_smiles_uri: Optional[str],
) -> ScaffoldHopRequest:
    """Substitute `input_smiles` from a URI if the form field was empty / URI given."""
    if input_smiles_uri:
        smiles = resolve_smiles_uri(input_smiles_uri, settings)
        return params.model_copy(update={"input_smiles": smiles})
    return params


def _persist_smiles(input_smiles: str, input_dir: Path) -> Path:
    """Save the SMILES that was actually submitted to <job_dir>/input/."""
    input_dir.mkdir(parents=True, exist_ok=True)
    p = input_dir / "input_smiles.txt"
    p.write_text(input_smiles + "\n")
    return p


# ---------------------------------------------------------------------------
# /api/scaffold_hop (submit/poll)
# ---------------------------------------------------------------------------


@app.post("/api/scaffold_hop", response_model=JobInfo)
def post_scaffold_hop(
    params: ScaffoldHopRequest = Depends(model_form_depends(ScaffoldHopRequest)),
    input_smiles_uri: Optional[str] = Form(default=None),
) -> JobInfo:
    """Run ChemBounce scaffold hopping.  Returns JobInfo; poll until completed."""

    resolved = _resolve_smiles(params, input_smiles_uri)

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        _persist_smiles(resolved.input_smiles, job_dir / "input")
        return scaffold_hop_argv(resolved, job_dir=job_dir, settings=settings)

    return app.state.runner.submit(
        build_argv=_build,
        label="scaffold_hop",
        input_params=resolved.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# /api/tasks/scaffold_hop (FC async task mode)
# ---------------------------------------------------------------------------


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/scaffold_hop", response_model=JobInfo)
    def post_scaffold_hop_task(
        request: Request,
        params: ScaffoldHopRequest = Depends(model_form_depends(ScaffoldHopRequest)),
        input_smiles_uri: Optional[str] = Form(default=None),
        x_bioagent_job_id: Optional[str] = Header(
            default=None, alias="X-Bioagent-Job-Id"
        ),
        x_fc_async_task_id: Optional[str] = Header(
            default=None, alias="X-Fc-Async-Task-Id"
        ),
    ) -> JobInfo:
        """FC async task mode — blocks until done; HTTP 202 on submit."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        resolved = _resolve_smiles(params, input_smiles_uri)

        def _save(_req: ScaffoldHopRequest, input_dir: Path) -> None:
            _persist_smiles(resolved.input_smiles, input_dir)

        def _build(req: ScaffoldHopRequest, _job_id: str, job_dir: Path) -> list[str]:
            # The framework hands us the params we registered; we already
            # have `resolved` from URI resolution, so prefer that.
            return scaffold_hop_argv(resolved, job_dir=job_dir, settings=settings)

        return execute_task(
            request,
            job_id=job_id,
            label="scaffold_hop",
            params=resolved,
            build_argv=_build,
            save_inputs=_save,
        )


# Must come AFTER all POST routes so MCP auto-discovery sees the full surface.
attach_mcp(app)
