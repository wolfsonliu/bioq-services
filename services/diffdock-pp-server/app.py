"""FastAPI app for diffdock-pp-server.

Exposes `/api/dock` (submit/poll) + `/api/tasks/dock` (FC async task mode).
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

from .adapter import DiffDockPPAdapter
from .models import DockRequest
from .settings import DiffDockPPSettings
from .tools import dock_argv

logger = logging.getLogger(__name__)

settings = DiffDockPPSettings()
adapter = DiffDockPPAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="DiffDock-PP Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---------------------------------------------------------------------------
# /healthz/detail override — surface NAS-mounted weight presence.
# ---------------------------------------------------------------------------
# FastAPI >=0.115 wraps included routers in `_IncludedRouter`; recurse into
# them to drop the framework's generic /healthz/detail before ours runs.

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


def _expected_weight_files() -> dict[str, Path]:
    """Return `{label: expected_path}` for all files that must exist on NAS.

    Missing files show up in `/healthz/detail` as `weights_missing` so an
    agent can detect a bad NAS mount before the first inference crash.

    - Score / confidence model dirs need BOTH `model_best_*.pth` AND `args.yaml`
      (upstream `args.py:process_args` reads args.yaml back to construct the net).
    - ESM-2 checkpoint + esm source dir are required by `torch.hub.load(...)`
      in offline mode.
    """
    wd = settings.weights_dir
    score_dir = wd / "large_model_dips" / "fold_0"
    conf_dir = wd / "confidence_model_dips" / "fold_0"
    # Take any file matching the checkpoint glob — the specific filename hash
    # can differ between upstream releases; we only care one exists.
    score_ckpts = list(score_dir.glob("model_best_*.pth"))
    conf_ckpts = list(conf_dir.glob("model_best_*.pth"))
    expected: dict[str, Path] = {
        "score_checkpoint": score_ckpts[0] if score_ckpts else score_dir / "model_best_*.pth",
        "score_args_yaml": wd / "large_model_dips" / "args.yaml",
        "confidence_checkpoint": conf_ckpts[0] if conf_ckpts else conf_dir / "model_best_*.pth",
        "confidence_args_yaml": wd / "confidence_model_dips" / "args.yaml",
        "esm2_checkpoint": wd / "esm_cache" / "hub" / "checkpoints" / "esm2_t33_650M_UR50D.pt",
        "esm_source_dir": wd / "esm_cache" / "hub" / "facebookresearch_esm_main" / "hubconf.py",
    }
    return expected


@app.get("/healthz/detail")
def healthz_detail(request: Request) -> dict:
    """Probe whether score / confidence / ESM-2 checkpoints are reachable on NAS.

    Weights live at `DIFFDOCK_PP_WEIGHTS_DIR` (default
    `/data/models/diffdock-pp/`). Service starts even when weights are
    missing — the failure is surfaced here so an agent can detect a
    misconfigured FC mount / unbound SIF without crashing imports.
    """
    expected = _expected_weight_files()
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


# ---------------------------------------------------------------------------
# /api/dock (submit/poll)
# ---------------------------------------------------------------------------


def _save_inputs(
    receptor: Optional[UploadFile],
    receptor_uri: Optional[str],
    ligand: Optional[UploadFile],
    ligand_uri: Optional[str],
    input_dir: Path,
) -> tuple[Path, Path]:
    """Persist + URI-resolve the two required PDB inputs."""
    from bioagent_service.uris import resolve_input

    input_dir.mkdir(parents=True, exist_ok=True)

    receptor_dest = input_dir / (
        receptor.filename if receptor and receptor.filename else "receptor.pdb"
    )
    receptor_path = resolve_input(receptor, receptor_uri, receptor_dest, settings)

    ligand_dest = input_dir / (
        ligand.filename if ligand and ligand.filename else "ligand.pdb"
    )
    ligand_path = resolve_input(ligand, ligand_uri, ligand_dest, settings)
    return receptor_path, ligand_path


@app.post("/api/dock", response_model=JobInfo)
def post_dock(
    params: DockRequest = Depends(model_form_depends(DockRequest)),
    receptor: Optional[UploadFile] = File(None),
    receptor_uri: Optional[str] = Form(None),
    ligand: Optional[UploadFile] = File(None),
    ligand_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Run rigid protein-protein docking. Returns a JobInfo; poll until completed."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        receptor_path, ligand_path = _save_inputs(
            receptor, receptor_uri, ligand, ligand_uri, job_dir / "input",
        )
        return dock_argv(
            params,
            job_dir=job_dir,
            receptor=receptor_path,
            ligand=ligand_path,
            settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build,
        label="dock",
        input_params=params.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# /api/tasks/dock (FC async task mode)
# ---------------------------------------------------------------------------


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/dock", response_model=JobInfo)
    def post_dock_task(
        request: Request,
        params: DockRequest = Depends(model_form_depends(DockRequest)),
        receptor: Optional[UploadFile] = File(None),
        receptor_uri: Optional[str] = Form(None),
        ligand: Optional[UploadFile] = File(None),
        ligand_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(
            default=None, alias="X-Bioagent-Job-Id"
        ),
        x_fc_async_task_id: Optional[str] = Header(
            default=None, alias="X-Fc-Async-Task-Id"
        ),
    ) -> JobInfo:
        """FC async task mode — blocks until done, HTTP request returns 202."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        # closure-shared dict bridges _save → _build
        paths: dict[str, Path] = {}

        def _save(_req: DockRequest, input_dir: Path) -> None:
            receptor_path, ligand_path = _save_inputs(
                receptor, receptor_uri, ligand, ligand_uri, input_dir,
            )
            paths["receptor"] = receptor_path
            paths["ligand"] = ligand_path

        def _build(req: DockRequest, _job_id: str, job_dir: Path) -> list[str]:
            return dock_argv(
                req,
                job_dir=job_dir,
                receptor=paths["receptor"],
                ligand=paths["ligand"],
                settings=settings,
            )

        return execute_task(
            request,
            job_id=job_id,
            label="dock",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )


# Must come AFTER all POST routes so MCP auto-discovery sees the full surface.
attach_mcp(app)
