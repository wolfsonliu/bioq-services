"""FastAPI app for diffdock-server.

Exposes 1 sync submit/poll endpoint + 1 async task endpoint:
- /api/dock            → single protein-ligand docking (DiffDock-L v1.1)
- /api/tasks/dock      → FC async task mode (recommended for >2 min jobs)

Job lifecycle endpoints (/healthz, /api/jobs/*, /api/manifest,
/openapi.json) come from ``bioagent_service.create_app``.

DiffDock upstream is MIT-licensed.
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
from fastapi import Depends, File, Form, Header, HTTPException, Request, UploadFile

from .adapter import DiffdockAdapter
from .models import DockRequest
from .settings import DiffdockSettings
from .tools import dock_argv
from .uris import maybe_resolve_input

logger = logging.getLogger(__name__)

settings = DiffdockSettings()
adapter = DiffdockAdapter(settings=settings)

app = create_app(
    adapter, settings,
    title="DiffDock Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---------------------------------------------------------------------------
# /healthz/detail override — surface NAS weight + SO(3)/torus LUT presence.
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
    """Weights + LUT cache probe.  Does not raise on missing pieces so the
    service can boot before NAS is mounted; agents see the status."""

    weights_missing: dict[str, str] = {}
    # Score model
    score_ckpt = settings.score_model_dir / "best_ema_inference_epoch_model.pt"
    score_yml = settings.score_model_dir / "model_parameters.yml"
    if not score_ckpt.exists() or score_ckpt.stat().st_size == 0:
        weights_missing["score_model_ckpt"] = str(score_ckpt)
    if not score_yml.exists():
        weights_missing["score_model_yml"] = str(score_yml)
    # Confidence model
    conf_ckpt = settings.confidence_model_dir / "best_model_epoch75.pt"
    conf_yml = settings.confidence_model_dir / "model_parameters.yml"
    if not conf_ckpt.exists() or conf_ckpt.stat().st_size == 0:
        weights_missing["confidence_model_ckpt"] = str(conf_ckpt)
    if not conf_yml.exists():
        weights_missing["confidence_model_yml"] = str(conf_yml)
    # ESM-2 embed weights (hard-required — every dock uses ESM embed)
    esm_ckpt = (
        settings.esm_cache_dir / "hub" / "checkpoints" / "esm2_t33_650M_UR50D.pt"
    )
    if not esm_ckpt.exists() or esm_ckpt.stat().st_size == 0:
        weights_missing["esm2_650M"] = str(esm_ckpt)

    # ESMFold weights are only needed on protein_sequence input;
    # report as a soft warning, not a hard-missing.
    esmfold_ckpt = settings.esm_cache_dir / "hub" / "checkpoints" / "esmfold_3B_v1.pt"
    esmfold_available = esmfold_ckpt.exists() and esmfold_ckpt.stat().st_size > 0

    # SO(3) / torus LUT — pre-computed at Docker build time, live under
    # /opt/diffdock/.  If missing, first request will hang ~2 min while
    # utils/precompute_series.py regenerates them (or fail on FC where
    # write to / may be forbidden).
    so3_cache = settings.root / ".so3_omegas_array4.npy"
    torus_cache = settings.root / ".torus_score_norms.npy"

    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "weights_dir": str(settings.weights_dir),
        "weights_loaded": not weights_missing,
        "weights_missing": weights_missing,
        "esmfold_available": esmfold_available,
        "so3_cache_ok": so3_cache.exists(),
        "torus_cache_ok": torus_cache.exists(),
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


# ---------------------------------------------------------------------------
# Shared input persistence.
# ---------------------------------------------------------------------------
def _resolve_inputs(
    params: DockRequest,
    protein: Optional[UploadFile],
    ligand: Optional[UploadFile],
    input_dir: Path,
) -> tuple[Optional[Path], Optional[str], str]:
    """Resolve protein + ligand to CLI-ready args.

    Returns ``(protein_path, protein_sequence, ligand_arg)`` where exactly
    one of ``protein_path``/``protein_sequence`` is non-None, and
    ``ligand_arg`` is either an absolute file path (str) or a SMILES.
    """
    input_dir.mkdir(parents=True, exist_ok=True)

    # Protein: exactly one of protein / protein_uri / protein_sequence.
    protein_flags = [
        protein is not None,
        params.protein_uri is not None,
        params.protein_sequence is not None,
    ]
    if sum(protein_flags) != 1:
        raise HTTPException(
            status_code=422,
            detail="Exactly one of protein (multipart), protein_uri, or "
            "protein_sequence must be provided.",
        )

    protein_path: Optional[Path] = None
    protein_seq: Optional[str] = None
    if params.protein_sequence is not None:
        protein_seq = params.protein_sequence
    else:
        dest = input_dir / (
            protein.filename if protein and protein.filename else "protein.pdb"
        )
        protein_path = maybe_resolve_input(
            protein, params.protein_uri, dest, settings,
        )

    # Ligand: exactly one of ligand / ligand_uri / ligand_description.
    ligand_flags = [
        ligand is not None,
        params.ligand_uri is not None,
        params.ligand_description is not None,
    ]
    if sum(ligand_flags) != 1:
        raise HTTPException(
            status_code=422,
            detail="Exactly one of ligand (multipart), ligand_uri, or "
            "ligand_description must be provided.",
        )

    if params.ligand_description is not None:
        ligand_arg = params.ligand_description
    else:
        dest = input_dir / (
            ligand.filename if ligand and ligand.filename else "ligand.sdf"
        )
        ligand_path = maybe_resolve_input(
            ligand, params.ligand_uri, dest, settings,
        )
        ligand_arg = str(ligand_path)

    return protein_path, protein_seq, ligand_arg


def _build_dock_argv(
    params: DockRequest,
    protein: Optional[UploadFile],
    ligand: Optional[UploadFile],
    job_dir: Path,
) -> list[str]:
    protein_path, protein_seq, ligand_arg = _resolve_inputs(
        params, protein, ligand, job_dir / "input",
    )
    return dock_argv(
        protein_path=protein_path,
        protein_sequence=protein_seq,
        ligand_arg=ligand_arg,
        out_dir=job_dir / "output",
        params=params,
        settings=settings,
    )


# ===========================================================================
# /api/dock — single-complex docking (submit/poll)
# ===========================================================================
@app.post("/api/dock", response_model=JobInfo)
def post_dock(
    params: DockRequest = Depends(model_form_depends(DockRequest)),
    protein: Optional[UploadFile] = File(None),
    ligand: Optional[UploadFile] = File(None),
) -> JobInfo:
    """Submit a single protein-ligand docking job.  Poll /api/jobs/<id>."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        return _build_dock_argv(params, protein, ligand, job_dir)

    return app.state.runner.submit(
        build_argv=_build, label="dock",
        input_params=params.model_dump(mode="json"),
    )


# ===========================================================================
# /api/tasks/dock — async task mode
# ===========================================================================
if settings.task_endpoints_enabled:

    @app.post("/api/tasks/dock", response_model=JobInfo)
    def post_dock_task(
        request: Request,
        params: DockRequest = Depends(model_form_depends(DockRequest)),
        protein: Optional[UploadFile] = File(None),
        ligand: Optional[UploadFile] = File(None),
        x_bioagent_job_id: Optional[str] = Header(
            default=None, alias="X-Bioagent-Job-Id",
        ),
        x_fc_async_task_id: Optional[str] = Header(
            default=None, alias="X-Fc-Async-Task-Id",
        ),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        cache: dict[str, object] = {}

        def _save(_req: DockRequest, input_dir: Path) -> None:
            protein_path, protein_seq, ligand_arg = _resolve_inputs(
                _req, protein, ligand, input_dir,
            )
            cache["protein_path"] = protein_path
            cache["protein_sequence"] = protein_seq
            cache["ligand_arg"] = ligand_arg

        def _build(req: DockRequest, _job_id: str, job_dir: Path) -> list[str]:
            return dock_argv(
                protein_path=cache["protein_path"],  # type: ignore[arg-type]
                protein_sequence=cache["protein_sequence"],  # type: ignore[arg-type]
                ligand_arg=cache["ligand_arg"],  # type: ignore[arg-type]
                out_dir=job_dir / "output",
                params=req,
                settings=settings,
            )

        return execute_task(
            request, job_id=job_id, label="dock", params=params,
            build_argv=_build, save_inputs=_save,
        )


# Must come AFTER all POST routes so MCP auto-discovery sees the full surface.
attach_mcp(app)
