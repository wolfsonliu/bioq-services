"""FastAPI app for drughive-server.

Exposes 3 sync endpoints (submit/poll) + 3 async task endpoints:
- /api/generate          → de novo ligand generation (MolGenerator)
- /api/generate_spatial  → scaffold hopping (MolGeneratorSpatial)
- /api/optimize          → multi-cycle QVina2 optimization
- /api/tasks/<same-name> → FC async task mode variants

Job lifecycle endpoints (/healthz, /api/jobs/*, /api/manifest,
/openapi.json) come from ``bioq_service.create_app``.

DrugHIVE is licensed under USC-RL v2.0 (non-commercial academic research
only).  See engineering/decisions/2026-07-02-drughive-server-design.md §1.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import yaml
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

from .adapter import DrughiveAdapter
from .configs import (
    build_generate_config,
    build_generate_spatial_config,
    build_optimize_config,
)
from .models import GenerateRequest, GenerateSpatialRequest, OptimizeRequest
from .settings import DrughiveSettings
from .tools import generate_argv, optimize_argv
from bioq_service.uris import maybe_resolve_input, resolve_input

logger = logging.getLogger(__name__)

settings = DrughiveSettings()
adapter = DrughiveAdapter(settings=settings)

app = create_app(
    adapter, settings,
    title="DrugHIVE Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---------------------------------------------------------------------------
# /healthz/detail override — surface NAS-mounted weight + qvina2 presence.
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
    """Weights + QVina2 probe.  Does not raise on missing pieces so the
    service can boot before NAS is mounted; agents see the status."""

    ckpt = settings.checkpoint_path
    weights_missing: dict[str, str] = {}
    if not ckpt.exists() or ckpt.stat().st_size == 0:
        weights_missing["checkpoint"] = str(ckpt)

    qvina2_path = shutil.which(settings.docking_cmd)

    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "weights_dir": str(settings.weights_dir),
        "weights_loaded": not weights_missing,
        "weights_missing": weights_missing,
        "qvina2_available": qvina2_path is not None,
        "qvina2_path": qvina2_path,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


# ---------------------------------------------------------------------------
# Shared input persistence.
# ---------------------------------------------------------------------------
def _write_config_yml(cfg: dict, path: Path) -> Path:
    """Serialize the YAML config dict for upstream to load via ``Hparams``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _save_target_ligand(
    target: Optional[UploadFile],
    target_uri: Optional[str],
    ligand: Optional[UploadFile],
    ligand_uri: Optional[str],
    input_dir: Path,
) -> tuple[Path, Path]:
    """Common two-input persistence for generate / generate_spatial / optimize."""
    input_dir.mkdir(parents=True, exist_ok=True)

    target_path = resolve_input(
        target, target_uri,
        input_dir / (target.filename if target and target.filename else "target.pdb"),
        settings,
    )
    ligand_path = resolve_input(
        ligand, ligand_uri,
        input_dir / (ligand.filename if ligand and ligand.filename else "ligand.sdf"),
        settings,
    )
    return target_path, ligand_path


def _save_spatial_frag(
    substruct_modify: Optional[UploadFile],
    substruct_modify_uri: Optional[str],
    substruct_modify_pattern: Optional[str],
    input_dir: Path,
) -> Optional[Path]:
    """Persist scaffold-hopping fragment file if provided; validate exactly
    one of {file/URI, SMARTS pattern} is set."""
    have_file = substruct_modify is not None or substruct_modify_uri is not None
    have_pattern = substruct_modify_pattern is not None

    if have_file and have_pattern:
        raise HTTPException(
            status_code=422,
            detail="Provide either substruct_modify file/URI OR "
            "substruct_modify_pattern, not both.",
        )
    if not have_file and not have_pattern:
        raise HTTPException(
            status_code=422,
            detail="Either substruct_modify file/URI or "
            "substruct_modify_pattern is required.",
        )
    if not have_file:
        return None
    dest = input_dir / (
        substruct_modify.filename
        if substruct_modify and substruct_modify.filename
        else "substruct_modify.sdf"
    )
    return maybe_resolve_input(substruct_modify, substruct_modify_uri, dest, settings)


# ===========================================================================
# /api/generate — de novo ligand generation
# ===========================================================================
def _build_generate(
    params: GenerateRequest,
    target: Optional[UploadFile],
    target_uri: Optional[str],
    ligand: Optional[UploadFile],
    ligand_uri: Optional[str],
    job_dir: Path,
) -> list[str]:
    target_path, ligand_path = _save_target_ligand(
        target, target_uri, ligand, ligand_uri, job_dir / "input",
    )
    output_dir = job_dir / "output"
    cfg = build_generate_config(
        req=params,
        target_path=target_path,
        ligand_path=ligand_path,
        output_dir=output_dir,
        settings=settings,
    )
    cfg_path = _write_config_yml(cfg, job_dir / "input" / "config.yml")
    return generate_argv(cfg_path=cfg_path, settings=settings)


@app.post("/api/generate", response_model=JobInfo)
def post_generate(
    params: GenerateRequest = Depends(model_form_depends(GenerateRequest)),
    target: Optional[UploadFile] = File(None),
    target_uri: Optional[str] = Form(None),
    ligand: Optional[UploadFile] = File(None),
    ligand_uri: Optional[str] = Form(None),
) -> JobInfo:
    """De novo ligand generation.  Returns a JobInfo; poll until completed."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        return _build_generate(
            params, target, target_uri, ligand, ligand_uri, job_dir,
        )

    return app.state.runner.submit(
        build_argv=_build, label="generate",
        input_params=params.model_dump(mode="json"),
    )


# ===========================================================================
# /api/generate_spatial — scaffold hopping
# ===========================================================================
def _build_generate_spatial(
    params: GenerateSpatialRequest,
    target: Optional[UploadFile],
    target_uri: Optional[str],
    ligand: Optional[UploadFile],
    ligand_uri: Optional[str],
    substruct_modify: Optional[UploadFile],
    substruct_modify_uri: Optional[str],
    job_dir: Path,
) -> list[str]:
    target_path, ligand_path = _save_target_ligand(
        target, target_uri, ligand, ligand_uri, job_dir / "input",
    )
    frag_path = _save_spatial_frag(
        substruct_modify, substruct_modify_uri,
        params.substruct_modify_pattern, job_dir / "input",
    )
    output_dir = job_dir / "output"
    cfg = build_generate_spatial_config(
        req=params,
        target_path=target_path,
        ligand_path=ligand_path,
        output_dir=output_dir,
        settings=settings,
        substruct_modify_path=frag_path,
    )
    cfg_path = _write_config_yml(cfg, job_dir / "input" / "config.yml")
    return generate_argv(cfg_path=cfg_path, settings=settings)


@app.post("/api/generate_spatial", response_model=JobInfo)
def post_generate_spatial(
    params: GenerateSpatialRequest = Depends(
        model_form_depends(GenerateSpatialRequest)
    ),
    target: Optional[UploadFile] = File(None),
    target_uri: Optional[str] = Form(None),
    ligand: Optional[UploadFile] = File(None),
    ligand_uri: Optional[str] = Form(None),
    substruct_modify: Optional[UploadFile] = File(None),
    substruct_modify_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Substructure modification / scaffold hopping.  Provide EITHER an
    ``substruct_modify`` SDF (upload/URI) OR a ``substruct_modify_pattern``
    SMILES/SMARTS string in params."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        return _build_generate_spatial(
            params, target, target_uri, ligand, ligand_uri,
            substruct_modify, substruct_modify_uri, job_dir,
        )

    return app.state.runner.submit(
        build_argv=_build, label="generate_spatial",
        input_params=params.model_dump(mode="json"),
    )


# ===========================================================================
# /api/optimize — multi-cycle QVina2 optimization
# ===========================================================================
def _build_optimize(
    params: OptimizeRequest,
    target: Optional[UploadFile],
    target_uri: Optional[str],
    ligand: Optional[UploadFile],
    ligand_uri: Optional[str],
    target_pdbqt: Optional[UploadFile],
    target_pdbqt_uri: Optional[str],
    job_dir: Path,
) -> list[str]:
    if params.key_opt == "affinity_qvina" and (
        target_pdbqt is None and target_pdbqt_uri is None
    ):
        raise HTTPException(
            status_code=422,
            detail="target_pdbqt file or URI is required when "
            "key_opt='affinity_qvina'.",
        )
    target_path, ligand_path = _save_target_ligand(
        target, target_uri, ligand, ligand_uri, job_dir / "input",
    )
    pdbqt_dest = job_dir / "input" / (
        target_pdbqt.filename
        if target_pdbqt and target_pdbqt.filename
        else "target.pdbqt"
    )
    pdbqt_path = maybe_resolve_input(
        target_pdbqt, target_pdbqt_uri, pdbqt_dest, settings,
    )
    output_dir = job_dir / "output"
    cfg = build_optimize_config(
        req=params,
        target_path=target_path,
        ligand_path=ligand_path,
        target_pdbqt_path=pdbqt_path,
        output_dir=output_dir,
        settings=settings,
    )
    cfg_path = _write_config_yml(cfg, job_dir / "input" / "config.yml")
    return optimize_argv(cfg_path=cfg_path, settings=settings)


@app.post("/api/optimize", response_model=JobInfo)
def post_optimize(
    params: OptimizeRequest = Depends(model_form_depends(OptimizeRequest)),
    target: Optional[UploadFile] = File(None),
    target_uri: Optional[str] = Form(None),
    ligand: Optional[UploadFile] = File(None),
    ligand_uri: Optional[str] = Form(None),
    target_pdbqt: Optional[UploadFile] = File(None),
    target_pdbqt_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Multi-cycle QVina2 property optimization.  Long-running: default
    params take 4-8 h — prefer /api/tasks/optimize (async task mode)."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        return _build_optimize(
            params, target, target_uri, ligand, ligand_uri,
            target_pdbqt, target_pdbqt_uri, job_dir,
        )

    return app.state.runner.submit(
        build_argv=_build, label="optimize",
        input_params=params.model_dump(mode="json"),
    )


# ===========================================================================
# Async task endpoints — /api/tasks/<same-name>
# ===========================================================================
if settings.task_endpoints_enabled:

    @app.post("/api/tasks/generate", response_model=JobInfo,
              summary="De novo ligand generation (single atomic task).")
    def post_generate_task(
        request: Request,
        params: GenerateRequest = Depends(model_form_depends(GenerateRequest)),
        target: Optional[UploadFile] = File(None),
        target_uri: Optional[str] = Form(None),
        ligand: Optional[UploadFile] = File(None),
        ligand_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(
            default=None, alias="X-Bioagent-Job-Id"
        ),
        x_fc_async_task_id: Optional[str] = Header(
            default=None, alias="X-Fc-Async-Task-Id"
        ),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        cache: dict[str, Path] = {}

        def _save(_req: GenerateRequest, input_dir: Path) -> None:
            t, lg = _save_target_ligand(
                target, target_uri, ligand, ligand_uri, input_dir,
            )
            cache["target"] = t
            cache["ligand"] = lg

        def _build(req: GenerateRequest, _job_id: str, job_dir: Path) -> list[str]:
            cfg = build_generate_config(
                req=req,
                target_path=cache["target"],
                ligand_path=cache["ligand"],
                output_dir=job_dir / "output",
                settings=settings,
            )
            cfg_path = _write_config_yml(cfg, job_dir / "input" / "config.yml")
            return generate_argv(cfg_path=cfg_path, settings=settings)

        return execute_task(
            request, job_id=job_id, label="generate", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/generate_spatial", response_model=JobInfo,
              summary="Substructure modification / scaffold hopping (single atomic task).")
    def post_generate_spatial_task(
        request: Request,
        params: GenerateSpatialRequest = Depends(
            model_form_depends(GenerateSpatialRequest)
        ),
        target: Optional[UploadFile] = File(None),
        target_uri: Optional[str] = Form(None),
        ligand: Optional[UploadFile] = File(None),
        ligand_uri: Optional[str] = Form(None),
        substruct_modify: Optional[UploadFile] = File(None),
        substruct_modify_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(
            default=None, alias="X-Bioagent-Job-Id"
        ),
        x_fc_async_task_id: Optional[str] = Header(
            default=None, alias="X-Fc-Async-Task-Id"
        ),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        cache: dict[str, Optional[Path]] = {}

        def _save(_req: GenerateSpatialRequest, input_dir: Path) -> None:
            t, lg = _save_target_ligand(
                target, target_uri, ligand, ligand_uri, input_dir,
            )
            frag = _save_spatial_frag(
                substruct_modify, substruct_modify_uri,
                _req.substruct_modify_pattern, input_dir,
            )
            cache["target"] = t
            cache["ligand"] = lg
            cache["frag"] = frag

        def _build(
            req: GenerateSpatialRequest, _job_id: str, job_dir: Path,
        ) -> list[str]:
            cfg = build_generate_spatial_config(
                req=req,
                target_path=cache["target"],  # type: ignore[arg-type]
                ligand_path=cache["ligand"],  # type: ignore[arg-type]
                output_dir=job_dir / "output",
                settings=settings,
                substruct_modify_path=cache["frag"],
            )
            cfg_path = _write_config_yml(cfg, job_dir / "input" / "config.yml")
            return generate_argv(cfg_path=cfg_path, settings=settings)

        return execute_task(
            request, job_id=job_id, label="generate_spatial", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/optimize", response_model=JobInfo,
              summary="Multi-cycle QVina2 property optimization (single atomic task; long-running).")
    def post_optimize_task(
        request: Request,
        params: OptimizeRequest = Depends(model_form_depends(OptimizeRequest)),
        target: Optional[UploadFile] = File(None),
        target_uri: Optional[str] = Form(None),
        ligand: Optional[UploadFile] = File(None),
        ligand_uri: Optional[str] = Form(None),
        target_pdbqt: Optional[UploadFile] = File(None),
        target_pdbqt_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(
            default=None, alias="X-Bioagent-Job-Id"
        ),
        x_fc_async_task_id: Optional[str] = Header(
            default=None, alias="X-Fc-Async-Task-Id"
        ),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        cache: dict[str, Optional[Path]] = {}

        def _save(_req: OptimizeRequest, input_dir: Path) -> None:
            if _req.key_opt == "affinity_qvina" and (
                target_pdbqt is None and target_pdbqt_uri is None
            ):
                raise HTTPException(
                    status_code=422,
                    detail="target_pdbqt file or URI required when "
                    "key_opt='affinity_qvina'.",
                )
            t, lg = _save_target_ligand(
                target, target_uri, ligand, ligand_uri, input_dir,
            )
            pdbqt_dest = input_dir / (
                target_pdbqt.filename
                if target_pdbqt and target_pdbqt.filename
                else "target.pdbqt"
            )
            pdbqt_path = maybe_resolve_input(
                target_pdbqt, target_pdbqt_uri, pdbqt_dest, settings,
            )
            cache["target"] = t
            cache["ligand"] = lg
            cache["pdbqt"] = pdbqt_path

        def _build(req: OptimizeRequest, _job_id: str, job_dir: Path) -> list[str]:
            cfg = build_optimize_config(
                req=req,
                target_path=cache["target"],  # type: ignore[arg-type]
                ligand_path=cache["ligand"],  # type: ignore[arg-type]
                target_pdbqt_path=cache["pdbqt"],
                output_dir=job_dir / "output",
                settings=settings,
            )
            cfg_path = _write_config_yml(cfg, job_dir / "input" / "config.yml")
            return optimize_argv(cfg_path=cfg_path, settings=settings)

        return execute_task(
            request, job_id=job_id, label="optimize", params=params,
            build_argv=_build, save_inputs=_save,
        )


# Must come AFTER all POST routes so MCP auto-discovery sees the full surface.
attach_mcp(app)
