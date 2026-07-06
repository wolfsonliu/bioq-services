"""FastAPI app for pocketxmol-server.

Exposes 6 sync endpoints (submit/poll) + 6 async task endpoints:
- /api/dock         → small-molecule / peptide docking
- /api/sbdd         → de novo SBDD
- /api/linking      → fragment linking / growing / PROTAC
- /api/optimize     → molecular optimization
- /api/pepdesign    → peptide design (linear / cyclic / inverse-fold / sc-pack)
- /api/confidence   → post-processing tuned-ranker scoring
- /api/tasks/<same-name>  → FC async task-mode variants

Job lifecycle endpoints (/healthz, /api/jobs/*, /api/manifest,
/openapi.json) come from ``bioagent_service.create_app``.

PocketXMol is MIT-licensed.  See engineering/decisions/
2026-07-06-pocketxmol-server-design.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml
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

from .adapter import PocketXMolAdapter
from .configs import (
    build_dock_config,
    build_linking_config,
    build_model_config,
    build_optimize_config,
    build_pepdesign_config,
    build_sbdd_config,
    confidence_yaml_path,
)
from .models import (
    ConfidenceRequest,
    DockRequest,
    LinkingRequest,
    OptimizeRequest,
    PepDesignMode,
    PepDesignRequest,
    SbddRequest,
)
from .settings import PocketXMolSettings
from .tools import confidence_argv, sample_argv
from .uris import maybe_resolve_input, resolve_input

logger = logging.getLogger(__name__)

settings = PocketXMolSettings()
adapter = PocketXMolAdapter(settings=settings)

app = create_app(
    adapter, settings,
    title="PocketXMol Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---------------------------------------------------------------------------
# /healthz/detail override — surface NAS-mounted weight state.
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
    """Report weight-file presence for the 3 checkpoints (PXM + 2 cfd).

    Does not raise on missing files so the service can boot before NAS is
    mounted; agents see the state via the response.  CCD dir is optional
    (only future sdf2pdb_robust endpoint uses it) so it's not part of the
    hard readiness gate.
    """
    expected = {
        "pxm_checkpoint": settings.pxm_checkpoint,
        "tuned_cfd_ckpt": settings.tuned_cfd_ckpt,
        "flex_cfd_ckpt": settings.flex_cfd_ckpt,
    }
    missing = {
        k: str(p) for k, p in expected.items()
        if not p.exists() or p.stat().st_size == 0
    }
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
# Shared helpers.
# ---------------------------------------------------------------------------
def _write_yaml(cfg: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _dump_task_and_model_yaml(
    task_cfg: dict, job_dir: Path,
) -> tuple[Path, Path]:
    """Write the task YAML + model config override YAML into job_dir/input/."""
    task_path = _write_yaml(task_cfg, job_dir / "input" / "task_config.yml")
    model_cfg = build_model_config(settings)
    model_path = _write_yaml(model_cfg, job_dir / "input" / "model_config.yml")
    return task_path, model_path


def _save_protein(
    protein: Optional[UploadFile],
    protein_uri: Optional[str],
    input_dir: Path,
) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    dest = input_dir / (
        protein.filename if protein and protein.filename else "protein.pdb"
    )
    return resolve_input(protein, protein_uri, dest, settings)


def _save_optional(
    upload: Optional[UploadFile],
    upload_uri: Optional[str],
    dest_name: str,
    input_dir: Path,
) -> Optional[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    dest = input_dir / (
        upload.filename if upload and upload.filename else dest_name
    )
    return maybe_resolve_input(upload, upload_uri, dest, settings)


# ===========================================================================
# 1. /api/dock
# ===========================================================================
def _build_dock(
    params: DockRequest,
    protein: Optional[UploadFile], protein_uri: Optional[str],
    ligand: Optional[UploadFile], ligand_uri: Optional[str],
    ref_ligand: Optional[UploadFile], ref_ligand_uri: Optional[str],
    job_dir: Path,
) -> list[str]:
    # Enforce input mutex: exactly one of ligand/smiles/pep_sequence.
    have_ligand = ligand is not None or ligand_uri is not None
    have_smiles = params.smiles is not None
    have_pepseq = params.pep_sequence is not None
    if sum([have_ligand, have_smiles, have_pepseq]) != 1:
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of: ligand file/URI, smiles, or pep_sequence.",
        )

    protein_path = _save_protein(protein, protein_uri, job_dir / "input")
    ligand_path = _save_optional(ligand, ligand_uri, "ligand.sdf", job_dir / "input")
    ref_path = _save_optional(
        ref_ligand, ref_ligand_uri, "ref_ligand.sdf", job_dir / "input",
    )

    output_dir = job_dir / "output"
    cfg = build_dock_config(
        req=params,
        protein_path=protein_path,
        ligand_path=ligand_path,
        ref_ligand_path=ref_path,
        output_dir=output_dir,
    )
    task_yml, model_yml = _dump_task_and_model_yaml(cfg, job_dir)
    return sample_argv(
        task_config_path=task_yml, model_config_path=model_yml,
        output_dir=output_dir, settings=settings, batch_size=params.batch_size,
    )


@app.post("/api/dock", response_model=JobInfo)
def post_dock(
    params: DockRequest = Depends(model_form_depends(DockRequest)),
    protein: Optional[UploadFile] = File(None),
    protein_uri: Optional[str] = Form(None),
    ligand: Optional[UploadFile] = File(None),
    ligand_uri: Optional[str] = Form(None),
    ref_ligand: Optional[UploadFile] = File(None),
    ref_ligand_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Molecular docking (small-molecule or peptide)."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        return _build_dock(
            params, protein, protein_uri, ligand, ligand_uri,
            ref_ligand, ref_ligand_uri, job_dir,
        )

    return app.state.runner.submit(
        build_argv=_build, label="dock",
        input_params=params.model_dump(mode="json"),
    )


# ===========================================================================
# 2. /api/sbdd
# ===========================================================================
def _build_sbdd(
    params: SbddRequest,
    protein: Optional[UploadFile], protein_uri: Optional[str],
    job_dir: Path,
) -> list[str]:
    protein_path = _save_protein(protein, protein_uri, job_dir / "input")
    output_dir = job_dir / "output"
    cfg = build_sbdd_config(
        req=params, protein_path=protein_path, output_dir=output_dir,
    )
    task_yml, model_yml = _dump_task_and_model_yaml(cfg, job_dir)
    return sample_argv(
        task_config_path=task_yml, model_config_path=model_yml,
        output_dir=output_dir, settings=settings, batch_size=params.batch_size,
    )


@app.post("/api/sbdd", response_model=JobInfo)
def post_sbdd(
    params: SbddRequest = Depends(model_form_depends(SbddRequest)),
    protein: Optional[UploadFile] = File(None),
    protein_uri: Optional[str] = Form(None),
) -> JobInfo:
    """De novo structure-based drug design."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        return _build_sbdd(params, protein, protein_uri, job_dir)

    return app.state.runner.submit(
        build_argv=_build, label="sbdd",
        input_params=params.model_dump(mode="json"),
    )


# ===========================================================================
# 3. /api/linking
# ===========================================================================
def _build_linking(
    params: LinkingRequest,
    protein: Optional[UploadFile], protein_uri: Optional[str],
    input_ligand: Optional[UploadFile], input_ligand_uri: Optional[str],
    job_dir: Path,
) -> list[str]:
    protein_path = _save_protein(protein, protein_uri, job_dir / "input")
    ligand_path = _save_optional(
        input_ligand, input_ligand_uri, "input_ligand.sdf", job_dir / "input",
    )
    if ligand_path is None:
        raise HTTPException(
            status_code=422,
            detail="input_ligand file or URI is required for /api/linking.",
        )
    output_dir = job_dir / "output"
    cfg = build_linking_config(
        req=params, protein_path=protein_path,
        input_ligand_path=ligand_path, output_dir=output_dir,
    )
    task_yml, model_yml = _dump_task_and_model_yaml(cfg, job_dir)
    return sample_argv(
        task_config_path=task_yml, model_config_path=model_yml,
        output_dir=output_dir, settings=settings, batch_size=params.batch_size,
    )


@app.post("/api/linking", response_model=JobInfo)
def post_linking(
    params: LinkingRequest = Depends(model_form_depends(LinkingRequest)),
    protein: Optional[UploadFile] = File(None),
    protein_uri: Optional[str] = Form(None),
    input_ligand: Optional[UploadFile] = File(None),
    input_ligand_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Fragment linking / growing / PROTAC linker design."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        return _build_linking(
            params, protein, protein_uri, input_ligand, input_ligand_uri, job_dir,
        )

    return app.state.runner.submit(
        build_argv=_build, label="linking",
        input_params=params.model_dump(mode="json"),
    )


# ===========================================================================
# 4. /api/optimize
# ===========================================================================
def _build_optimize(
    params: OptimizeRequest,
    protein: Optional[UploadFile], protein_uri: Optional[str],
    input_ligand: Optional[UploadFile], input_ligand_uri: Optional[str],
    job_dir: Path,
) -> list[str]:
    protein_path = _save_protein(protein, protein_uri, job_dir / "input")
    ligand_path = _save_optional(
        input_ligand, input_ligand_uri, "input_ligand.sdf", job_dir / "input",
    )
    if ligand_path is None:
        raise HTTPException(
            status_code=422,
            detail="input_ligand file or URI is required for /api/optimize.",
        )
    output_dir = job_dir / "output"
    cfg = build_optimize_config(
        req=params, protein_path=protein_path,
        input_ligand_path=ligand_path, output_dir=output_dir,
    )
    task_yml, model_yml = _dump_task_and_model_yaml(cfg, job_dir)
    return sample_argv(
        task_config_path=task_yml, model_config_path=model_yml,
        output_dir=output_dir, settings=settings, batch_size=params.batch_size,
    )


@app.post("/api/optimize", response_model=JobInfo)
def post_optimize(
    params: OptimizeRequest = Depends(model_form_depends(OptimizeRequest)),
    protein: Optional[UploadFile] = File(None),
    protein_uri: Optional[str] = Form(None),
    input_ligand: Optional[UploadFile] = File(None),
    input_ligand_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Molecular optimization (local refinement of an input ligand)."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        return _build_optimize(
            params, protein, protein_uri, input_ligand, input_ligand_uri, job_dir,
        )

    return app.state.runner.submit(
        build_argv=_build, label="optimize",
        input_params=params.model_dump(mode="json"),
    )


# ===========================================================================
# 5. /api/pepdesign
# ===========================================================================
def _build_pepdesign(
    params: PepDesignRequest,
    protein: Optional[UploadFile], protein_uri: Optional[str],
    input_peptide: Optional[UploadFile], input_peptide_uri: Optional[str],
    ref_ligand: Optional[UploadFile], ref_ligand_uri: Optional[str],
    job_dir: Path,
) -> list[str]:
    protein_path = _save_protein(protein, protein_uri, job_dir / "input")
    pep_path = _save_optional(
        input_peptide, input_peptide_uri, "input_peptide.pdb", job_dir / "input",
    )
    ref_path = _save_optional(
        ref_ligand, ref_ligand_uri, "ref_ligand.sdf", job_dir / "input",
    )
    if params.mode in (PepDesignMode.inverse_fold, PepDesignMode.sc_pack) and pep_path is None:
        raise HTTPException(
            status_code=422,
            detail=f"input_peptide is required for mode={params.mode.value}.",
        )
    output_dir = job_dir / "output"
    cfg = build_pepdesign_config(
        req=params, protein_path=protein_path,
        input_peptide_path=pep_path, ref_ligand_path=ref_path,
        output_dir=output_dir,
    )
    task_yml, model_yml = _dump_task_and_model_yaml(cfg, job_dir)
    return sample_argv(
        task_config_path=task_yml, model_config_path=model_yml,
        output_dir=output_dir, settings=settings, batch_size=params.batch_size,
    )


@app.post("/api/pepdesign", response_model=JobInfo)
def post_pepdesign(
    params: PepDesignRequest = Depends(model_form_depends(PepDesignRequest)),
    protein: Optional[UploadFile] = File(None),
    protein_uri: Optional[str] = Form(None),
    input_peptide: Optional[UploadFile] = File(None),
    input_peptide_uri: Optional[str] = Form(None),
    ref_ligand: Optional[UploadFile] = File(None),
    ref_ligand_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Peptide design (linear/cyclic de novo, inverse folding, sc-packing)."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        return _build_pepdesign(
            params, protein, protein_uri, input_peptide, input_peptide_uri,
            ref_ligand, ref_ligand_uri, job_dir,
        )

    return app.state.runner.submit(
        build_argv=_build, label="pepdesign",
        input_params=params.model_dump(mode="json"),
    )


# ===========================================================================
# 6. /api/confidence
# ===========================================================================
def _resolve_source_exp_dir(source_job_id: str) -> Path:
    """Locate the single timestamped experiment dir inside a source job's
    output/.  Raises 404 if the source job or its exp dir is missing."""
    src_job_dir = settings.jobs_base_dir / source_job_id
    if not src_job_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Source job not found: {source_job_id}",
        )
    src_output = src_job_dir / "output"
    if not src_output.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Source job has no output/ directory: {source_job_id}",
        )
    exp_dirs = [p for p in src_output.iterdir() if p.is_dir()]
    if len(exp_dirs) != 1:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected exactly 1 experiment sub-directory under "
                f"{src_output}, found {len(exp_dirs)}."
            ),
        )
    return exp_dirs[0]


def _build_confidence(
    params: ConfidenceRequest, job_dir: Path,
) -> list[str]:
    exp_dir = _resolve_source_exp_dir(params.source_job_id)
    # Write a small marker so job_dir/input isn't empty (agent debugging).
    (job_dir / "input").mkdir(parents=True, exist_ok=True)
    (job_dir / "input" / "source_ref.txt").write_text(
        f"source_job_id={params.source_job_id}\n"
        f"variant={params.variant.value}\n"
        f"resolved_exp_dir={exp_dir}\n"
    )
    # Confidence outputs land inside exp_dir/ranking/, not job_dir/output/.
    # We want detect_outputs() to still succeed for this job — symlink the
    # confidence run's ranking dir back into our job's output/ so the
    # framework's zip/download works.
    (job_dir / "output").mkdir(parents=True, exist_ok=True)
    yaml_path = confidence_yaml_path(params.variant, settings)
    return confidence_argv(
        req=params,
        source_output_dir=exp_dir,
        confidence_yaml_path=yaml_path,
        settings=settings,
    )


@app.post("/api/confidence", response_model=JobInfo)
def post_confidence(
    params: ConfidenceRequest = Depends(model_form_depends(ConfidenceRequest)),
) -> JobInfo:
    """Tuned-ranker confidence scoring on a previously completed job."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        return _build_confidence(params, job_dir)

    return app.state.runner.submit(
        build_argv=_build, label="confidence",
        input_params=params.model_dump(mode="json"),
    )


# ===========================================================================
# Async task endpoints — /api/tasks/<same-name>
# ===========================================================================
if settings.task_endpoints_enabled:

    @app.post("/api/tasks/dock", response_model=JobInfo)
    def post_dock_task(
        request: Request,
        params: DockRequest = Depends(model_form_depends(DockRequest)),
        protein: Optional[UploadFile] = File(None),
        protein_uri: Optional[str] = Form(None),
        ligand: Optional[UploadFile] = File(None),
        ligand_uri: Optional[str] = Form(None),
        ref_ligand: Optional[UploadFile] = File(None),
        ref_ligand_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(
            default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(
            default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        def _save(_req: DockRequest, input_dir: Path) -> None:
            pass  # actual work happens in _build; uploads read only once

        def _build(req: DockRequest, _job_id: str, job_dir: Path) -> list[str]:
            return _build_dock(
                req, protein, protein_uri, ligand, ligand_uri,
                ref_ligand, ref_ligand_uri, job_dir,
            )

        return execute_task(
            request, job_id=job_id, label="dock", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/sbdd", response_model=JobInfo)
    def post_sbdd_task(
        request: Request,
        params: SbddRequest = Depends(model_form_depends(SbddRequest)),
        protein: Optional[UploadFile] = File(None),
        protein_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(
            default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(
            default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        def _save(_req: SbddRequest, input_dir: Path) -> None:
            pass

        def _build(req: SbddRequest, _job_id: str, job_dir: Path) -> list[str]:
            return _build_sbdd(req, protein, protein_uri, job_dir)

        return execute_task(
            request, job_id=job_id, label="sbdd", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/linking", response_model=JobInfo)
    def post_linking_task(
        request: Request,
        params: LinkingRequest = Depends(model_form_depends(LinkingRequest)),
        protein: Optional[UploadFile] = File(None),
        protein_uri: Optional[str] = Form(None),
        input_ligand: Optional[UploadFile] = File(None),
        input_ligand_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(
            default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(
            default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        def _save(_req: LinkingRequest, input_dir: Path) -> None:
            pass

        def _build(req: LinkingRequest, _job_id: str, job_dir: Path) -> list[str]:
            return _build_linking(
                req, protein, protein_uri, input_ligand, input_ligand_uri, job_dir,
            )

        return execute_task(
            request, job_id=job_id, label="linking", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/optimize", response_model=JobInfo)
    def post_optimize_task(
        request: Request,
        params: OptimizeRequest = Depends(model_form_depends(OptimizeRequest)),
        protein: Optional[UploadFile] = File(None),
        protein_uri: Optional[str] = Form(None),
        input_ligand: Optional[UploadFile] = File(None),
        input_ligand_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(
            default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(
            default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        def _save(_req: OptimizeRequest, input_dir: Path) -> None:
            pass

        def _build(req: OptimizeRequest, _job_id: str, job_dir: Path) -> list[str]:
            return _build_optimize(
                req, protein, protein_uri, input_ligand, input_ligand_uri, job_dir,
            )

        return execute_task(
            request, job_id=job_id, label="optimize", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/pepdesign", response_model=JobInfo)
    def post_pepdesign_task(
        request: Request,
        params: PepDesignRequest = Depends(model_form_depends(PepDesignRequest)),
        protein: Optional[UploadFile] = File(None),
        protein_uri: Optional[str] = Form(None),
        input_peptide: Optional[UploadFile] = File(None),
        input_peptide_uri: Optional[str] = Form(None),
        ref_ligand: Optional[UploadFile] = File(None),
        ref_ligand_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(
            default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(
            default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        def _save(_req: PepDesignRequest, input_dir: Path) -> None:
            pass

        def _build(req: PepDesignRequest, _job_id: str, job_dir: Path) -> list[str]:
            return _build_pepdesign(
                req, protein, protein_uri, input_peptide, input_peptide_uri,
                ref_ligand, ref_ligand_uri, job_dir,
            )

        return execute_task(
            request, job_id=job_id, label="pepdesign", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/confidence", response_model=JobInfo)
    def post_confidence_task(
        request: Request,
        params: ConfidenceRequest = Depends(model_form_depends(ConfidenceRequest)),
        x_bioagent_job_id: Optional[str] = Header(
            default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(
            default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        def _save(_req: ConfidenceRequest, input_dir: Path) -> None:
            pass

        def _build(req: ConfidenceRequest, _job_id: str, job_dir: Path) -> list[str]:
            return _build_confidence(req, job_dir)

        return execute_task(
            request, job_id=job_id, label="confidence", params=params,
            build_argv=_build, save_inputs=_save,
        )


# Must come AFTER all POST routes so MCP auto-discovery sees the full surface.
attach_mcp(app)
