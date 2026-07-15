"""FastAPI app for haddock3-server.

Endpoints (each with a matching /api/tasks/<name> for FC async task mode):

  * `/api/dock`                              — general workflow runner
  * `/api/dock/protein-protein`             — curated two-body docking
  * `/api/score`                            — standalone HADDOCK scoring
  * `/api/restraints/restrain-bodies`       — CNS-free body restraints
  * `/api/restraints/active-passive-to-ambig` — CNS-free ambig restraints

Job lifecycle (/healthz, /api/jobs/*, /api/manifest, /openapi.json) comes from
`bioagent_service.create_app`.
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
    maybe_resolve_input,
    model_form_depends,
    read_version_file,
    resolve_input,
    resolve_task_id,
)
from bioagent_service.uris import resolve_uri
from fastapi import Depends, File, Form, Header, Request, UploadFile

from .adapter import Haddock3Adapter
from .configs import build_protein_protein_cfg, finalize_general_cfg, write_cfg
from .models import (
    ActpassToAmbigRequest,
    DockRequest,
    ProteinProteinRequest,
    RestrainBodiesRequest,
    ScoreRequest,
)
from .settings import Haddock3Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = Haddock3Settings()
adapter = Haddock3Adapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="HADDOCK3 Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---------------------------------------------------------------------------
# /healthz/detail override — report CNS availability (the real readiness signal
# for docking/scoring). Restraints endpoints work regardless.
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


def _haddock3_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("haddock3")
    except Exception:  # noqa: BLE001 — probe must never crash the endpoint
        return None


@app.get("/healthz/detail")
def healthz_detail(request: Request) -> dict:
    cns = settings.cns_exec
    cns_available = cns.exists() and cns.is_file()
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "haddock3_version": _haddock3_version(),
        "cns_exec": str(cns),
        "cns_available": cns_available,
        # Retained for cross-service uniformity — HADDOCK3 has no NN weights;
        # CNS presence is the readiness signal for docking/scoring.
        "weights_dir": str(settings.weights_dir),
        "weights_loaded": cns_available,
        "restraints_available": True,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
        "task_endpoints_enabled": settings.task_endpoints_enabled,
    }


# ---------------------------------------------------------------------------
# Input staging helpers
# ---------------------------------------------------------------------------


def _safe_name(name: Optional[str], default: str) -> str:
    if not name:
        return default
    return Path(name).name or default


def _stage_molecules(
    input_dir: Path,
    molecules: Optional[list[UploadFile]],
    molecules_uri: Optional[list[str]],
) -> list[Path]:
    """Stage 1+ molecule PDBs (uploads and/or URIs), preserving filenames."""
    input_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for i, up in enumerate(molecules or []):
        dest = input_dir / _safe_name(up.filename, f"mol_{i + 1}.pdb")
        staged.append(resolve_input(up, None, dest, settings, f"molecules[{i}]"))
    for i, uri in enumerate(molecules_uri or []):
        dest = input_dir / f"mol_uri_{i + 1}.pdb"
        staged.append(resolve_uri(uri, dest, settings))
    if not staged:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail="no molecules provided")
    return staged


def _stage_tbls(
    input_dir: Path,
    tbls: Optional[list[UploadFile]],
    tbl_uri: Optional[list[str]],
) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for i, up in enumerate(tbls or []):
        dest = input_dir / _safe_name(up.filename, f"restraint_{i + 1}.tbl")
        staged.append(resolve_input(up, None, dest, settings, f"tbl[{i}]"))
    for i, uri in enumerate(tbl_uri or []):
        dest = input_dir / f"restraint_uri_{i + 1}.tbl"
        staged.append(resolve_uri(uri, dest, settings))
    return staged


# ---------------------------------------------------------------------------
# Build helpers (shared by submit + task paths)
# ---------------------------------------------------------------------------


def _build_general_dock(
    req: DockRequest, config_text: str, input_dir: Path, job_dir: Path,
    molecules: list[Path], _tbls: list[Path],
) -> list[str]:
    from .tools import dock_argv

    run_dir = job_dir / "output" / req.run_name
    cfg_text = finalize_general_cfg(
        config_text,
        molecules=[m.name for m in molecules],
        run_dir=str(run_dir),
        ncores=req.ncores or settings.default_ncores,
    )
    cfg_path = write_cfg(cfg_text, input_dir / "workflow.cfg")
    return dock_argv(
        req, config_path=cfg_path, workdir=input_dir, job_dir=job_dir, settings=settings,
    )


def _build_protein_protein(
    req: ProteinProteinRequest, input_dir: Path, job_dir: Path,
    mol1: Path, mol2: Path, ambig: Optional[Path], reference: Optional[Path],
) -> list[str]:
    from .tools import dock_argv

    run_dir = job_dir / "output" / "run"
    cfg_text = build_protein_protein_cfg(
        molecules=[str(mol1), str(mol2)],
        run_dir=str(run_dir),
        ncores=req.ncores or settings.default_ncores,
        sampling=req.sampling,
        do_flexref=req.do_flexref,
        do_emref=req.do_emref,
        clustering=req.clustering,
        top_models=req.top_models,
        ambig_fname=str(ambig) if ambig else None,
        reference_fname=str(reference) if reference else None,
    )
    cfg_path = write_cfg(cfg_text, input_dir / "workflow.cfg")
    return dock_argv(
        req, config_path=cfg_path, workdir=input_dir, job_dir=job_dir, settings=settings,
    )


# ---------------------------------------------------------------------------
# Submit/poll endpoints
# ---------------------------------------------------------------------------


@app.post("/api/dock", response_model=JobInfo)
def post_dock(
    config: str = Form(..., description="Workflow body ([module] sections; no "
                       "run_dir/molecules/mode/ncores)."),
    params: DockRequest = Depends(model_form_depends(DockRequest)),
    molecules: Optional[list[UploadFile]] = File(None),
    molecules_uri: Optional[list[str]] = Form(None),
    tbl: Optional[list[UploadFile]] = File(None),
    tbl_uri: Optional[list[str]] = Form(None),
) -> JobInfo:
    """Run an arbitrary haddock3 workflow. Needs CNS for docking modules."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        input_dir = job_dir / "input"
        mols = _stage_molecules(input_dir, molecules, molecules_uri)
        tbls = _stage_tbls(input_dir, tbl, tbl_uri)
        return _build_general_dock(params, config, input_dir, job_dir, mols, tbls)

    return app.state.runner.submit(
        build_argv=_build, label="dock",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/dock/protein-protein", response_model=JobInfo)
def post_dock_protein_protein(
    params: ProteinProteinRequest = Depends(model_form_depends(ProteinProteinRequest)),
    mol1: Optional[UploadFile] = File(None),
    mol1_uri: Optional[str] = Form(None),
    mol2: Optional[UploadFile] = File(None),
    mol2_uri: Optional[str] = Form(None),
    ambig: Optional[UploadFile] = File(None),
    ambig_uri: Optional[str] = Form(None),
    reference: Optional[UploadFile] = File(None),
    reference_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Curated two-body protein-protein docking. Needs CNS."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        m1 = resolve_input(mol1, mol1_uri, input_dir / "mol_1.pdb", settings, "mol1")
        m2 = resolve_input(mol2, mol2_uri, input_dir / "mol_2.pdb", settings, "mol2")
        amb = maybe_resolve_input(ambig, ambig_uri, input_dir / "ambig.tbl", settings, "ambig")
        ref = maybe_resolve_input(
            reference, reference_uri, input_dir / "reference.pdb", settings, "reference",
        )
        return _build_protein_protein(params, input_dir, job_dir, m1, m2, amb, ref)

    return app.state.runner.submit(
        build_argv=_build, label="protein-protein",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/score", response_model=JobInfo)
def post_score(
    params: ScoreRequest = Depends(model_form_depends(ScoreRequest)),
    complex: Optional[UploadFile] = File(None),
    complex_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Standalone HADDOCK scoring of a complex. Needs CNS."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        from .tools import score_argv

        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        pdb = resolve_input(complex, complex_uri, input_dir / "complex.pdb", settings, "complex")
        return score_argv(params, pdb=pdb, job_dir=job_dir, settings=settings)

    return app.state.runner.submit(
        build_argv=_build, label="score",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/restraints/restrain-bodies", response_model=JobInfo)
def post_restrain_bodies(
    params: RestrainBodiesRequest = Depends(model_form_depends(RestrainBodiesRequest)),
    structure: Optional[UploadFile] = File(None),
    structure_uri: Optional[str] = Form(None),
) -> JobInfo:
    """CNS-free: distance restraints locking chains as rigid bodies."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        from .tools import restrain_bodies_argv

        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        pdb = resolve_input(
            structure, structure_uri, input_dir / "structure.pdb", settings, "structure",
        )
        return restrain_bodies_argv(params, pdb=pdb, job_dir=job_dir, settings=settings)

    return app.state.runner.submit(
        build_argv=_build, label="restrain-bodies",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/restraints/active-passive-to-ambig", response_model=JobInfo)
def post_actpass_to_ambig(
    params: ActpassToAmbigRequest = Depends(model_form_depends(ActpassToAmbigRequest)),
    actpass1: Optional[UploadFile] = File(None),
    actpass1_uri: Optional[str] = Form(None),
    actpass2: Optional[UploadFile] = File(None),
    actpass2_uri: Optional[str] = Form(None),
) -> JobInfo:
    """CNS-free: ambiguous restraints from two active/passive residue files."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        from .tools import actpass_to_ambig_argv

        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        a1 = resolve_input(actpass1, actpass1_uri, input_dir / "a.actpass", settings, "actpass1")
        a2 = resolve_input(actpass2, actpass2_uri, input_dir / "b.actpass", settings, "actpass2")
        return actpass_to_ambig_argv(
            params, actpass1=a1, actpass2=a2, job_dir=job_dir, settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build, label="actpass-to-ambig",
        input_params=params.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# Task endpoints (FC async task mode)
# ---------------------------------------------------------------------------


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/dock", response_model=JobInfo)
    def post_dock_task(
        request: Request,
        config: str = Form(...),
        params: DockRequest = Depends(model_form_depends(DockRequest)),
        molecules: Optional[list[UploadFile]] = File(None),
        molecules_uri: Optional[list[str]] = Form(None),
        tbl: Optional[list[UploadFile]] = File(None),
        tbl_uri: Optional[list[str]] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        state: dict[str, list[Path]] = {}

        def _save(_req: DockRequest, input_dir: Path) -> None:
            state["molecules"] = _stage_molecules(input_dir, molecules, molecules_uri)
            state["tbls"] = _stage_tbls(input_dir, tbl, tbl_uri)

        def _build(req: DockRequest, _job_id: str, job_dir: Path) -> list[str]:
            return _build_general_dock(
                req, config, job_dir / "input", job_dir, state["molecules"], state["tbls"],
            )

        return execute_task(
            request, job_id=job_id, label="dock", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/dock/protein-protein", response_model=JobInfo)
    def post_dock_protein_protein_task(
        request: Request,
        params: ProteinProteinRequest = Depends(model_form_depends(ProteinProteinRequest)),
        mol1: Optional[UploadFile] = File(None),
        mol1_uri: Optional[str] = Form(None),
        mol2: Optional[UploadFile] = File(None),
        mol2_uri: Optional[str] = Form(None),
        ambig: Optional[UploadFile] = File(None),
        ambig_uri: Optional[str] = Form(None),
        reference: Optional[UploadFile] = File(None),
        reference_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        state: dict[str, Optional[Path]] = {}

        def _save(_req: ProteinProteinRequest, input_dir: Path) -> None:
            state["mol1"] = resolve_input(mol1, mol1_uri, input_dir / "mol_1.pdb", settings, "mol1")
            state["mol2"] = resolve_input(mol2, mol2_uri, input_dir / "mol_2.pdb", settings, "mol2")
            state["ambig"] = maybe_resolve_input(
                ambig, ambig_uri, input_dir / "ambig.tbl", settings, "ambig",
            )
            state["reference"] = maybe_resolve_input(
                reference, reference_uri, input_dir / "reference.pdb", settings, "reference",
            )

        def _build(req: ProteinProteinRequest, _job_id: str, job_dir: Path) -> list[str]:
            return _build_protein_protein(
                req, job_dir / "input", job_dir,
                state["mol1"], state["mol2"], state["ambig"], state["reference"],
            )

        return execute_task(
            request, job_id=job_id, label="protein-protein", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/score", response_model=JobInfo)
    def post_score_task(
        request: Request,
        params: ScoreRequest = Depends(model_form_depends(ScoreRequest)),
        complex: Optional[UploadFile] = File(None),
        complex_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        state: dict[str, Path] = {}

        def _save(_req: ScoreRequest, input_dir: Path) -> None:
            state["pdb"] = resolve_input(
                complex, complex_uri, input_dir / "complex.pdb", settings, "complex",
            )

        def _build(req: ScoreRequest, _job_id: str, job_dir: Path) -> list[str]:
            from .tools import score_argv

            return score_argv(req, pdb=state["pdb"], job_dir=job_dir, settings=settings)

        return execute_task(
            request, job_id=job_id, label="score", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/restraints/restrain-bodies", response_model=JobInfo)
    def post_restrain_bodies_task(
        request: Request,
        params: RestrainBodiesRequest = Depends(model_form_depends(RestrainBodiesRequest)),
        structure: Optional[UploadFile] = File(None),
        structure_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        state: dict[str, Path] = {}

        def _save(_req: RestrainBodiesRequest, input_dir: Path) -> None:
            state["pdb"] = resolve_input(
                structure, structure_uri, input_dir / "structure.pdb", settings, "structure",
            )

        def _build(req: RestrainBodiesRequest, _job_id: str, job_dir: Path) -> list[str]:
            from .tools import restrain_bodies_argv

            return restrain_bodies_argv(req, pdb=state["pdb"], job_dir=job_dir, settings=settings)

        return execute_task(
            request, job_id=job_id, label="restrain-bodies", params=params,
            build_argv=_build, save_inputs=_save,
        )

    @app.post("/api/tasks/restraints/active-passive-to-ambig", response_model=JobInfo)
    def post_actpass_to_ambig_task(
        request: Request,
        params: ActpassToAmbigRequest = Depends(model_form_depends(ActpassToAmbigRequest)),
        actpass1: Optional[UploadFile] = File(None),
        actpass1_uri: Optional[str] = Form(None),
        actpass2: Optional[UploadFile] = File(None),
        actpass2_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        state: dict[str, Path] = {}

        def _save(_req: ActpassToAmbigRequest, input_dir: Path) -> None:
            state["a1"] = resolve_input(
                actpass1, actpass1_uri, input_dir / "a.actpass", settings, "actpass1",
            )
            state["a2"] = resolve_input(
                actpass2, actpass2_uri, input_dir / "b.actpass", settings, "actpass2",
            )

        def _build(req: ActpassToAmbigRequest, _job_id: str, job_dir: Path) -> list[str]:
            from .tools import actpass_to_ambig_argv

            return actpass_to_ambig_argv(
                req, actpass1=state["a1"], actpass2=state["a2"],
                job_dir=job_dir, settings=settings,
            )

        return execute_task(
            request, job_id=job_id, label="actpass-to-ambig", params=params,
            build_argv=_build, save_inputs=_save,
        )


# Must come AFTER all POST routes so MCP auto-discovery sees the full surface.
attach_mcp(app)
