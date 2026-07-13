"""FastAPI app for reinvent-server.

5 structured endpoints (one per REINVENT run mode) + FC async task variants
(task_endpoints_enabled=True). See engineering/decisions/2026-07-08-reinvent-server-design.md.
"""
from __future__ import annotations

import shutil
import subprocess
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
from fastapi import Depends, File, Header, Request, UploadFile

from .adapter import ReinventAdapter
from .config_builder import PRIOR_FILES
from .models import (
    EnumerationRequest, SamplingRequest, ScoringRequest,
    StagedLearningRequest, TransferLearningRequest,
)
from .settings import ReinventSettings
from .tools import (
    enumeration_argv, sampling_argv, scoring_argv,
    staged_learning_argv, transfer_learning_argv,
)

settings = ReinventSettings()
adapter = ReinventAdapter(settings=settings)

app = create_app(
    adapter, settings,
    title="Reinvent Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---- /healthz/detail override: report GPU + prior mount ----

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


def _probe_gpus() -> list[str]:
    """Names of attached NVIDIA GPUs via nvidia-smi.

    Uses nvidia-smi (fast, injected by the FC GPU runtime) instead of a cold
    `import torch` subprocess: the torch probe took 10-15s and its 30s timeout
    fired on the health path, so cuda_available read False even though jobs got
    the GPU (reinvent.log showed "Using GPU device:0 Tesla T4"). Cached per
    process — GPU topology is fixed for an instance's lifetime.
    """
    global _GPU_CACHE
    if _GPU_CACHE is not None:
        return _GPU_CACHE
    gpus: list[str] = []
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            gpus = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    except Exception:
        gpus = []
    _GPU_CACHE = gpus
    return gpus


_GPU_CACHE: list[str] | None = None


@app.get("/healthz/detail")
def healthz_detail(request: Request) -> dict:
    priors = {k: settings.prior_base / v for k, v in PRIOR_FILES.items()}
    missing = {k: str(p) for k, p in priors.items() if not p.exists()}
    gpus = _probe_gpus()
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "prior_base": str(settings.prior_base),
        "priors_loaded": not missing,
        "priors_missing": missing,
        "cuda_available": bool(gpus),
        "gpus": gpus,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
        "task_endpoints_enabled": settings.task_endpoints_enabled,
    }


# ---- helpers ----

def _persist(upload: UploadFile, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dst


def _files_sampling(job_dir, smiles_file):
    d = {}
    if smiles_file is not None:
        d["smiles_file"] = _persist(smiles_file, job_dir / "inputs" / smiles_file.filename)
    return d


# ---- sync endpoints ----

@app.post("/api/sampling", response_model=JobInfo)
def sampling(
    params: SamplingRequest = Depends(model_form_depends(SamplingRequest)),
    smiles_file: Optional[UploadFile] = File(None),
) -> JobInfo:
    def _build(job_id, job_dir):
        return sampling_argv(params, _files_sampling(job_dir, smiles_file), job_dir, settings)
    return app.state.runner.submit(build_argv=_build, label="sampling",
                                   input_params=params.model_dump(mode="json"))


@app.post("/api/scoring", response_model=JobInfo)
def scoring(
    params: ScoringRequest = Depends(model_form_depends(ScoringRequest)),
    smiles_file: UploadFile = File(...),
) -> JobInfo:
    def _build(job_id, job_dir):
        f = {"smiles_file": _persist(smiles_file, job_dir / "inputs" / smiles_file.filename)}
        return scoring_argv(params, f, job_dir, settings)
    return app.state.runner.submit(build_argv=_build, label="scoring",
                                   input_params=params.model_dump(mode="json"))


@app.post("/api/enumeration", response_model=JobInfo)
def enumeration(
    params: EnumerationRequest = Depends(model_form_depends(EnumerationRequest)),
    peptide_smiles: UploadFile = File(...),
    amino_acid_library: UploadFile = File(...),
) -> JobInfo:
    def _build(job_id, job_dir):
        f = {
            "smiles_file": _persist(peptide_smiles, job_dir / "inputs" / peptide_smiles.filename),
            "amino_acid_library": _persist(amino_acid_library, job_dir / "inputs" / amino_acid_library.filename),
        }
        return enumeration_argv(params, f, job_dir, settings)
    return app.state.runner.submit(build_argv=_build, label="enumeration",
                                   input_params=params.model_dump(mode="json"))


@app.post("/api/transfer-learning", response_model=JobInfo)
def transfer_learning(
    params: TransferLearningRequest = Depends(model_form_depends(TransferLearningRequest)),
    smiles_file: UploadFile = File(...),
    validation_smiles_file: Optional[UploadFile] = File(None),
    input_model_upload: Optional[UploadFile] = File(None),
) -> JobInfo:
    def _build(job_id, job_dir):
        f = {"smiles_file": _persist(smiles_file, job_dir / "inputs" / smiles_file.filename)}
        if validation_smiles_file is not None:
            f["validation_smiles_file"] = _persist(
                validation_smiles_file, job_dir / "inputs" / validation_smiles_file.filename)
        if input_model_upload is not None:
            f["model_file"] = _persist(
                input_model_upload, job_dir / "inputs" / input_model_upload.filename)
        return transfer_learning_argv(params, f, job_dir, settings)
    return app.state.runner.submit(build_argv=_build, label="transfer_learning",
                                   input_params=params.model_dump(mode="json"))


@app.post("/api/staged-learning", response_model=JobInfo)
def staged_learning(
    params: StagedLearningRequest = Depends(model_form_depends(StagedLearningRequest)),
    smiles_file: Optional[UploadFile] = File(None),
    prior_upload: Optional[UploadFile] = File(None),
    agent_upload: Optional[UploadFile] = File(None),
) -> JobInfo:
    def _build(job_id, job_dir):
        f = {}
        if smiles_file is not None:
            f["smiles_file"] = _persist(smiles_file, job_dir / "inputs" / smiles_file.filename)
        if prior_upload is not None:
            f["prior_file"] = _persist(prior_upload, job_dir / "inputs" / prior_upload.filename)
        if agent_upload is not None:
            f["agent_file"] = _persist(agent_upload, job_dir / "inputs" / agent_upload.filename)
        return staged_learning_argv(params, f, job_dir, settings)
    return app.state.runner.submit(build_argv=_build, label="staged_learning",
                                   input_params=params.model_dump(mode="json"))


# ---- FC async task endpoints (task_endpoints_enabled=True) ----

if settings.task_endpoints_enabled:

    @app.post("/api/tasks/sampling", response_model=JobInfo)
    def task_sampling(
        request: Request,
        params: SamplingRequest = Depends(model_form_depends(SamplingRequest)),
        smiles_file: Optional[UploadFile] = File(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias=settings.task_job_id_header),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        def _build(req, _job_id, job_dir):
            return sampling_argv(req, _files_sampling(job_dir, smiles_file), job_dir, settings)
        return execute_task(request, job_id=job_id, label="sampling",
                            params=params, build_argv=_build)

    @app.post("/api/tasks/scoring", response_model=JobInfo)
    def task_scoring(
        request: Request,
        params: ScoringRequest = Depends(model_form_depends(ScoringRequest)),
        smiles_file: UploadFile = File(...),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias=settings.task_job_id_header),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        def _build(req, _job_id, job_dir):
            f = {"smiles_file": _persist(smiles_file, job_dir / "inputs" / smiles_file.filename)}
            return scoring_argv(req, f, job_dir, settings)
        return execute_task(request, job_id=job_id, label="scoring",
                            params=params, build_argv=_build)

    @app.post("/api/tasks/enumeration", response_model=JobInfo)
    def task_enumeration(
        request: Request,
        params: EnumerationRequest = Depends(model_form_depends(EnumerationRequest)),
        peptide_smiles: UploadFile = File(...),
        amino_acid_library: UploadFile = File(...),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias=settings.task_job_id_header),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        def _build(req, _job_id, job_dir):
            f = {
                "smiles_file": _persist(peptide_smiles, job_dir / "inputs" / peptide_smiles.filename),
                "amino_acid_library": _persist(amino_acid_library, job_dir / "inputs" / amino_acid_library.filename),
            }
            return enumeration_argv(req, f, job_dir, settings)
        return execute_task(request, job_id=job_id, label="enumeration",
                            params=params, build_argv=_build)

    @app.post("/api/tasks/transfer-learning", response_model=JobInfo)
    def task_transfer_learning(
        request: Request,
        params: TransferLearningRequest = Depends(model_form_depends(TransferLearningRequest)),
        smiles_file: UploadFile = File(...),
        validation_smiles_file: Optional[UploadFile] = File(None),
        input_model_upload: Optional[UploadFile] = File(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias=settings.task_job_id_header),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        def _build(req, _job_id, job_dir):
            f = {"smiles_file": _persist(smiles_file, job_dir / "inputs" / smiles_file.filename)}
            if validation_smiles_file is not None:
                f["validation_smiles_file"] = _persist(
                    validation_smiles_file, job_dir / "inputs" / validation_smiles_file.filename)
            if input_model_upload is not None:
                f["model_file"] = _persist(
                    input_model_upload, job_dir / "inputs" / input_model_upload.filename)
            return transfer_learning_argv(req, f, job_dir, settings)
        return execute_task(request, job_id=job_id, label="transfer_learning",
                            params=params, build_argv=_build)

    @app.post("/api/tasks/staged-learning", response_model=JobInfo)
    def task_staged_learning(
        request: Request,
        params: StagedLearningRequest = Depends(model_form_depends(StagedLearningRequest)),
        smiles_file: Optional[UploadFile] = File(None),
        prior_upload: Optional[UploadFile] = File(None),
        agent_upload: Optional[UploadFile] = File(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias=settings.task_job_id_header),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        def _build(req, _job_id, job_dir):
            f = {}
            if smiles_file is not None:
                f["smiles_file"] = _persist(smiles_file, job_dir / "inputs" / smiles_file.filename)
            if prior_upload is not None:
                f["prior_file"] = _persist(prior_upload, job_dir / "inputs" / prior_upload.filename)
            if agent_upload is not None:
                f["agent_file"] = _persist(agent_upload, job_dir / "inputs" / agent_upload.filename)
            return staged_learning_argv(req, f, job_dir, settings)
        return execute_task(request, job_id=job_id, label="staged_learning",
                            params=params, build_argv=_build)


attach_mcp(app)
