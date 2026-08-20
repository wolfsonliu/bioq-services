"""FastAPI app for reinvent-server.

5 structured endpoints (one per REINVENT run mode) + FC async task variants
(task_endpoints_enabled=True). See engineering/decisions/2026-07-08-reinvent-server-design.md.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from bioq_service import (
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
from bioq_service.uris import maybe_resolve_input, resolve_input

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
#
# Every file input accepts EITHER a multipart upload OR a `*_uri` (job/oss/file/
# http scheme) resolved via uris.py — same pattern as boltz / diffdock / drughive.
# `_dest` keeps the on-disk filename stable (upload name → URI basename → fallback)
# so reinvent_cli's flag values point at a sensibly-named file.

def _dest(job_dir: Path, upload: Optional[UploadFile], uri: Optional[str], fallback: str) -> Path:
    name = None
    if upload is not None and upload.filename:
        name = Path(upload.filename).name
    elif uri:
        name = uri.rstrip("/").split("/")[-1] or None
    return job_dir / "inputs" / (name or fallback)


def _resolve(job_dir: Path, upload: Optional[UploadFile], uri: Optional[str],
             fallback: str, *, required: bool) -> Optional[Path]:
    dest = _dest(job_dir, upload, uri, fallback)
    if required:
        return resolve_input(upload, uri, dest, settings)
    return maybe_resolve_input(upload, uri, dest, settings)


# Per-mode `files` builders — map reinvent_cli flags → resolved on-disk paths.
# Shared by the sync (/api/*) and FC async task (/api/tasks/*) endpoints so both
# accept uploads and `*_uri` inputs identically.

def _files_sampling(job_dir, params, smiles_file):
    f = {}
    p = _resolve(job_dir, smiles_file, params.smiles_file_uri, "input.smi", required=False)
    if p is not None:
        f["smiles_file"] = p
    return f


def _files_scoring(job_dir, params, smiles_file):
    return {"smiles_file": _resolve(job_dir, smiles_file, params.smiles_file_uri,
                                    "input.smi", required=True)}


def _files_enumeration(job_dir, params, peptide_smiles, amino_acid_library):
    return {
        "smiles_file": _resolve(job_dir, peptide_smiles, params.peptide_smiles_uri,
                                "peptides.smi", required=True),
        "amino_acid_library": _resolve(job_dir, amino_acid_library, params.amino_acid_library_uri,
                                       "amino_acids.csv", required=True),
    }


def _files_transfer(job_dir, params, smiles_file, validation_smiles_file, input_model_upload):
    f = {"smiles_file": _resolve(job_dir, smiles_file, params.smiles_file_uri,
                                 "input.smi", required=True)}
    p = _resolve(job_dir, validation_smiles_file, params.validation_smiles_file_uri,
                 "validation.smi", required=False)
    if p is not None:
        f["validation_smiles_file"] = p
    p = _resolve(job_dir, input_model_upload, params.input_model_uri, "input.model", required=False)
    if p is not None:
        f["model_file"] = p
    return f


def _files_staged(job_dir, params, smiles_file, prior_upload, agent_upload):
    f = {}
    p = _resolve(job_dir, smiles_file, params.smiles_file_uri, "input.smi", required=False)
    if p is not None:
        f["smiles_file"] = p
    p = _resolve(job_dir, prior_upload, params.prior_file_uri, "prior.model", required=False)
    if p is not None:
        f["prior_file"] = p
    p = _resolve(job_dir, agent_upload, params.agent_file_uri, "agent.model", required=False)
    if p is not None:
        f["agent_file"] = p
    return f


# ---- sync endpoints ----

@app.post("/api/sampling", response_model=JobInfo,
          summary="De novo sampling from a Reinvent generator.")
def sampling(
    params: SamplingRequest = Depends(model_form_depends(SamplingRequest)),
    smiles_file: Optional[UploadFile] = File(None),
) -> JobInfo:
    def _build(job_id, job_dir):
        return sampling_argv(params, _files_sampling(job_dir, params, smiles_file), job_dir, settings)
    return app.state.runner.submit(build_argv=_build, label="sampling",
                                   input_params=params.model_dump(mode="json"))


@app.post("/api/scoring", response_model=JobInfo,
          summary="Score SMILES with a scoring function.")
def scoring(
    params: ScoringRequest = Depends(model_form_depends(ScoringRequest)),
    smiles_file: Optional[UploadFile] = File(None),
) -> JobInfo:
    def _build(job_id, job_dir):
        return scoring_argv(params, _files_scoring(job_dir, params, smiles_file), job_dir, settings)
    return app.state.runner.submit(build_argv=_build, label="scoring",
                                   input_params=params.model_dump(mode="json"))


@app.post("/api/enumeration", response_model=JobInfo,
          summary="Peptide enumeration with pepinvent.")
def enumeration(
    params: EnumerationRequest = Depends(model_form_depends(EnumerationRequest)),
    peptide_smiles: Optional[UploadFile] = File(None),
    amino_acid_library: Optional[UploadFile] = File(None),
) -> JobInfo:
    def _build(job_id, job_dir):
        return enumeration_argv(
            params, _files_enumeration(job_dir, params, peptide_smiles, amino_acid_library),
            job_dir, settings)
    return app.state.runner.submit(build_argv=_build, label="enumeration",
                                   input_params=params.model_dump(mode="json"))


@app.post("/api/transfer-learning", response_model=JobInfo,
          summary="Fine-tune a generative prior on target molecules (long-running).")
def transfer_learning(
    params: TransferLearningRequest = Depends(model_form_depends(TransferLearningRequest)),
    smiles_file: Optional[UploadFile] = File(None),
    validation_smiles_file: Optional[UploadFile] = File(None),
    input_model_upload: Optional[UploadFile] = File(None),
) -> JobInfo:
    def _build(job_id, job_dir):
        return transfer_learning_argv(
            params,
            _files_transfer(job_dir, params, smiles_file, validation_smiles_file, input_model_upload),
            job_dir, settings)
    return app.state.runner.submit(build_argv=_build, label="transfer_learning",
                                   input_params=params.model_dump(mode="json"))


@app.post("/api/staged-learning", response_model=JobInfo,
          summary="Staged learning: RL / curriculum over multiple stages (long-running).")
def staged_learning(
    params: StagedLearningRequest = Depends(model_form_depends(StagedLearningRequest)),
    smiles_file: Optional[UploadFile] = File(None),
    prior_upload: Optional[UploadFile] = File(None),
    agent_upload: Optional[UploadFile] = File(None),
) -> JobInfo:
    def _build(job_id, job_dir):
        return staged_learning_argv(
            params, _files_staged(job_dir, params, smiles_file, prior_upload, agent_upload),
            job_dir, settings)
    return app.state.runner.submit(build_argv=_build, label="staged_learning",
                                   input_params=params.model_dump(mode="json"))


# ---- FC async task endpoints (task_endpoints_enabled=True) ----

if settings.task_endpoints_enabled:

    @app.post("/api/tasks/sampling", response_model=JobInfo,
              summary="De novo sampling from a Reinvent generator (single atomic task).")
    def task_sampling(
        request: Request,
        params: SamplingRequest = Depends(model_form_depends(SamplingRequest)),
        smiles_file: Optional[UploadFile] = File(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias=settings.task_job_id_header),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        def _build(req, _job_id, job_dir):
            return sampling_argv(req, _files_sampling(job_dir, req, smiles_file), job_dir, settings)
        return execute_task(request, job_id=job_id, label="sampling",
                            params=params, build_argv=_build)

    @app.post("/api/tasks/scoring", response_model=JobInfo,
              summary="Score SMILES with a scoring function (single atomic task).")
    def task_scoring(
        request: Request,
        params: ScoringRequest = Depends(model_form_depends(ScoringRequest)),
        smiles_file: Optional[UploadFile] = File(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias=settings.task_job_id_header),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        def _build(req, _job_id, job_dir):
            return scoring_argv(req, _files_scoring(job_dir, req, smiles_file), job_dir, settings)
        return execute_task(request, job_id=job_id, label="scoring",
                            params=params, build_argv=_build)

    @app.post("/api/tasks/enumeration", response_model=JobInfo,
              summary="Peptide enumeration with pepinvent (single atomic task).")
    def task_enumeration(
        request: Request,
        params: EnumerationRequest = Depends(model_form_depends(EnumerationRequest)),
        peptide_smiles: Optional[UploadFile] = File(None),
        amino_acid_library: Optional[UploadFile] = File(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias=settings.task_job_id_header),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        def _build(req, _job_id, job_dir):
            return enumeration_argv(
                req, _files_enumeration(job_dir, req, peptide_smiles, amino_acid_library),
                job_dir, settings)
        return execute_task(request, job_id=job_id, label="enumeration",
                            params=params, build_argv=_build)

    @app.post("/api/tasks/transfer-learning", response_model=JobInfo,
              summary="Fine-tune a generative prior on target molecules (single atomic task; long-running).")
    def task_transfer_learning(
        request: Request,
        params: TransferLearningRequest = Depends(model_form_depends(TransferLearningRequest)),
        smiles_file: Optional[UploadFile] = File(None),
        validation_smiles_file: Optional[UploadFile] = File(None),
        input_model_upload: Optional[UploadFile] = File(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias=settings.task_job_id_header),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)

        def _build(req, _job_id, job_dir):
            return transfer_learning_argv(
                req,
                _files_transfer(job_dir, req, smiles_file, validation_smiles_file, input_model_upload),
                job_dir, settings)
        return execute_task(request, job_id=job_id, label="transfer_learning",
                            params=params, build_argv=_build)

    @app.post("/api/tasks/staged-learning", response_model=JobInfo,
              summary="Staged learning: RL / curriculum over multiple stages (single atomic task; long-running).")
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
            return staged_learning_argv(
                req, _files_staged(job_dir, req, smiles_file, prior_upload, agent_upload),
                job_dir, settings)
        return execute_task(request, job_id=job_id, label="staged_learning",
                            params=params, build_argv=_build)


attach_mcp(app)
