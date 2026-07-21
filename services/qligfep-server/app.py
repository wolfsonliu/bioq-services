"""FastAPI app for qligfep-server.

HTTP endpoints mirror the CLI batch entries. `settings.task_endpoints_enabled`
is False by default — no `/api/tasks/*` registration (HPC-primary; not
deployed to FC).  See engineering/decisions/2026-07-06-qligfep-server-design.md.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from bioq_service import (
    JobInfo,
    attach_mcp,
    create_app,
    model_form_depends,
    read_version_file,
)
from fastapi import Depends, File, Form, Request, UploadFile

from .adapter import QligfepAdapter
from .models import (
    AnalyzeFepRequest, AnalyzeLieRequest, CogRequest, LigprepRequest,
    ProtprepRequest, RunFepRequest, SetupLieRequest, SetupLigfepRequest,
    SetupResfepRequest,
)
from .settings import QligfepSettings
from .tools import (
    analyze_fep_argv, analyze_lie_argv, cog_argv, ligprep_argv,
    protprep_argv, run_fep_argv, setup_lie_argv, setup_ligfep_argv,
    setup_resfep_argv,
)

settings = QligfepSettings()
adapter = QligfepAdapter(settings=settings)

app = create_app(
    adapter, settings,
    title="Qligfep Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---- /healthz/detail override: report Q6 binaries + settings shim ----

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
    """Q6 binary + qligfep settings shim readiness probe."""
    expected = {
        "qdyn":       settings.q_bin_dir / "qdyn",
        "qdynp":      settings.q_bin_dir / "qdynp",
        "qdyn_cuda":  settings.q_bin_dir / "qdyn_cuda",
        "qprep":      settings.q_bin_dir / "qprep",
        "qfep":       settings.q_bin_dir / "qfep",
        "qcalc":      settings.q_bin_dir / "qcalc",
        "qligfep_settings_shim": settings.upstream_dir / "settings.py",
        "opls_ff":    settings.upstream_dir / "FF" / "OPLSAAM.prm",
    }
    missing = {k: str(p) for k, p in expected.items() if not p.exists()}
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "binaries_dir": str(settings.q_bin_dir),
        "binaries_loaded": not missing,
        "binaries_missing": missing,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
        "task_endpoints_enabled": settings.task_endpoints_enabled,
    }


# ---- helpers ----

def _persist_upload(upload: UploadFile, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dst


def _unzip(zip_path: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dst)
    return dst


# ---- endpoints ----

@app.post("/api/ligprep", response_model=JobInfo)
def ligprep(
    params: LigprepRequest = Depends(model_form_depends(LigprepRequest)),
    ligand: UploadFile = File(...),
) -> JobInfo:
    def _build(job_id: str, job_dir: Path) -> list[str]:
        lig_path = _persist_upload(ligand, job_dir / "inputs" / ligand.filename)
        return ligprep_argv(params, lig_path, job_dir, settings)
    return app.state.runner.submit(
        build_argv=_build, label="ligprep",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/protprep", response_model=JobInfo)
def protprep(
    params: ProtprepRequest = Depends(model_form_depends(ProtprepRequest)),
    protein_pdb: UploadFile = File(...),
) -> JobInfo:
    def _build(job_id, job_dir):
        pdb = _persist_upload(protein_pdb, job_dir / "inputs" / protein_pdb.filename)
        return protprep_argv(params, pdb, job_dir, settings)
    return app.state.runner.submit(
        build_argv=_build, label="protprep",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/cog", response_model=JobInfo)
def cog(
    params: CogRequest = Depends(model_form_depends(CogRequest)),
    pdb: UploadFile = File(...),
) -> JobInfo:
    def _build(job_id, job_dir):
        p = _persist_upload(pdb, job_dir / "inputs" / pdb.filename)
        return cog_argv(params, p, job_dir, settings)
    return app.state.runner.submit(
        build_argv=_build, label="cog",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/setup-ligfep", response_model=JobInfo)
def setup_ligfep(
    params: SetupLigfepRequest = Depends(model_form_depends(SetupLigfepRequest)),
    ligprep_zip: UploadFile = File(...),
    protprep_zip: UploadFile = File(...),
) -> JobInfo:
    def _build(job_id, job_dir):
        lz = _persist_upload(ligprep_zip, job_dir / "inputs" / "ligprep.zip")
        pz = _persist_upload(protprep_zip, job_dir / "inputs" / "protprep.zip")
        lig_dir = _unzip(lz, job_dir / "inputs" / "ligprep")
        prot_dir = _unzip(pz, job_dir / "inputs" / "protprep")
        return setup_ligfep_argv(params, lig_dir, prot_dir, job_dir, settings)
    return app.state.runner.submit(
        build_argv=_build, label="setup-ligfep",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/setup-resfep", response_model=JobInfo)
def setup_resfep(
    params: SetupResfepRequest = Depends(model_form_depends(SetupResfepRequest)),
    protprep_zip: UploadFile = File(...),
) -> JobInfo:
    def _build(job_id, job_dir):
        pz = _persist_upload(protprep_zip, job_dir / "inputs" / "protprep.zip")
        prot_dir = _unzip(pz, job_dir / "inputs" / "protprep")
        return setup_resfep_argv(params, prot_dir, job_dir, settings)
    return app.state.runner.submit(
        build_argv=_build, label="setup-resfep",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/setup-lie", response_model=JobInfo)
def setup_lie(
    params: SetupLieRequest = Depends(model_form_depends(SetupLieRequest)),
    ligprep_zip: UploadFile = File(...),
    protprep_zip: UploadFile = File(...),
) -> JobInfo:
    def _build(job_id, job_dir):
        lz = _persist_upload(ligprep_zip, job_dir / "inputs" / "ligprep.zip")
        pz = _persist_upload(protprep_zip, job_dir / "inputs" / "protprep.zip")
        lig_dir = _unzip(lz, job_dir / "inputs" / "ligprep")
        prot_dir = _unzip(pz, job_dir / "inputs" / "protprep")
        return setup_lie_argv(params, lig_dir, prot_dir, job_dir, settings)
    return app.state.runner.submit(
        build_argv=_build, label="setup-lie",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/run-fep", response_model=JobInfo)
def run_fep(
    params: RunFepRequest = Depends(model_form_depends(RunFepRequest)),
    setup_zip: UploadFile = File(...),
) -> JobInfo:
    def _build(job_id, job_dir):
        sz = _persist_upload(setup_zip, job_dir / "inputs" / "setup.zip")
        setup_dir = _unzip(sz, job_dir / "inputs" / "setup")
        return run_fep_argv(params, setup_dir, job_dir, settings)
    return app.state.runner.submit(
        build_argv=_build, label="run-fep",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/analyze-fep", response_model=JobInfo)
def analyze_fep(
    params: AnalyzeFepRequest = Depends(model_form_depends(AnalyzeFepRequest)),
    fep_run_zip: UploadFile = File(...),
) -> JobInfo:
    def _build(job_id, job_dir):
        z = _persist_upload(fep_run_zip, job_dir / "inputs" / "fep_run.zip")
        run_dir = _unzip(z, job_dir / "inputs" / "run")
        return analyze_fep_argv(params, run_dir, job_dir, settings)
    return app.state.runner.submit(
        build_argv=_build, label="analyze-fep",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/analyze-lie", response_model=JobInfo)
def analyze_lie(
    params: AnalyzeLieRequest = Depends(model_form_depends(AnalyzeLieRequest)),
    lie_run_zip: UploadFile = File(...),
) -> JobInfo:
    def _build(job_id, job_dir):
        z = _persist_upload(lie_run_zip, job_dir / "inputs" / "lie_run.zip")
        run_dir = _unzip(z, job_dir / "inputs" / "run")
        return analyze_lie_argv(params, run_dir, job_dir, settings)
    return app.state.runner.submit(
        build_argv=_build, label="analyze-lie",
        input_params=params.model_dump(mode="json"),
    )


# Task endpoints intentionally NOT registered — settings.task_endpoints_enabled
# is False; see design doc §2.2.
assert settings.task_endpoints_enabled is False, (
    "qligfep-server is HPC-primary and does not register /api/tasks/*. "
    "If you change task_endpoints_enabled, also review whether FC deployment "
    "makes sense given run-fep's 30 min - 2 h wall clock per window."
)

attach_mcp(app)
