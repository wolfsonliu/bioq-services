"""FastAPI web server for RFantibody pipeline.

Designed for deployment on Alibaba Cloud Function Compute (FC) with GPU instances.
Provides HTTP endpoints for running the 3-step antibody design pipeline
(RFdiffusion → ProteinMPNN → RF2) either individually or as a full pipeline.

FC deployment requirements (https://help.aliyun.com/zh/functioncompute/fc/):
  - HTTP server must listen on 0.0.0.0:CAPort (default 9000, NOT 127.0.0.1)
  - Server must start and respond within 120 seconds
  - Keep-alive timeout must be >= 15 minutes (900s)
  - Writable disk is limited (512MB default, up to 10GB with config)
  - Image must be AMD64 and hosted in ACR (same region/account)
  - GPU images up to 15GB uncompressed
"""

import io
import logging
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from . import tasks
from .models import (
    JobInfo,
    JobStatus,
    PipelineRequest,
    StepName,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RFantibody Server",
    description="HTTP API for RFantibody antibody design pipeline (RFdiffusion → ProteinMPNN → RF2)",
    version="0.1.0",
)

executor = ThreadPoolExecutor(max_workers=1)

# FC writable disk limit — auto-cleanup when usage exceeds threshold
DISK_LIMIT_MB = int(os.getenv("RFANTIBODY_DISK_LIMIT_MB", "8000"))


def _save_upload(upload: UploadFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(upload.file.read())
    return dest


def _check_disk_usage():
    """Auto-cleanup completed/failed jobs when disk usage is high."""
    jobs_dir = tasks.JOBS_BASE_DIR
    if not jobs_dir.exists():
        return
    total = sum(f.stat().st_size for f in jobs_dir.rglob("*") if f.is_file())
    if total > DISK_LIMIT_MB * 1024 * 1024:
        logger.warning("Disk usage %.1f MB exceeds limit %d MB, cleaning up old jobs", total / 1024 / 1024, DISK_LIMIT_MB)
        for job_id, job in list(tasks._jobs.items()):
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                tasks.cleanup_job(job_id)


# ---------------------------------------------------------------------------
# Health & initialization
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Lightweight health check for FC startup probe. Must respond quickly (<120s)."""
    return {"status": "ok"}


@app.get("/health/detail")
def health_detail():
    """Detailed health check including path validation."""
    path_status = {
        "project_root": str(tasks._PROJECT_ROOT),
        "project_root_exists": tasks._PROJECT_ROOT.exists(),
        "weights_dir": str(tasks._WEIGHTS_DIR),
        "weights_dir_exists": tasks._WEIGHTS_DIR.exists(),
        "scripts_dir": str(tasks._SCRIPTS_DIR),
        "scripts_dir_exists": tasks._SCRIPTS_DIR.exists(),
        "jobs_dir": str(tasks.JOBS_BASE_DIR),
        "jobs_dir_exists": tasks.JOBS_BASE_DIR.exists(),
    }
    return {"status": "ok", "paths": path_status}


# ---------------------------------------------------------------------------
# Single-step endpoints
# ---------------------------------------------------------------------------


@app.post("/api/rfdiffusion", response_model=JobInfo)
def run_rfdiffusion_endpoint(
    target: UploadFile = File(..., description="Target antigen PDB file"),
    framework: UploadFile = File(..., description="Antibody framework PDB file"),
    num_designs: int = Form(10),
    design_loops: str = Form("H1:,H2:,H3:"),
    hotspots: Optional[str] = Form(None),
    diffuser_t: int = Form(50),
    final_step: int = Form(1),
    deterministic: bool = Form(False),
    no_trajectory: bool = Form(True),
):
    """Run RFdiffusion backbone design (Step 1)."""
    _check_disk_usage()
    job_id = tasks.create_job()
    job_dir = tasks._get_job_dir(job_id)

    target_path = _save_upload(target, job_dir / "input" / "target.pdb")
    framework_path = _save_upload(framework, job_dir / "input" / "framework.pdb")

    def _run():
        tasks.update_job(job_id, status=JobStatus.RUNNING, step=StepName.RFDIFFUSION)
        rc = tasks.run_rfdiffusion(
            job_id, target_path, framework_path,
            num_designs=num_designs,
            design_loops=design_loops,
            hotspots=hotspots,
            diffuser_t=diffuser_t,
            final_step=final_step,
            deterministic=deterministic,
            no_trajectory=no_trajectory,
        )
        if rc == 0:
            tasks.update_job(job_id, status=JobStatus.COMPLETED, message="RFdiffusion completed")
        else:
            tasks.update_job(job_id, status=JobStatus.FAILED, message=f"RFdiffusion failed (rc={rc})")

    executor.submit(_run)
    return tasks.get_job(job_id)


@app.post("/api/proteinmpnn", response_model=JobInfo)
def run_proteinmpnn_endpoint(
    input_quiver: UploadFile = File(..., description="Input Quiver file from RFdiffusion"),
    loops: str = Form("H1,H2,H3"),
    seqs_per_struct: int = Form(4),
    temperature: float = Form(0.2),
    omit_aas: str = Form("CX"),
    deterministic: bool = Form(False),
):
    """Run ProteinMPNN sequence design (Step 2)."""
    _check_disk_usage()
    job_id = tasks.create_job()
    job_dir = tasks._get_job_dir(job_id)

    qv_path = _save_upload(input_quiver, job_dir / "input" / "input.qv")

    def _run():
        tasks.update_job(job_id, status=JobStatus.RUNNING, step=StepName.PROTEINMPNN)
        rc = tasks.run_proteinmpnn(
            job_id, qv_path,
            loops=loops,
            seqs_per_struct=seqs_per_struct,
            temperature=temperature,
            omit_aas=omit_aas,
            deterministic=deterministic,
        )
        if rc == 0:
            tasks.update_job(job_id, status=JobStatus.COMPLETED, message="ProteinMPNN completed")
        else:
            tasks.update_job(job_id, status=JobStatus.FAILED, message=f"ProteinMPNN failed (rc={rc})")

    executor.submit(_run)
    return tasks.get_job(job_id)


@app.post("/api/rf2", response_model=JobInfo)
def run_rf2_endpoint(
    input_quiver: UploadFile = File(..., description="Input Quiver file from ProteinMPNN"),
    num_recycles: int = Form(10),
    hotspot_show_prop: float = Form(0.1),
    seed: Optional[int] = Form(None),
):
    """Run RF2 structure prediction (Step 3)."""
    _check_disk_usage()
    job_id = tasks.create_job()
    job_dir = tasks._get_job_dir(job_id)

    qv_path = _save_upload(input_quiver, job_dir / "input" / "input.qv")

    def _run():
        tasks.update_job(job_id, status=JobStatus.RUNNING, step=StepName.RF2)
        rc = tasks.run_rf2(
            job_id, qv_path,
            num_recycles=num_recycles,
            hotspot_show_prop=hotspot_show_prop,
            seed=seed,
        )
        if rc == 0:
            tasks.update_job(job_id, status=JobStatus.COMPLETED, message="RF2 completed")
        else:
            tasks.update_job(job_id, status=JobStatus.FAILED, message=f"RF2 failed (rc={rc})")

    executor.submit(_run)
    return tasks.get_job(job_id)


# ---------------------------------------------------------------------------
# Full pipeline endpoint
# ---------------------------------------------------------------------------


@app.post("/api/pipeline", response_model=JobInfo)
def run_pipeline_endpoint(
    target: UploadFile = File(..., description="Target antigen PDB file"),
    framework: UploadFile = File(..., description="Antibody framework PDB file"),
    config: str = Form("{}"),
):
    """Run the full 3-step pipeline (RFdiffusion → ProteinMPNN → RF2).

    The `config` field accepts a JSON string matching PipelineRequest schema.
    If omitted, default parameters are used.
    """
    import json
    try:
        cfg = PipelineRequest.model_validate(json.loads(config))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid config: {e}")

    _check_disk_usage()
    job_id = tasks.create_job()
    job_dir = tasks._get_job_dir(job_id)

    target_path = _save_upload(target, job_dir / "input" / "target.pdb")
    framework_path = _save_upload(framework, job_dir / "input" / "framework.pdb")

    def _run():
        tasks.run_pipeline(
            job_id,
            target_path,
            framework_path,
            rfdiffusion_kwargs=cfg.rfdiffusion.model_dump(),
            proteinmpnn_kwargs=cfg.proteinmpnn.model_dump(),
            rf2_kwargs=cfg.rf2.model_dump(),
        )

    tasks.update_job(job_id, status=JobStatus.PENDING, message="Pipeline queued")
    executor.submit(_run)
    return tasks.get_job(job_id)


# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------


@app.get("/api/jobs/{job_id}", response_model=JobInfo)
def get_job_status(job_id: str):
    """Get the status of a job."""
    job = tasks.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/files")
def list_job_files(job_id: str):
    """List output files for a completed job."""
    job = tasks.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    files = tasks.list_output_files(job_id)
    return {"job_id": job_id, "files": files}


@app.get("/api/jobs/{job_id}/download")
def download_job_results(job_id: str):
    """Download all output files as a zip archive."""
    job = tasks.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job is {job.status.value}, not completed")

    output_dir = tasks._get_job_dir(job_id) / "output"
    files = list(output_dir.iterdir())
    if not files:
        raise HTTPException(status_code=404, detail="No output files")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.is_file():
                zf.write(f, f.name)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=rfantibody_{job_id}.zip"},
    )


@app.get("/api/jobs/{job_id}/file/{filename}")
def download_single_file(job_id: str, filename: str):
    """Download a single output file."""
    job = tasks.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    filepath = tasks._get_job_dir(job_id) / "output" / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return StreamingResponse(
        open(filepath, "rb"),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    """Delete a job and its files."""
    job = tasks.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    tasks.cleanup_job(job_id)
    return {"status": "deleted", "job_id": job_id}
