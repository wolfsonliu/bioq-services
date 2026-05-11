"""FastAPI web server for `genie3 generate`.

Exposes a thin HTTP wrapper around the `genie3 generate` CLI. Designed for deployment on
Alibaba Cloud Function Compute (FC) GPU instances, mirroring the pattern of
`services/rfantibody-server`.

The server supports three task kinds:
  - unconditional protein generation (no dataset)
  - motif scaffolding (requires a problem-set zip with motifs/ + problems/)
  - binder design (requires a problem-set zip with targets/ + problems/)

A generic `/api/generate` endpoint is also available for fully custom configs.

FC deployment requirements:
  - HTTP server listens on 0.0.0.0:CAPort (default 9000)
  - Server must respond to /health within 120s
  - Keep-alive timeout >= 15 minutes
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from . import tasks
from .models import JobInfo, JobStatus, TaskKind

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Genie3 Server",
    description="HTTP API for genie3 generate (unconditional / motif / binder protein design)",
    version="0.1.0",
)

executor = ThreadPoolExecutor(max_workers=1)

DISK_LIMIT_MB = int(os.getenv("GENIE3_DISK_LIMIT_MB", "8000"))


def _save_upload(upload: UploadFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(upload.file.read())
    return dest


def _check_disk_usage():
    jobs_dir = tasks.JOBS_BASE_DIR
    if not jobs_dir.exists():
        return
    total = sum(f.stat().st_size for f in jobs_dir.rglob("*") if f.is_file())
    if total > DISK_LIMIT_MB * 1024 * 1024:
        logger.warning("Disk usage %.1f MB exceeds limit %d MB, cleaning up old jobs",
                       total / 1024 / 1024, DISK_LIMIT_MB)
        for job_id, job in list(tasks._jobs.items()):
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                tasks.cleanup_job(job_id)


def _finalize(job_id: str, rc: int, label: str):
    if rc != 0:
        tasks.update_job(job_id, status=JobStatus.FAILED, message=f"{label} failed (rc={rc})")
    elif not tasks.has_outputs(job_id):
        tasks.update_job(job_id, status=JobStatus.FAILED,
                         message=f"{label} exited 0 but no PDB outputs were produced")
    else:
        tasks.update_job(job_id, status=JobStatus.COMPLETED, message=f"{label} completed")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/detail")
def health_detail():
    return {
        "status": "ok",
        "paths": {
            "project_root": str(tasks._PROJECT_ROOT),
            "project_root_exists": tasks._PROJECT_ROOT.exists(),
            "jobs_dir": str(tasks.JOBS_BASE_DIR),
            "jobs_dir_exists": tasks.JOBS_BASE_DIR.exists(),
        },
    }


# ---------------------------------------------------------------------------
# Task-specific generation endpoints
# ---------------------------------------------------------------------------


@app.post("/api/generate/unconditional", response_model=JobInfo)
def generate_unconditional(
    min_length: int = Form(100),
    max_length: int = Form(100),
    length_step: int = Form(50),
    n_sample: int = Form(4),
    direction_scale: float = Form(0.8),
    batch_size: int = Form(1),
    num_devices: Optional[int] = Form(None),
):
    """Run unconditional backbone generation."""
    _check_disk_usage()
    job_id = tasks.create_job(TaskKind.UNCONDITIONAL)
    job_dir = tasks._get_job_dir(job_id)

    config = tasks.build_unconditional_config(
        rootdir=job_dir / "output",
        min_length=min_length,
        max_length=max_length,
        length_step=length_step,
        n_sample=n_sample,
        direction_scale=direction_scale,
        batch_size=batch_size,
    )

    def _run():
        tasks.update_job(job_id, status=JobStatus.RUNNING)
        rc = tasks.run_generate(job_id, config, num_devices=num_devices)
        _finalize(job_id, rc, "unconditional generate")

    executor.submit(_run)
    return tasks.get_job(job_id)


@app.post("/api/generate/motif", response_model=JobInfo)
def generate_motif(
    dataset: UploadFile = File(..., description="Zip of the motif scaffolding problem set (problems/ + motifs/)"),
    selections: Optional[str] = Form(None),
    n_sample: int = Form(4),
    direction_scale: float = Form(0.1),
    batch_size: int = Form(1),
    num_devices: Optional[int] = Form(None),
):
    """Run motif scaffolding generation. Upload a dataset zip with `problems/` and `motifs/`."""
    _check_disk_usage()
    job_id = tasks.create_job(TaskKind.MOTIF)
    job_dir = tasks._get_job_dir(job_id)

    zip_path = _save_upload(dataset, job_dir / "input" / "dataset.zip")
    try:
        dataset_root = tasks.extract_dataset(zip_path, job_dir / "input" / "dataset")
    except (zipfile.BadZipFile, ValueError) as e:
        tasks.cleanup_job(job_id)
        raise HTTPException(status_code=422, detail=f"Invalid dataset zip: {e}")

    config = tasks.build_motif_config(
        rootdir=job_dir / "output",
        dataset_root=dataset_root,
        selections=selections,
        n_sample=n_sample,
        direction_scale=direction_scale,
        batch_size=batch_size,
    )

    def _run():
        tasks.update_job(job_id, status=JobStatus.RUNNING)
        rc = tasks.run_generate(job_id, config, num_devices=num_devices)
        _finalize(job_id, rc, "motif generate")

    executor.submit(_run)
    return tasks.get_job(job_id)


@app.post("/api/generate/binder", response_model=JobInfo)
def generate_binder(
    dataset: UploadFile = File(..., description="Zip of the binder design problem set (problems/ + targets/)"),
    selections: Optional[str] = Form(None),
    n_sample: int = Form(4),
    direction_scale: float = Form(0.0),
    batch_size: int = Form(1),
    num_devices: Optional[int] = Form(None),
):
    """Run binder design generation. Upload a dataset zip with `problems/` and `targets/`."""
    _check_disk_usage()
    job_id = tasks.create_job(TaskKind.BINDER)
    job_dir = tasks._get_job_dir(job_id)

    zip_path = _save_upload(dataset, job_dir / "input" / "dataset.zip")
    try:
        dataset_root = tasks.extract_dataset(zip_path, job_dir / "input" / "dataset")
    except (zipfile.BadZipFile, ValueError) as e:
        tasks.cleanup_job(job_id)
        raise HTTPException(status_code=422, detail=f"Invalid dataset zip: {e}")

    config = tasks.build_binder_config(
        rootdir=job_dir / "output",
        dataset_root=dataset_root,
        selections=selections,
        n_sample=n_sample,
        direction_scale=direction_scale,
        batch_size=batch_size,
    )

    def _run():
        tasks.update_job(job_id, status=JobStatus.RUNNING)
        rc = tasks.run_generate(job_id, config, num_devices=num_devices)
        _finalize(job_id, rc, "binder generate")

    executor.submit(_run)
    return tasks.get_job(job_id)


# ---------------------------------------------------------------------------
# Generic generation endpoint (custom config)
# ---------------------------------------------------------------------------


@app.post("/api/generate", response_model=JobInfo)
def generate_custom(
    config_yaml: str = Form(..., description="Full experiment YAML as a string"),
    dataset: Optional[UploadFile] = File(None, description="Optional dataset zip; extracted to <job>/input/dataset/"),
    num_devices: Optional[int] = Form(None),
):
    """Run `genie3 generate` with a fully custom YAML config.

    `paths.rootdir` and (if a dataset is provided) `paths.dataset` are rewritten to
    point at the job-local directory so the config is portable.
    """
    _check_disk_usage()
    try:
        config = yaml.safe_load(config_yaml)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=422, detail=f"Invalid YAML: {e}")
    if not isinstance(config, dict):
        raise HTTPException(status_code=422, detail="config_yaml must be a mapping at the top level")

    job_id = tasks.create_job(TaskKind.CUSTOM)
    job_dir = tasks._get_job_dir(job_id)

    paths = config.setdefault("paths", {})
    paths["rootdir"] = str((job_dir / "output").resolve())

    if dataset is not None:
        zip_path = _save_upload(dataset, job_dir / "input" / "dataset.zip")
        try:
            dataset_root = tasks.extract_dataset(zip_path, job_dir / "input" / "dataset")
        except (zipfile.BadZipFile, ValueError) as e:
            tasks.cleanup_job(job_id)
            raise HTTPException(status_code=422, detail=f"Invalid dataset zip: {e}")
        paths["dataset"] = str(dataset_root.resolve())

    def _run():
        tasks.update_job(job_id, status=JobStatus.RUNNING)
        rc = tasks.run_generate(job_id, config, num_devices=num_devices)
        _finalize(job_id, rc, "custom generate")

    executor.submit(_run)
    return tasks.get_job(job_id)


# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------


@app.get("/api/jobs/{job_id}", response_model=JobInfo)
def get_job_status(job_id: str):
    job = tasks.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/files")
def list_job_files(job_id: str):
    job = tasks.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "files": tasks.list_output_files(job_id)}


@app.get("/api/jobs/{job_id}/log")
def get_job_log(job_id: str):
    job = tasks.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    log_path = tasks._get_job_dir(job_id) / "logs" / "generate.log"
    if not log_path.exists():
        return {"job_id": job_id, "log": ""}
    return {"job_id": job_id, "log": log_path.read_text(encoding="utf-8", errors="replace")}


@app.get("/api/jobs/{job_id}/download")
def download_job_results(job_id: str):
    job = tasks.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job is {job.status.value}, not completed")

    output_dir = tasks._get_job_dir(job_id) / "output"
    files = [f for f in output_dir.rglob("*") if f.is_file()]
    if not files:
        raise HTTPException(status_code=404, detail="No output files")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.relative_to(output_dir))
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=genie3_{job_id}.zip"},
    )


@app.get("/api/jobs/{job_id}/file/{file_path:path}")
def download_single_file(job_id: str, file_path: str):
    job = tasks.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = tasks._get_job_dir(job_id) / "output"
    requested = (output_dir / file_path).resolve()
    try:
        requested.relative_to(output_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Path traversal not allowed")
    if not requested.exists() or not requested.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=requested,
        media_type="application/octet-stream",
        filename=requested.name,
    )


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = tasks.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    tasks.cleanup_job(job_id)
    return {"status": "deleted", "job_id": job_id}
