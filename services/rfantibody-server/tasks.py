"""Background task execution for RFantibody pipeline steps."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

from .models import JobInfo, JobStatus, StepName

# Path configuration — mirrors rfantibody.config.PathConfig but without the dependency.
# In Docker, set RFANTIBODY_ROOT to the RFantibody checkout directory.
_PROJECT_ROOT = Path(os.getenv("RFANTIBODY_ROOT", "/opt/rfantibody"))
_WEIGHTS_DIR = Path(os.getenv("RFANTIBODY_WEIGHTS", _PROJECT_ROOT / "weights"))
_SCRIPTS_DIR = Path(os.getenv("RFANTIBODY_SCRIPTS", _PROJECT_ROOT / "scripts"))

_WEIGHT_FILES = {
    "rfdiffusion": _WEIGHTS_DIR / "RFdiffusion_Ab.pt",
    "proteinmpnn": _WEIGHTS_DIR / "ProteinMPNN_v48_noise_0.2.pt",
    "rf2": _WEIGHTS_DIR / "RF2_ab.pt",
}


def _get_weight_path(tool: str) -> Path:
    return _WEIGHT_FILES.get(tool, _WEIGHTS_DIR / tool)

logger = logging.getLogger(__name__)

JOBS_BASE_DIR = Path("/data/rfantibody_jobs")


def _get_job_dir(job_id: str) -> Path:
    return JOBS_BASE_DIR / job_id


def _create_job_dir(job_id: str) -> Path:
    job_dir = _get_job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input").mkdir(exist_ok=True)
    (job_dir / "output").mkdir(exist_ok=True)
    return job_dir


# -- Job store (in-memory + JSON persistence on disk) --

_jobs: dict[str, JobInfo] = {}


def _job_meta_path(job_id: str) -> Path:
    return _get_job_dir(job_id) / "job.json"


def _persist_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        return
    meta_path = _job_meta_path(job_id)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(job.model_dump_json(indent=2), encoding="utf-8")


def _infer_job_from_dir(job_dir: Path) -> JobInfo:
    """Infer job metadata from directory contents (for legacy dirs without job.json)."""
    job_id = job_dir.name
    output_dir = job_dir / "output"
    output_files = sorted(f.name for f in output_dir.iterdir() if f.is_file()) if output_dir.exists() else []

    step_map = {
        "3_rf2.qv": StepName.RF2,
        "2_proteinmpnn.qv": StepName.PROTEINMPNN,
        "1_rfdiffusion.qv": StepName.RFDIFFUSION,
    }
    step = None
    for filename, step_name in step_map.items():
        if filename in output_files:
            step = step_name
            break

    if output_files:
        status = JobStatus.COMPLETED
        message = f"Recovered from disk (outputs: {', '.join(output_files)})"
    else:
        status = JobStatus.FAILED
        message = "Recovered from disk (no output files)"

    return JobInfo(job_id=job_id, status=status, step=step, message=message)


def _load_jobs_from_disk():
    """Restore job metadata from disk on startup (survives container restarts)."""
    if not JOBS_BASE_DIR.exists():
        return
    for job_dir in sorted(JOBS_BASE_DIR.iterdir()):
        if not job_dir.is_dir():
            continue
        job_id = job_dir.name
        meta_path = job_dir / "job.json"
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                job = JobInfo.model_validate(data)
                if job.status == JobStatus.RUNNING:
                    job.status = JobStatus.FAILED
                    job.message = "Interrupted by container restart"
                _jobs[job.job_id] = job
            except Exception as e:
                logger.warning("Failed to restore job from %s: %s", meta_path, e)
                continue
        else:
            job = _infer_job_from_dir(job_dir)
            _jobs[job_id] = job
            logger.info("Inferred job %s from directory contents: status=%s", job_id, job.status.value)
        _persist_job(job_id)
    logger.info("Restored %d jobs from disk", len(_jobs))


_load_jobs_from_disk()


def create_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = JobInfo(job_id=job_id, status=JobStatus.PENDING)
    _create_job_dir(job_id)
    _persist_job(job_id)
    return job_id


def get_job(job_id: str) -> Optional[JobInfo]:
    return _jobs.get(job_id)


def update_job(
    job_id: str,
    *,
    status: Optional[JobStatus] = None,
    step: Optional[StepName] = None,
    message: Optional[str] = None,
    progress: Optional[str] = None,
):
    job = _jobs.get(job_id)
    if job is None:
        return
    if status is not None:
        job.status = status
    if step is not None:
        job.step = step
    if message is not None:
        job.message = message
    if progress is not None:
        job.progress = progress
    _persist_job(job_id)


def list_output_files(job_id: str) -> list[str]:
    output_dir = _get_job_dir(job_id) / "output"
    if not output_dir.exists():
        return []
    return sorted(f.name for f in output_dir.iterdir() if f.is_file())


def cleanup_job(job_id: str):
    job_dir = _get_job_dir(job_id)
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    _jobs.pop(job_id, None)


# -- Step runners --

_STEP_TIMEOUTS = {
    "rfdiffusion": 2 * 3600,
    "proteinmpnn": 2 * 3600,
    "rf2": 6 * 3600,
}


def _run_subprocess(cmd: list[str], step_name: str, job_id: str) -> int:
    logger.info("[%s] Running %s: %s", job_id, step_name, " ".join(cmd))
    log_dir = _get_job_dir(job_id) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{step_name}.log"
    timeout = _STEP_TIMEOUTS.get(step_name, 4 * 3600)

    try:
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                cmd,
                cwd=str(_PROJECT_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        logger.error("[%s] %s timed out after %ds", job_id, step_name, timeout)
        return -1

    if result.returncode != 0:
        tail = _read_log_tail(log_path, 2000)
        logger.error("[%s] %s failed (rc=%d):\n%s", job_id, step_name, result.returncode, tail)
    else:
        tail = _read_log_tail(log_path, 500)
        logger.info("[%s] %s completed (rc=0). tail: %s", job_id, step_name, tail)
    return result.returncode


def _read_log_tail(path: Path, chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-chars:] if len(text) > chars else text
    except OSError:
        return "(log file unreadable)"


def _check_weights(tool: str) -> Path:
    weights = _get_weight_path(tool)
    if not weights.exists():
        logger.warning("Weight file not found for %s: %s (tool will use its default)", tool, weights)
    return weights


def run_rfdiffusion(
    job_id: str,
    target_pdb: Path,
    framework_pdb: Path,
    *,
    num_designs: int = 10,
    design_loops: str = "H1:,H2:,H3:",
    hotspots: Optional[str] = None,
    diffuser_t: int = 50,
    final_step: int = 1,
    deterministic: bool = False,
    no_trajectory: bool = True,
) -> int:
    job_dir = _get_job_dir(job_id)
    output_qv = job_dir / "output" / "1_rfdiffusion.qv"

    script = _SCRIPTS_DIR / "rfdiffusion_inference.py"
    cmd = [
        sys.executable, str(script),
        "--config-name", "antibody",
        f"antibody.target_pdb={target_pdb.resolve()}",
        f"antibody.framework_pdb={framework_pdb.resolve()}",
        f"inference.quiver={output_qv}",
        f"inference.num_designs={num_designs}",
        f"diffuser.T={diffuser_t}",
        f"inference.final_step={final_step}",
    ]

    loops_list = [x.strip() for x in design_loops.split(",")]
    cmd.append(f"antibody.design_loops=[{','.join(loops_list)}]")

    if hotspots:
        hs = [h.strip() for h in hotspots.split(",")]
        cmd.append(f"ppi.hotspot_res=[{','.join(hs)}]")

    weights = _check_weights("rfdiffusion")
    if weights.exists():
        cmd.append(f"inference.ckpt_override_path={weights}")

    if deterministic:
        cmd.append("inference.deterministic=True")
    if no_trajectory:
        cmd.append("inference.write_trajectory=False")

    return _run_subprocess(cmd, "rfdiffusion", job_id)


def run_proteinmpnn(
    job_id: str,
    input_quiver: Path,
    *,
    loops: str = "H1,H2,H3",
    seqs_per_struct: int = 4,
    temperature: float = 0.2,
    omit_aas: str = "CX",
    deterministic: bool = False,
) -> int:
    job_dir = _get_job_dir(job_id)
    output_qv = job_dir / "output" / "2_proteinmpnn.qv"

    script = _SCRIPTS_DIR / "proteinmpnn_interface_design.py"
    cmd = [
        sys.executable, str(script),
        "-quiver", str(input_quiver.resolve()),
        "-outquiver", str(output_qv),
        "-loop_string", loops,
        "-seqs_per_struct", str(seqs_per_struct),
        "-temperature", str(temperature),
        "-omit_AAs", omit_aas,
    ]

    weights = _check_weights("proteinmpnn")
    if weights.exists():
        cmd.extend(["-checkpoint_path", str(weights)])

    if deterministic:
        cmd.append("-deterministic")

    return _run_subprocess(cmd, "proteinmpnn", job_id)


def run_rf2(
    job_id: str,
    input_quiver: Path,
    *,
    num_recycles: int = 10,
    hotspot_show_prop: float = 0.1,
    seed: Optional[int] = None,
) -> int:
    job_dir = _get_job_dir(job_id)
    output_qv = job_dir / "output" / "3_rf2.qv"

    script = _SCRIPTS_DIR / "rf2_predict.py"
    cmd = [
        sys.executable, str(script),
        f"input.quiver={input_quiver.resolve()}",
        f"output.quiver={output_qv}",
        f"inference.num_recycles={num_recycles}",
        f"inference.hotspot_show_proportion={hotspot_show_prop}",
        "inference.cautious=False",
    ]

    weights = _check_weights("rf2")
    if weights.exists():
        cmd.append(f"model.model_weights={weights}")

    if seed is not None:
        cmd.append(f"+inference.seed={seed}")

    return _run_subprocess(cmd, "rf2", job_id)


def _check_step_output(job_id: str, step_name: str, output_path: Path) -> bool:
    if not output_path.exists() or output_path.stat().st_size == 0:
        update_job(job_id, status=JobStatus.FAILED,
                   message=f"{step_name} exited 0 but output file missing: {output_path.name}")
        return False
    return True


def run_pipeline(
    job_id: str,
    target_pdb: Path,
    framework_pdb: Path,
    *,
    rfdiffusion_kwargs: dict,
    proteinmpnn_kwargs: dict,
    rf2_kwargs: dict,
):
    """Run the full 3-step pipeline. Intended to be called in a background thread."""
    job_dir = _get_job_dir(job_id)

    # Step 1: RFdiffusion
    update_job(job_id, status=JobStatus.RUNNING, step=StepName.RFDIFFUSION, progress="1/3")
    rc = run_rfdiffusion(job_id, target_pdb, framework_pdb, **rfdiffusion_kwargs)
    if rc != 0:
        update_job(job_id, status=JobStatus.FAILED, message=f"RFdiffusion failed (rc={rc})")
        return

    rfdiff_qv = job_dir / "output" / "1_rfdiffusion.qv"
    if not _check_step_output(job_id, "RFdiffusion", rfdiff_qv):
        return

    # Step 2: ProteinMPNN
    update_job(job_id, step=StepName.PROTEINMPNN, progress="2/3")
    rc = run_proteinmpnn(job_id, rfdiff_qv, **proteinmpnn_kwargs)
    if rc != 0:
        update_job(job_id, status=JobStatus.FAILED, message=f"ProteinMPNN failed (rc={rc})")
        return

    mpnn_qv = job_dir / "output" / "2_proteinmpnn.qv"
    if not _check_step_output(job_id, "ProteinMPNN", mpnn_qv):
        return

    # Step 3: RF2
    update_job(job_id, step=StepName.RF2, progress="3/3")
    rc = run_rf2(job_id, mpnn_qv, **rf2_kwargs)
    if rc != 0:
        update_job(job_id, status=JobStatus.FAILED, message=f"RF2 failed (rc={rc})")
        return

    rf2_qv = job_dir / "output" / "3_rf2.qv"
    if not _check_step_output(job_id, "RF2", rf2_qv):
        return

    update_job(job_id, status=JobStatus.COMPLETED, message="Pipeline completed", progress="3/3")
