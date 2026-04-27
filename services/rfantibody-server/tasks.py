"""Background task execution for RFantibody pipeline steps."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from .models import JobInfo, JobStatus, StepName

# Path configuration — mirrors rfantibody.config.PathConfig but without the dependency.
# In Docker, set RFANTIBODY_ROOT to the RFantibody checkout directory.
_PROJECT_ROOT = Path(os.getenv("RFANTIBODY_ROOT", "/opt/rfantibody"))
_WEIGHTS_DIR = Path(os.getenv("RFANTIBODY_WEIGHTS", _PROJECT_ROOT / "weights"))
_SCRIPTS_DIR = Path(os.getenv("RFANTIBODY_SCRIPTS", _PROJECT_ROOT / "scripts"))

_WEIGHT_SUBDIRS = {
    "rfdiffusion": _WEIGHTS_DIR / "rfdiffusion",
    "proteinmpnn": _WEIGHTS_DIR / "proteinmpnn",
    "rf2": _WEIGHTS_DIR / "rf2",
}


def _get_weight_path(tool: str) -> Path:
    return _WEIGHT_SUBDIRS.get(tool, _WEIGHTS_DIR / tool)

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


# -- In-memory job store (sufficient for single-instance FC) --

_jobs: dict[str, JobInfo] = {}


def create_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = JobInfo(job_id=job_id, status=JobStatus.PENDING)
    _create_job_dir(job_id)
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


def _run_subprocess(cmd: list[str], step_name: str, job_id: str) -> int:
    logger.info("[%s] Running %s: %s", job_id, step_name, " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("[%s] %s failed (rc=%d):\n%s", job_id, step_name, result.returncode, result.stderr[-2000:])
    else:
        logger.info("[%s] %s completed successfully", job_id, step_name)
    return result.returncode


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
        "python", str(script),
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

    weights = _get_weight_path("rfdiffusion")
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
        "python", str(script),
        "-quiver", str(input_quiver.resolve()),
        "-outquiver", str(output_qv),
        "-loop_string", loops,
        "-seqs_per_struct", str(seqs_per_struct),
        "-temperature", str(temperature),
        "-omit_AAs", omit_aas,
    ]

    weights = _get_weight_path("proteinmpnn")
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
        "python", str(script),
        f"input.quiver={input_quiver.resolve()}",
        f"output.quiver={output_qv}",
        f"inference.num_recycles={num_recycles}",
        f"inference.hotspot_show_proportion={hotspot_show_prop}",
        "inference.cautious=False",
    ]

    weights = _get_weight_path("rf2")
    if weights.exists():
        cmd.append(f"model.model_weights='{weights}'")

    if seed is not None:
        cmd.append(f"+inference.seed={seed}")

    return _run_subprocess(cmd, "rf2", job_id)


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
        update_job(job_id, status=JobStatus.FAILED, message="RFdiffusion failed")
        return

    rfdiff_qv = job_dir / "output" / "1_rfdiffusion.qv"

    # Step 2: ProteinMPNN
    update_job(job_id, step=StepName.PROTEINMPNN, progress="2/3")
    rc = run_proteinmpnn(job_id, rfdiff_qv, **proteinmpnn_kwargs)
    if rc != 0:
        update_job(job_id, status=JobStatus.FAILED, message="ProteinMPNN failed")
        return

    mpnn_qv = job_dir / "output" / "2_proteinmpnn.qv"

    # Step 3: RF2
    update_job(job_id, step=StepName.RF2, progress="3/3")
    rc = run_rf2(job_id, mpnn_qv, **rf2_kwargs)
    if rc != 0:
        update_job(job_id, status=JobStatus.FAILED, message="RF2 failed")
        return

    update_job(job_id, status=JobStatus.COMPLETED, message="Pipeline completed", progress="3/3")
