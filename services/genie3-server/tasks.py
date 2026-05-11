"""Background task execution for `genie3 generate`."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path
from typing import Optional

import yaml

from .models import JobInfo, JobStatus, TaskKind

logger = logging.getLogger(__name__)

# In Docker, RFANTIBODY_ROOT-style root for genie3.
_PROJECT_ROOT = Path(os.getenv("GENIE3_ROOT", "/opt/genie3"))
_GENIE3_BIN = os.getenv("GENIE3_BIN", "genie3")

JOBS_BASE_DIR = Path(os.getenv("GENIE3_JOBS_DIR", "/data/genie3_jobs"))


def _get_job_dir(job_id: str) -> Path:
    return JOBS_BASE_DIR / job_id


def _create_job_dir(job_id: str) -> Path:
    job_dir = _get_job_dir(job_id)
    (job_dir / "input").mkdir(parents=True, exist_ok=True)
    (job_dir / "output").mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir(parents=True, exist_ok=True)
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


def _load_jobs_from_disk():
    if not JOBS_BASE_DIR.exists():
        return
    for job_dir in sorted(JOBS_BASE_DIR.iterdir()):
        if not job_dir.is_dir():
            continue
        meta_path = job_dir / "job.json"
        if not meta_path.exists():
            continue
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            job = JobInfo.model_validate(data)
            if job.status == JobStatus.RUNNING:
                job.status = JobStatus.FAILED
                job.message = "Interrupted by container restart"
            _jobs[job.job_id] = job
            _persist_job(job.job_id)
        except Exception as e:
            logger.warning("Failed to restore job from %s: %s", meta_path, e)
    logger.info("Restored %d jobs from disk", len(_jobs))


_load_jobs_from_disk()


def create_job(task: TaskKind) -> str:
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = JobInfo(job_id=job_id, status=JobStatus.PENDING, task=task)
    _create_job_dir(job_id)
    _persist_job(job_id)
    return job_id


def get_job(job_id: str) -> Optional[JobInfo]:
    return _jobs.get(job_id)


def update_job(
    job_id: str,
    *,
    status: Optional[JobStatus] = None,
    message: Optional[str] = None,
    progress: Optional[str] = None,
):
    job = _jobs.get(job_id)
    if job is None:
        return
    if status is not None:
        job.status = status
    if message is not None:
        job.message = message
    if progress is not None:
        job.progress = progress
    _persist_job(job_id)


def list_output_files(job_id: str) -> list[str]:
    output_dir = _get_job_dir(job_id) / "output"
    if not output_dir.exists():
        return []
    files: list[str] = []
    for f in output_dir.rglob("*"):
        if f.is_file():
            files.append(str(f.relative_to(output_dir)))
    return sorted(files)


def cleanup_job(job_id: str):
    job_dir = _get_job_dir(job_id)
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    _jobs.pop(job_id, None)


# -- Dataset extraction & path normalization --

# Path keys in problem JSONs whose values reference files inside the dataset.
_PATH_KEYS_SCALAR = (
    "target_pdb_filepath",
    "target_fasta_filepath",
    "target_msa_filepath",
)
_PATH_KEYS_LIST = (
    "target_pdb_filepath_by_chain",
    "target_fasta_filepath_by_chain",
    "target_msa_filepath_by_chain",
    "motif_filepaths",
)


def _rewrite_problem_paths(problems_dir: Path, dataset_root: Path):
    """Rewrite filepath fields inside problem JSONs to absolute paths under dataset_root.

    Genie3 problem JSONs originally use cwd-relative paths like
    ``data/design/binder_design/binderbench/targets/pdb/01_bhrf1.pdb``. When the dataset is
    uploaded into a job-local directory, these paths must be rewritten so genie3 can find
    the files regardless of the working directory.

    The rewrite preserves the suffix from ``targets/`` or ``motifs/`` onwards and prefixes
    it with ``dataset_root``.
    """
    if not problems_dir.exists():
        return

    def _resolve(value: str) -> str:
        # Already absolute and exists → keep as-is.
        p = Path(value)
        if p.is_absolute() and p.exists():
            return str(p)

        # Try direct join with dataset_root.
        candidate = dataset_root / value
        if candidate.exists():
            return str(candidate.resolve())

        # Strip any prefix up to and including ``targets/`` or ``motifs/``.
        for marker in ("targets/", "motifs/"):
            idx = value.find(marker)
            if idx != -1:
                tail = value[idx:]
                candidate = dataset_root / tail
                if candidate.exists():
                    return str(candidate.resolve())

        # Last resort: try basename in well-known subdirs.
        basename = p.name
        for sub in ("targets/pdb", "targets/fasta", "targets/msa", "motifs"):
            candidate = dataset_root / sub / basename
            if candidate.exists():
                return str(candidate.resolve())

        logger.warning("Could not resolve dataset path %r under %s", value, dataset_root)
        return value

    for json_path in problems_dir.glob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Skipping unparseable JSON %s: %s", json_path, e)
            continue

        changed = False
        for key in _PATH_KEYS_SCALAR:
            if key in data and isinstance(data[key], str):
                new = _resolve(data[key])
                if new != data[key]:
                    data[key] = new
                    changed = True
        for key in _PATH_KEYS_LIST:
            if key in data and isinstance(data[key], list):
                new_list = [_resolve(v) if isinstance(v, str) else v for v in data[key]]
                if new_list != data[key]:
                    data[key] = new_list
                    changed = True
        if changed:
            json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def extract_dataset(zip_path: Path, dest_dir: Path) -> Path:
    """Extract a dataset zip and normalize problem JSON paths.

    The zip may contain either:
      - a single top-level directory (e.g. ``binderbench/{problems,targets}``)
      - or the contents directly (``problems/``, ``targets/``)

    Returns the resolved dataset root (the directory containing ``problems/``).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    # Find the dataset root: the directory containing problems/.
    candidates = [p.parent for p in dest_dir.rglob("problems") if p.is_dir()]
    if not candidates:
        raise ValueError(f"Dataset zip does not contain a 'problems/' directory: {zip_path.name}")
    dataset_root = sorted(candidates, key=lambda p: len(p.parts))[0]

    _rewrite_problem_paths(dataset_root / "problems", dataset_root)
    return dataset_root


# -- Config builders --


def build_unconditional_config(
    *,
    rootdir: Path,
    min_length: int,
    max_length: int,
    length_step: int,
    n_sample: int,
    direction_scale: float,
    batch_size: int,
) -> dict:
    return {
        "experiment": {"name": "unconditional"},
        "paths": {"rootdir": str(rootdir.resolve())},
        "generation": {
            "dataset": {
                "source": "unconditional",
                "min_length": min_length,
                "max_length": max_length,
                "length_step": length_step,
                "n_sample": n_sample,
                "batch_size": batch_size,
            },
            "sampler": {
                "sampler": {"direction_scale": direction_scale},
            },
        },
    }


def build_motif_config(
    *,
    rootdir: Path,
    dataset_root: Path,
    selections: Optional[str],
    n_sample: int,
    direction_scale: float,
    batch_size: int,
) -> dict:
    dataset: dict = {
        "source": "motif",
        "n_sample": n_sample,
        "batch_size": batch_size,
    }
    if selections:
        dataset["selections"] = selections
    return {
        "experiment": {"name": "motif"},
        "paths": {
            "rootdir": str(rootdir.resolve()),
            "dataset": str(dataset_root.resolve()),
        },
        "generation": {
            "dataset": dataset,
            "sampler": {"sampler": {"direction_scale": direction_scale}},
        },
    }


def build_binder_config(
    *,
    rootdir: Path,
    dataset_root: Path,
    selections: Optional[str],
    n_sample: int,
    direction_scale: float,
    batch_size: int,
) -> dict:
    dataset: dict = {
        "source": "target",
        "n_sample": n_sample,
        "batch_size": batch_size,
    }
    if selections:
        dataset["selections"] = selections
    return {
        "experiment": {"name": "binder"},
        "paths": {
            "rootdir": str(rootdir.resolve()),
            "dataset": str(dataset_root.resolve()),
        },
        "generation": {
            "dataset": dataset,
            "sampler": {"sampler": {"direction_scale": direction_scale}},
        },
    }


# -- genie3 generate runner --

_GENERATE_TIMEOUT = int(os.getenv("GENIE3_TIMEOUT", str(86400)))


def _run_generate(job_id: str, config_path: Path, num_devices: Optional[int] = None) -> int:
    log_path = _get_job_dir(job_id) / "logs" / "generate.log"

    cmd = [_GENIE3_BIN, "generate", "-c", str(config_path)]
    if num_devices is not None:
        cmd.extend(["--num-devices", str(num_devices)])

    logger.info("[%s] Running: %s", job_id, " ".join(cmd))
    try:
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                cmd,
                cwd=str(_PROJECT_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=_GENERATE_TIMEOUT,
            )
    except subprocess.TimeoutExpired:
        logger.error("[%s] genie3 generate timed out after %ds", job_id, _GENERATE_TIMEOUT)
        return -1

    if result.returncode != 0:
        logger.error(
            "[%s] genie3 generate failed (rc=%d). log tail:\n%s",
            job_id,
            result.returncode,
            _read_log_tail(log_path, 2000),
        )
    else:
        logger.info("[%s] genie3 generate completed", job_id)
    return result.returncode


def _read_log_tail(path: Path, chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-chars:] if len(text) > chars else text
    except OSError:
        return "(log file unreadable)"


def write_config(job_id: str, config: dict) -> Path:
    job_dir = _get_job_dir(job_id)
    config_path = job_dir / "input" / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def run_generate(job_id: str, config: dict, num_devices: Optional[int] = None) -> int:
    """Persist the config to disk and run `genie3 generate`."""
    config_path = write_config(job_id, config)
    return _run_generate(job_id, config_path, num_devices=num_devices)


def has_outputs(job_id: str) -> bool:
    """Return True if at least one PDB was generated."""
    output_dir = _get_job_dir(job_id) / "output"
    if not output_dir.exists():
        return False
    return any(output_dir.rglob("*.pdb"))
