"""argv builders for each of PPIFlow's five sampling scripts.

Each function takes a parsed request + the per-job input/output paths and
returns the list[str] that becomes the subprocess argv. The framework runs
this from `subprocess_cwd = settings.root` (i.e. PPIFlow's `tool/PPIFlow/`),
so script paths are relative to that dir.

Output filenames are documented in `manifest_extras` on the adapter.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from .models import (
    AntibodyRequest,
    BinderRequest,
    MonomerRequest,
    NanobodyRequest,
    ScaffoldingRequest,
)
from .settings import PPIFlowSettings

logger = logging.getLogger(__name__)


# --- Checkpoint resolution ---------------------------------------------------

_CKPT_FILES = {
    "binder": "binder.ckpt",
    "antibody": "antibody.ckpt",
    "nanobody": "nanobody.ckpt",
    "monomer": "monomer.ckpt",
}


def ckpt_path(settings: PPIFlowSettings, kind: str) -> Path:
    """Resolve one of the four checkpoint paths. Logs a warning (does NOT raise)
    when the file is missing — PPIFlow's own scripts will error with a clearer
    message if so, which is more useful diagnostically than a pre-flight raise."""
    p = settings.ckpt_dir / _CKPT_FILES[kind]
    if not p.exists():
        logger.warning("ppiflow checkpoint missing: %s — subprocess will fail", p)
    return p


def _default_config(settings: PPIFlowSettings, name: str) -> Path:
    """Resolve a default inference YAML shipped with upstream PPIFlow."""
    return settings.config_dir / name


# --- argv builders -----------------------------------------------------------


def binder_argv(
    req: BinderRequest, target_pdb: Path, job_dir: Path, settings: PPIFlowSettings,
) -> list[str]:
    """Construct `python sample_binder.py ...` argv."""
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable, "sample_binder.py",
        "--input_pdb", str(target_pdb.resolve()),
        "--target_chain", req.target_chain,
        "--binder_chain", req.binder_chain,
        "--config", str(_default_config(settings, "inference_binder.yaml")),
        "--sample_hotspot_rate_min", str(req.sample_hotspot_rate_min),
        "--sample_hotspot_rate_max", str(req.sample_hotspot_rate_max),
        "--samples_min_length", str(req.samples_min_length),
        "--samples_max_length", str(req.samples_max_length),
        "--samples_per_target", str(req.samples_per_target),
        "--model_weights", str(ckpt_path(settings, "binder")),
        "--output_dir", str(out_dir.resolve()),
        "--name", req.name,
    ]
    if req.specified_hotspots:
        argv.extend(["--specified_hotspots", req.specified_hotspots])
    return argv


def antibody_argv(
    req: AntibodyRequest,
    antigen_pdb: Path,
    framework_pdb: Path,
    job_dir: Path,
    settings: PPIFlowSettings,
) -> list[str]:
    """Construct `python sample_antibody_nanobody.py ...` argv with antibody.ckpt."""
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable, "sample_antibody_nanobody.py",
        "--antigen_pdb", str(antigen_pdb.resolve()),
        "--framework_pdb", str(framework_pdb.resolve()),
        "--antigen_chain", req.antigen_chain,
        "--heavy_chain", req.heavy_chain,
        "--light_chain", req.light_chain,
        "--cdr_length", req.cdr_length,
        "--samples_per_target", str(req.samples_per_target),
        "--config", str(_default_config(settings, "inference_nanobody.yaml")),
        "--model_weights", str(ckpt_path(settings, "antibody")),
        "--output_dir", str(out_dir.resolve()),
        "--name", req.name,
    ]
    if req.specified_hotspots:
        argv.extend(["--specified_hotspots", req.specified_hotspots])
    return argv


def nanobody_argv(
    req: NanobodyRequest,
    antigen_pdb: Path,
    framework_pdb: Path,
    job_dir: Path,
    settings: PPIFlowSettings,
) -> list[str]:
    """Construct `python sample_antibody_nanobody.py ...` argv with nanobody.ckpt.

    Nanobody has no light chain; we omit `--light_chain` so the upstream script
    uses its `default=None`.
    """
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable, "sample_antibody_nanobody.py",
        "--antigen_pdb", str(antigen_pdb.resolve()),
        "--framework_pdb", str(framework_pdb.resolve()),
        "--antigen_chain", req.antigen_chain,
        "--heavy_chain", req.heavy_chain,
        "--cdr_length", req.cdr_length,
        "--samples_per_target", str(req.samples_per_target),
        "--config", str(_default_config(settings, "inference_nanobody.yaml")),
        "--model_weights", str(ckpt_path(settings, "nanobody")),
        "--output_dir", str(out_dir.resolve()),
        "--name", req.name,
    ]
    if req.specified_hotspots:
        argv.extend(["--specified_hotspots", req.specified_hotspots])
    return argv


def monomer_argv(
    req: MonomerRequest, job_dir: Path, settings: PPIFlowSettings,
) -> list[str]:
    """Construct `python sample_monomer.py ...` argv in unconditional mode."""
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Upstream parses `--length_subset` as a JSON-encoded list (e.g. "[50, 100]").
    return [
        sys.executable, "sample_monomer.py",
        "--config", str(_default_config(settings, "inference_unconditional.yaml")),
        "--model_weights", str(ckpt_path(settings, "monomer")),
        "--output_dir", str((out_dir / req.name).resolve()),
        "--length_subset", json.dumps(req.length_subset),
        "--samples_num", str(req.samples_per_target),
    ]


def scaffolding_argv(
    req: ScaffoldingRequest, motif_csv: Path, job_dir: Path, settings: PPIFlowSettings,
) -> list[str]:
    """Construct `python sample_monomer.py ...` argv in motif-scaffolding mode."""
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        sys.executable, "sample_monomer.py",
        "--config", str(_default_config(settings, "inference_scaffolding.yaml")),
        "--model_weights", str(ckpt_path(settings, "monomer")),
        "--output_dir", str((out_dir / req.name).resolve()),
        "--motif_csv", str(motif_csv.resolve()),
        # Upstream parses `--motif_names` as JSON list, e.g. "['01_1LDB']".
        "--motif_names", json.dumps(req.motif_names),
        "--samples_num", str(req.samples_per_target),
    ]
