"""argv builders for dockq-server.

Two flows:
  * single: `score_argv(...)` → direct `DockQ <model> <native> --json out.json`
  * batch:  `batch_argv(...)` → invoke our `batch_score.py` driver, which loops
    over each model file and writes per-model JSONs + a sorted summary CSV.

The batch driver lives in this package (next to `tools.py`) so the venv can
import it without extra installation steps.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .models import ScoreBatchRequest, ScoreRequest
from .settings import DockQSettings


def _common_flags(req: ScoreRequest | ScoreBatchRequest, settings: DockQSettings) -> list[str]:
    """Shared DockQ CLI flags derived from `_DockQCommon` fields."""
    flags: list[str] = []
    n_cpu = req.n_cpu if req.n_cpu is not None else settings.default_n_cpu
    flags += ["--n_cpu", str(n_cpu)]
    if req.mapping:
        flags += ["--mapping", req.mapping]
    if req.small_molecule:
        flags.append("--small_molecule")
    if req.capri_peptide:
        flags.append("--capri_peptide")
    if req.no_align:
        flags.append("--no_align")
    if req.optDockQF1:
        flags.append("--optDockQF1")
    if req.allowed_mismatches:
        flags += ["--allowed_mismatches", str(req.allowed_mismatches)]
    return flags


def score_argv(
    req: ScoreRequest,
    *,
    job_dir: Path,
    model_path: Path,
    native_path: Path,
    settings: DockQSettings,
) -> list[str]:
    """Compose argv for a single-pair `DockQ <model> <native> --json output/<name>.json`."""
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / f"{req.name}.json"
    argv: list[str] = [
        settings.binary,
        str(model_path),
        str(native_path),
        "--json", str(json_out),
        "--short",  # compact stdout — full results land in --json file
        *_common_flags(req, settings),
    ]
    return argv


def batch_argv(
    req: ScoreBatchRequest,
    *,
    job_dir: Path,
    native_path: Path,
    models_dir: Path,
    settings: DockQSettings,
) -> list[str]:
    """Compose argv for the batch driver script.

    `models_dir` is a directory holding one .pdb / .cif file per candidate
    (basename = candidate identifier in the summary). The driver invokes the
    `DockQ` CLI once per model, parses each `--json` output, and writes:

      output/scores.csv         — sorted summary (one row per model)
      output/per_model/<m>.json — raw DockQ JSON per model
      output/failed.csv         — models that errored (basename + stderr tail)
    """
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    driver = Path(__file__).resolve().parent / "batch_score.py"
    argv: list[str] = [
        sys.executable, str(driver),
        "--native", str(native_path),
        "--models-dir", str(models_dir),
        "--output-dir", str(out_dir),
        "--dockq-bin", settings.binary,
        "--sort-by", req.sort_by,
        *_common_flags(req, settings),
    ]
    return argv
