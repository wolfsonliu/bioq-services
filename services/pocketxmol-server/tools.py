"""Argv assembly for pocketxmol-server.

Upstream ``scripts/sample_use.py`` and ``scripts/believe_use_pdb.py`` are
CLI scripts that take a task config YAML + a model config YAML.  Each
build_*_argv function assembles the subprocess command line for a given
endpoint after the YAML(s) have been written to job_dir/input/.

Argv shape:
    <python> <sample_script> \
        --config_task  <task_config.yml>   \
        --config_model <model_config.yml>  \
        --outdir       <output_dir>        \
        --device       cuda:0              \
       [--batch_size   <n>]
"""

from __future__ import annotations

from pathlib import Path

from .models import ConfidenceRequest
from .settings import PocketXMolSettings


def sample_argv(
    *,
    task_config_path: Path,
    model_config_path: Path,
    output_dir: Path,
    settings: PocketXMolSettings,
    batch_size: int | None = None,
    device: str = "cuda:0",
) -> list[str]:
    """Argv for /api/dock, /api/sbdd, /api/linking, /api/optimize, /api/pepdesign.

    All five generation endpoints call the same upstream script; task
    routing happens inside the YAML (task.name, task.transform.name).
    """
    argv = [
        settings.python,
        str(settings.sample_script),
        "--config_task", str(task_config_path),
        "--config_model", str(model_config_path),
        "--outdir", str(output_dir),
        "--device", device,
    ]
    if batch_size is not None:
        argv.extend(["--batch_size", str(batch_size)])
    return argv


def confidence_argv(
    *,
    req: ConfidenceRequest,
    source_output_dir: Path,
    confidence_yaml_path: Path,
    settings: PocketXMolSettings,
    device: str = "cuda:0",
) -> list[str]:
    """Argv for /api/confidence.

    Upstream ``believe_use_pdb.py`` expects ``--result_root`` to be the
    parent dir and ``--exp_name`` to be a substring matching a
    ``{exp_name}_{timestamp}`` sub-directory inside it.

    Our layout after a generation job:
        <source_job>/output/<exp_name>_<timestamp>/...

    So --result_root=<source_job>/output and --exp_name is any distinguishing
    substring (we pass the whole subdir name to be exact).
    """
    # Find the single generated sub-directory.  `source_output_dir` is
    # our stable /output/ mount; the timestamped subdir lives directly
    # under it.  The endpoint is responsible for resolving this before
    # calling us (we take it as a Path pointing to the timestamped dir).
    exp_dir = source_output_dir
    return [
        settings.python,
        str(settings.confidence_script),
        "--exp_name", exp_dir.name,
        "--result_root", str(exp_dir.parent),
        "--config", str(confidence_yaml_path),
        "--device", device,
    ]


__all__ = ["sample_argv", "confidence_argv"]
