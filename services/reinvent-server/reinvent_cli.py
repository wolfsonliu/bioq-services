"""Shared exec wrapper for all reinvent-server run modes.

Stages uploaded inputs into work/, builds config.toml from params via
config_builder, runs `reinvent`, and copies the config + JSON echo to output/.
Invoked as `python -m server.reinvent_cli` by tools.py argv builders (HTTP + CLI).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config_builder import BUILDERS

# argv flag → params key that config_builder consumes (after staging into work/).
FILE_FLAGS = {
    "smiles_file": "smiles_file",
    "validation_smiles_file": "validation_smiles_file",
    "model_file": "model_file",
    "prior_file": "prior_file",
    "agent_file": "agent_file",
    "amino_acid_library": "amino_acid_library_file",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-type", required=True)
    p.add_argument("--params-json", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--prior-base", type=Path, required=True)
    p.add_argument("--reinvent-bin", type=Path, required=True)
    for flag in FILE_FLAGS:
        p.add_argument(f"--{flag.replace('_', '-')}", type=Path, default=None)
    return p.parse_args(argv)


def _resolve_device(device: str) -> str:
    """Fall back to CPU when a CUDA device is requested but unavailable.

    Keeps the service usable on CPU-only instances (REINVENT sampling/scoring run
    fine on CPU) instead of crashing; still uses the GPU when one is present.
    """
    if device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                print(
                    f"[reinvent_cli] device {device!r} requested but CUDA is "
                    f"unavailable; falling back to cpu",
                    file=sys.stderr,
                )
                return "cpu"
        except Exception as exc:  # torch import/probe failure → be safe, use cpu
            print(f"[reinvent_cli] CUDA probe failed ({exc!r}); using cpu",
                  file=sys.stderr)
            return "cpu"
    return device


def _stage(files: dict[str, Path], work: Path, params: dict) -> None:
    """Copy each provided input file into work/ and set params[<key>] to it."""
    for flag, path in files.items():
        if path is None:
            continue
        dst = work / Path(path).name
        if Path(path).resolve() != dst.resolve():
            shutil.copy2(path, dst)
        params[FILE_FLAGS[flag]] = str(dst)


def run(*, run_type: str, params_json: Path, work_dir: Path, output_dir: Path,
        device: str, prior_base: Path, reinvent_bin: Path,
        files: dict[str, Path]) -> int:
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = json.loads(Path(params_json).read_text())
    params["device"] = _resolve_device(device)
    _stage(files, work_dir, params)

    cfg = BUILDERS[run_type](params, output_dir, prior_base)
    config_path = work_dir / "config.toml"
    _dump_toml(cfg, config_path)

    env = dict(os.environ)
    env["REINVENT_PRIOR_BASE"] = str(prior_base)
    argv = [str(reinvent_bin), "-l", str(work_dir / "reinvent.log"), str(config_path)]
    # Inherit stdout/stderr so reinvent's output flows up to the framework's
    # job log (SubprocessRunner tees THIS process's streams to log_path). Do NOT
    # redirect to a file here — that swallowed the error and left failed jobs
    # undiagnosable via /api/jobs/<id>/log (error_tail was empty).
    r = subprocess.run(argv, cwd=work_dir, env=env)
    _collect(work_dir, output_dir, success=(r.returncode == 0))
    return r.returncode


def _dump_toml(cfg: dict, path: Path) -> None:
    import tomli_w
    with open(path, "wb") as f:
        tomli_w.dump(cfg, f)


def _collect(work_dir: Path, output_dir: Path, *, success: bool) -> None:
    """Copy audit artifacts to output/.

    config.toml + reinvent.log are copied on BOTH success and failure so a
    failed job is diagnosable via /api/jobs/<id>/files (reinvent's own -l log).
    The JSON config echo (_*.json) is only meaningful on success. Primary
    results already land in output/ (config uses absolute output paths).
    """
    for name in ("config.toml", "reinvent.log"):
        src = work_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
    if success:
        for js in work_dir.glob("_*.json"):
            shutil.copy2(js, output_dir / js.name)


def main():
    a = parse_args()
    files = {flag: getattr(a, flag) for flag in FILE_FLAGS}
    return run(
        run_type=a.run_type, params_json=a.params_json, work_dir=a.work_dir,
        output_dir=a.output_dir, device=a.device, prior_base=a.prior_base,
        reinvent_bin=a.reinvent_bin, files=files,
    )


if __name__ == "__main__":
    sys.exit(main())
