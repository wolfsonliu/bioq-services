"""CLI batch-mode entry point for genie3-server.

Usage::

    python -m server unconditional --n-sample 10 --min-length 100 --max-length 200 \\
        --output-dir /scratch/results/

    python -m server binder --dataset /data/targets.zip \\
        --output-dir /scratch/results/

Note: motif and binder endpoints require a dataset zip file (containing
problems/ + motifs/ or targets/ directories). The unconditional endpoint
requires no input files.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from bioagent_service.cli import CLIEndpoint, create_cli

from .adapter import Genie3Adapter
from .configs import build_binder_config, build_motif_config, build_unconditional_config
from .datasets import extract_dataset
from .models import BinderRequest, MotifRequest, UnconditionalRequest
from .settings import Genie3Settings

try:
    import yaml
except ImportError:
    import json as yaml  # type: ignore[no-redef]

settings = Genie3Settings()
adapter = Genie3Adapter(settings=settings)


def _write_yaml(config: dict, job_dir: Path) -> Path:
    path = job_dir / "input" / "experiment.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    import yaml as _yaml

    path.write_text(_yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _genie3_argv(config_path: Path, num_devices) -> list[str]:
    cmd = [settings.bin, "generate", "-c", str(config_path)]
    if num_devices is not None:
        cmd.extend(["--num-devices", str(num_devices)])
    return cmd


def _unconditional_build(req, inputs, job_dir, settings):
    config = build_unconditional_config(rootdir=job_dir / "output", req=req)
    config_path = _write_yaml(config, job_dir)
    return _genie3_argv(config_path, req.num_devices)


def _motif_build(req, inputs, job_dir, settings):
    dataset_root = extract_dataset(inputs["dataset"], job_dir / "input" / "dataset")
    config = build_motif_config(rootdir=job_dir / "output", dataset_root=dataset_root, req=req)
    config_path = _write_yaml(config, job_dir)
    return _genie3_argv(config_path, req.num_devices)


def _binder_build(req, inputs, job_dir, settings):
    dataset_root = extract_dataset(inputs["dataset"], job_dir / "input" / "dataset")
    config = build_binder_config(rootdir=job_dir / "output", dataset_root=dataset_root, req=req)
    config_path = _write_yaml(config, job_dir)
    return _genie3_argv(config_path, req.num_devices)


endpoints = {
    "unconditional": CLIEndpoint(
        name="unconditional",
        help="Unconditional protein backbone generation",
        request_model=UnconditionalRequest,
        build_argv=_unconditional_build,
    ),
    "motif": CLIEndpoint(
        name="motif",
        help="Motif scaffolding (dataset zip with problems/ + motifs/)",
        request_model=MotifRequest,
        build_argv=_motif_build,
        inputs={"dataset": ("Dataset zip file (problems/ + motifs/)", True)},
    ),
    "binder": CLIEndpoint(
        name="binder",
        help="Binder design (dataset zip with problems/ + targets/)",
        request_model=BinderRequest,
        build_argv=_binder_build,
        inputs={"dataset": ("Dataset zip file (problems/ + targets/)", True)},
    ),
}

create_cli(adapter, settings, endpoints, version="0.2.0")
