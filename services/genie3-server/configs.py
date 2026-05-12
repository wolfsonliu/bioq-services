"""Build the experiment YAML config that `genie3 generate -c <yaml>` consumes.

Each builder accepts the request body + job-local paths and returns a plain
dict (which the endpoint writes out as YAML before subprocess launch).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .models import BinderRequest, MotifRequest, UnconditionalRequest


def build_unconditional_config(
    *, rootdir: Path, req: UnconditionalRequest
) -> dict[str, Any]:
    return {
        "experiment": {"name": "unconditional"},
        "paths": {"rootdir": str(rootdir.resolve())},
        "generation": {
            "dataset": {
                "source": "unconditional",
                "min_length": req.min_length,
                "max_length": req.max_length,
                "length_step": req.length_step,
                "n_sample": req.n_sample,
                "batch_size": req.batch_size,
            },
            "sampler": {
                "sampler": {"direction_scale": req.direction_scale},
            },
        },
    }


def build_motif_config(
    *,
    rootdir: Path,
    dataset_root: Path,
    req: MotifRequest,
) -> dict[str, Any]:
    dataset: dict[str, Any] = {
        "source": "motif",
        "n_sample": req.n_sample,
        "batch_size": req.batch_size,
    }
    if req.selections:
        dataset["selections"] = req.selections
    return {
        "experiment": {"name": "motif"},
        "paths": {
            "rootdir": str(rootdir.resolve()),
            "dataset": str(dataset_root.resolve()),
        },
        "generation": {
            "dataset": dataset,
            "sampler": {"sampler": {"direction_scale": req.direction_scale}},
        },
    }


def build_binder_config(
    *,
    rootdir: Path,
    dataset_root: Path,
    req: BinderRequest,
) -> dict[str, Any]:
    dataset: dict[str, Any] = {
        "source": "target",
        "n_sample": req.n_sample,
        "batch_size": req.batch_size,
    }
    if req.selections:
        dataset["selections"] = req.selections
    return {
        "experiment": {"name": "binder"},
        "paths": {
            "rootdir": str(rootdir.resolve()),
            "dataset": str(dataset_root.resolve()),
        },
        "generation": {
            "dataset": dataset,
            "sampler": {"sampler": {"direction_scale": req.direction_scale}},
        },
    }


def rewrite_custom_paths(
    config: dict[str, Any],
    *,
    rootdir: Path,
    dataset_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Inject `paths.rootdir` (and optionally `paths.dataset`) into a user-supplied YAML.

    Used by `POST /api/generate` so clients only have to think about `generation`
    + `experiment` blocks; the framework owns where outputs land on the NAS.
    """
    paths = config.setdefault("paths", {})
    paths["rootdir"] = str(rootdir.resolve())
    if dataset_root is not None:
        paths["dataset"] = str(dataset_root.resolve())
    return config
