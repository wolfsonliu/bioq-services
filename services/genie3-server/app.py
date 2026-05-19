"""FastAPI app for genie3-server.

Exposes four POST endpoints:

  * `/api/generate/unconditional` — no dataset, configurable length range
  * `/api/generate/motif`         — motif scaffolding (zip with `motifs/`)
  * `/api/generate/binder`        — binder design (zip with `targets/`)
  * `/api/generate`               — freeform YAML for advanced configs (iterative,
                                    custom `cond_strategy`, etc.)

Lifecycle (status / log / download / single-file / delete / manifest / OpenAPI)
is contributed by `bioagent_service.create_app`.

FC deployment:
  - 0.0.0.0:CAPort (default 9000)
  - /healthz must respond within 120 s of start
  - keep-alive >= 15 min so long-running generations don't get cut off
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any, Optional

import yaml
from bioagent_service import JobInfo, attach_mcp, create_app, model_form_depends, read_version_file
from fastapi import Depends, File, Form, HTTPException, UploadFile

from .adapter import Genie3Adapter
from .configs import (
    build_binder_config,
    build_motif_config,
    build_unconditional_config,
    rewrite_custom_paths,
)
from .datasets import extract_dataset
from .models import BinderRequest, MotifRequest, UnconditionalRequest
from .settings import Genie3Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = Genie3Settings()
adapter = Genie3Adapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="Genie3 Server",
    version=read_version_file(__file__, default="0.2.0"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_upload(upload: UploadFile, dest: Path) -> Path:
    """Stream UploadFile to disk in chunks (multi-GB datasets allowed)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return dest


def _write_yaml(config: dict[str, Any], job_dir: Path) -> Path:
    """Persist the experiment config so it's reproducible from the job dir alone."""
    path = job_dir / "input" / "experiment.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _genie3_argv(config_path: Path, num_devices: Optional[int]) -> list[str]:
    """`genie3 generate -c <yaml> [--num-devices N]`."""
    cmd = [settings.bin, "generate", "-c", str(config_path)]
    if num_devices is not None:
        cmd.extend(["--num-devices", str(num_devices)])
    return cmd


# ---------------------------------------------------------------------------
# Structured endpoints
# ---------------------------------------------------------------------------


@app.post("/api/generate/unconditional", response_model=JobInfo)
def generate_unconditional(
    params: UnconditionalRequest = Depends(model_form_depends(UnconditionalRequest)),
) -> JobInfo:
    """Unconditional protein backbone generation (no input structure)."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        config = build_unconditional_config(rootdir=job_dir / "output", req=params)
        config_path = _write_yaml(config, job_dir)
        return _genie3_argv(config_path, params.num_devices)

    return app.state.runner.submit(build_argv=_build, label="unconditional")


@app.post("/api/generate/motif", response_model=JobInfo)
def generate_motif(
    dataset: UploadFile = File(..., description="Zip with problems/ + motifs/."),
    params: MotifRequest = Depends(model_form_depends(MotifRequest)),
) -> JobInfo:
    """Motif scaffolding generation. Dataset zip must contain `problems/` + `motifs/`."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        zip_path = _save_upload(dataset, job_dir / "input" / "dataset.zip")
        try:
            dataset_root = extract_dataset(zip_path, job_dir / "input" / "dataset")
        except (zipfile.BadZipFile, ValueError) as e:
            raise HTTPException(status_code=422, detail=f"Invalid dataset zip: {e}") from e
        config = build_motif_config(
            rootdir=job_dir / "output", dataset_root=dataset_root, req=params,
        )
        config_path = _write_yaml(config, job_dir)
        return _genie3_argv(config_path, params.num_devices)

    return app.state.runner.submit(build_argv=_build, label="motif")


@app.post("/api/generate/binder", response_model=JobInfo)
def generate_binder(
    dataset: UploadFile = File(..., description="Zip with problems/ + targets/."),
    params: BinderRequest = Depends(model_form_depends(BinderRequest)),
) -> JobInfo:
    """Binder design generation. Dataset zip must contain `problems/` + `targets/`."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        zip_path = _save_upload(dataset, job_dir / "input" / "dataset.zip")
        try:
            dataset_root = extract_dataset(zip_path, job_dir / "input" / "dataset")
        except (zipfile.BadZipFile, ValueError) as e:
            raise HTTPException(status_code=422, detail=f"Invalid dataset zip: {e}") from e
        config = build_binder_config(
            rootdir=job_dir / "output", dataset_root=dataset_root, req=params,
        )
        config_path = _write_yaml(config, job_dir)
        return _genie3_argv(config_path, params.num_devices)

    return app.state.runner.submit(build_argv=_build, label="binder")


# ---------------------------------------------------------------------------
# Freeform YAML
# ---------------------------------------------------------------------------


@app.post("/api/generate", response_model=JobInfo)
def generate_custom(
    config_yaml: str = Form(..., description="Full experiment YAML as a string."),
    dataset: Optional[UploadFile] = File(
        None, description="Optional dataset zip; extracted to <job>/input/dataset/.",
    ),
    num_devices: Optional[int] = Form(None, description="Override GPU auto-detect."),
) -> JobInfo:
    """Run `genie3 generate` with a fully custom YAML config.

    `paths.rootdir` and (if a dataset zip is provided) `paths.dataset` are
    rewritten so the YAML you supply doesn't need to know its job_dir.
    """

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        try:
            user_config = yaml.safe_load(config_yaml)
        except yaml.YAMLError as e:
            raise HTTPException(status_code=422, detail=f"Invalid YAML: {e}") from e
        if not isinstance(user_config, dict):
            raise HTTPException(
                status_code=422, detail="config_yaml must be a mapping at the top level",
            )

        dataset_root: Optional[Path] = None
        if dataset is not None:
            zip_path = _save_upload(dataset, job_dir / "input" / "dataset.zip")
            try:
                dataset_root = extract_dataset(zip_path, job_dir / "input" / "dataset")
            except (zipfile.BadZipFile, ValueError) as e:
                raise HTTPException(status_code=422, detail=f"Invalid dataset zip: {e}") from e

        config = rewrite_custom_paths(
            user_config, rootdir=job_dir / "output", dataset_root=dataset_root,
        )
        config_path = _write_yaml(config, job_dir)
        return _genie3_argv(config_path, num_devices)

    return app.state.runner.submit(build_argv=_build, label="custom")


# Mount MCP server — must be AFTER all POST routes are registered so the
# auto-discovery walk sees the full surface.
attach_mcp(app)
