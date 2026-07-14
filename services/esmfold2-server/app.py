"""FastAPI app for esmfold2-server.

Exposes `/api/fold` for structure prediction. Job lifecycle endpoints
(`/healthz`, `/api/jobs/*`, `/api/manifest`, `/openapi.json`) come from
`bioagent_service.create_app`.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Optional

from bioagent_service import (
    JobInfo,
    attach_mcp,
    create_app,
    execute_task,
    model_form_depends,
    read_version_file,
    resolve_task_id,
)
from fastapi import Depends, File, Form, Header, HTTPException, Request, UploadFile

from .adapter import ESMFold2Adapter
from .models import FoldRequest
from .settings import ESMFold2Settings
from .tools import build_input_json, fold_argv
from bioagent_service.uris import resolve_uri, save_upload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = ESMFold2Settings()
adapter = ESMFold2Adapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="ESMFold2 Server",
    version=read_version_file(__file__, default="0.0.1"),
)


def _save_msa_uploads(
    msa_files: Optional[list[UploadFile]], input_dir: Path
) -> dict[str, Path]:
    """Save uploaded A3M files under `input/msa/`. Key by filename stem (= chain id)."""
    saved: dict[str, Path] = {}
    if not msa_files:
        return saved
    msa_dir = input_dir / "msa"
    msa_dir.mkdir(parents=True, exist_ok=True)
    for upload in msa_files:
        basename = Path(upload.filename or "").name
        if not basename:
            continue
        dest = msa_dir / basename
        save_upload(upload, dest)
        chain_id = dest.stem
        saved[chain_id] = dest
    return saved


def _extract_msa_zip(zip_uri: str, input_dir: Path) -> dict[str, Path]:
    """Resolve an a3m zip URI and extract `*.a3m` into `input/msa/`.

    Gateway-friendly alternative to multipart `msa_files`: the client uploads a
    zip of per-chain A3M files to OSS and passes its URI; the gateway rewrites
    `oss://` → the downstream mount. Keyed by filename stem (= chain id), same
    as `_save_msa_uploads`.
    """
    msa_dir = input_dir / "msa"
    msa_dir.mkdir(parents=True, exist_ok=True)
    zip_path = resolve_uri(zip_uri, input_dir / "msa.zip", settings)
    saved: dict[str, Path] = {}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                basename = Path(name).name
                if not basename.lower().endswith(".a3m"):
                    continue
                dest = msa_dir / basename
                with zf.open(name) as src, open(dest, "wb") as out:
                    out.write(src.read())
                saved[dest.stem] = dest
    except zipfile.BadZipFile as e:
        raise HTTPException(status_code=422, detail=f"Invalid MSA zip: {e}") from e
    return saved


def _resolve_msa(
    msa_files: Optional[list[UploadFile]], msa_zip_uri: Optional[str], input_dir: Path
) -> dict[str, Path]:
    """MSA from a multipart upload OR a zip URI (mutually exclusive; optional)."""
    if msa_files:
        return _save_msa_uploads(msa_files, input_dir)
    if msa_zip_uri:
        return _extract_msa_zip(msa_zip_uri, input_dir)
    return {}


@app.post("/api/fold", response_model=JobInfo)
def post_fold(
    params: FoldRequest = Depends(model_form_depends(FoldRequest)),
    msa_files: Optional[list[UploadFile]] = File(None),
) -> JobInfo:
    """Predict 3D structure of a biomolecular complex using ESMFold2."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        saved_msa = _save_msa_uploads(msa_files, input_dir)

        input_json = build_input_json(
            params,
            job_dir=job_dir,
            saved_msa_paths=saved_msa,
        )
        return fold_argv(
            params, job_dir=job_dir, input_json=input_json, settings=settings
        )

    return app.state.runner.submit(
        build_argv=_build,
        label="fold",
        input_params=params.model_dump(mode="json"),
    )


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/fold", response_model=JobInfo)
    def post_fold_task(
        request: Request,
        params: FoldRequest = Depends(model_form_depends(FoldRequest)),
        msa_files: Optional[list[UploadFile]] = File(None),
        msa_zip_uri: Optional[str] = Form(
            None,
            description="URI to a zip of per-chain A3M files (oss://, file://, "
                        "job://, http(s)://) as an alternative to multipart "
                        "msa_files — used by the gateway.",
        ),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Predict 3D structure as a single atomic task.

        Blocks until pipeline completion.  For submit/poll, use POST /api/fold.
        """
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        saved_state: dict[str, dict] = {"msa": {}}

        def _save(_req, input_dir: Path) -> None:
            saved_state["msa"] = _resolve_msa(msa_files, msa_zip_uri, input_dir)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            input_json = build_input_json(
                req, job_dir=job_dir, saved_msa_paths=saved_state["msa"]
            )
            return fold_argv(req, job_dir=job_dir, input_json=input_json, settings=settings)

        return execute_task(
            request,
            job_id=job_id,
            label="fold",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )


attach_mcp(app)
