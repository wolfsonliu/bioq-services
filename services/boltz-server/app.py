"""FastAPI app for boltz-server.

Exposes `/api/predict_structure` and `/api/predict_affinity`. Job lifecycle
endpoints (`/healthz`, `/api/jobs/*`, `/api/manifest`, `/openapi.json`) come
from `bioagent_service.create_app`.
"""

from __future__ import annotations

import logging
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
from fastapi import Depends, File, Header, Request, UploadFile

from .adapter import BoltzAdapter
from .models import PredictAffinityRequest, PredictStructureRequest
from .settings import BoltzSettings
from .tools import build_yaml, predict_argv
from bioagent_service.uris import resolve_uri, save_upload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = BoltzSettings()
adapter = BoltzAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="Boltz Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# Remove framework's generic /healthz/detail so our boltz-specific weights
# probe takes over. FastAPI uses first-match routing and >=0.115 wraps
# included routers in `_IncludedRouter`; descend into them to find the
# framework route.
def _strip_route(router, path: str, method: str) -> None:
    router.routes = [
        r for r in router.routes
        if not (getattr(r, "path", None) == path
                and method in getattr(r, "methods", set()))
    ]
    for r in router.routes:
        inner = getattr(r, "original_router", None)
        if inner is not None:
            _strip_route(inner, path, method)


_strip_route(app.router, "/healthz/detail", "GET")


@app.get("/healthz/detail")
def healthz_detail(request: Request) -> dict:
    """Extended health: report whether NAS-mounted weights are reachable.

    Boltz-2 weights live on NAS at `BOLTZ_CACHE_DIR` (default
    `/data/models/boltz/`).  We probe the 3 paths boltz expects;
    `weights_loaded=false` lets the agent surface a misconfigured FC mount
    / unbound SIF without crashing the service.
    """
    expected = {
        "boltz2_conf.ckpt": settings.cache_dir / "boltz2_conf.ckpt",
        "boltz2_aff.ckpt": settings.cache_dir / "boltz2_aff.ckpt",
        "mols": settings.cache_dir / "mols",
    }
    missing = {k: str(p) for k, p in expected.items() if not p.exists()}
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "weights_dir": str(settings.cache_dir),
        "weights_loaded": not missing,
        "weights_missing": missing,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


def _save_msa_uploads(
    msa_files: Optional[list[UploadFile]], input_dir: Path
) -> dict[str, Path]:
    """Save uploaded a3m files under `input/msa/`. Key by filename stem (= chain id)."""
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
        chain_id = dest.stem  # "<chain>.a3m" → "<chain>"
        saved[chain_id] = dest
    return saved


def _save_template_uploads(
    template_files: Optional[list[UploadFile]], input_dir: Path
) -> dict[str, Path]:
    """Save uploaded CIF/PDB templates under `input/templates/`. Key by basename."""
    saved: dict[str, Path] = {}
    if not template_files:
        return saved
    tmpl_dir = input_dir / "templates"
    tmpl_dir.mkdir(parents=True, exist_ok=True)
    for upload in template_files:
        basename = Path(upload.filename or "").name
        if not basename:
            continue
        dest = tmpl_dir / basename
        save_upload(upload, dest)
        saved[basename] = dest
    return saved


def _resolve_per_chain_msa_uris(
    req: PredictStructureRequest | PredictAffinityRequest,
    input_dir: Path,
    saved_msa_paths: dict[str, Path],
) -> dict[str, Path]:
    """Resolve any SequenceEntry.msa_uri that points to a URI scheme.

    Multipart uploads were already saved via `_save_msa_uploads`; this fills in
    the remaining `job://`, `file://`, `oss://`, `http(s)://` cases. Modifies
    `saved_msa_paths` in place and returns it (for chaining).
    """
    msa_dir = input_dir / "msa"
    for entry in req.sequences:
        if entry.type != "protein" or not entry.msa_uri or entry.msa_uri == "empty":
            continue
        chain_id = entry.id if isinstance(entry.id, str) else entry.id[0]
        if chain_id in saved_msa_paths:
            continue  # already saved via multipart
        # Only resolve true URIs; bare filenames are treated as multipart misses
        # (the validator path already rejects msa_mode=provided without a3m).
        if entry.msa_uri.startswith(("job://", "file://", "oss://", "http://", "https://", "/")):
            msa_dir.mkdir(parents=True, exist_ok=True)
            dest = msa_dir / f"{chain_id}.a3m"
            resolve_uri(entry.msa_uri, dest, settings)
            saved_msa_paths[chain_id] = dest
    return saved_msa_paths


def _resolve_template_uris(
    req: PredictStructureRequest | PredictAffinityRequest,
    input_dir: Path,
    saved_template_paths: dict[str, Path],
) -> dict[str, Path]:
    """Resolve TemplateEntry URIs to local files under `input/templates/`."""
    tmpl_dir = input_dir / "templates"
    for i, t in enumerate(req.templates):
        uri = t.cif_uri or t.pdb_uri
        assert uri is not None
        if uri in saved_template_paths:
            continue
        if uri.startswith(("job://", "file://", "oss://", "http://", "https://", "/")):
            tmpl_dir.mkdir(parents=True, exist_ok=True)
            ext = "cif" if t.cif_uri else "pdb"
            dest = tmpl_dir / f"template_{i:03d}.{ext}"
            resolve_uri(uri, dest, settings)
            saved_template_paths[uri] = dest
    return saved_template_paths


@app.post("/api/predict_structure", response_model=JobInfo)
def post_predict_structure(
    params: PredictStructureRequest = Depends(
        model_form_depends(PredictStructureRequest)
    ),
    msa_files: Optional[list[UploadFile]] = File(None),
    template_files: Optional[list[UploadFile]] = File(None),
    raw_yaml_upload: Optional[UploadFile] = File(None),
) -> JobInfo:
    """Predict the 3D structure of a biomolecular complex (no affinity)."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Optional: raw YAML uploaded as a file alongside other params.
        if raw_yaml_upload is not None:
            save_upload(raw_yaml_upload, input_dir / "input.yaml")
            # build_yaml will skip rendering when raw_yaml is set; mirror that
            # here by re-using its validator and short-circuiting.
            params.raw_yaml = (input_dir / "input.yaml").read_text(encoding="utf-8")

        saved_msa = _save_msa_uploads(msa_files, input_dir)
        _resolve_per_chain_msa_uris(params, input_dir, saved_msa)

        saved_tmpl = _save_template_uploads(template_files, input_dir)
        _resolve_template_uris(params, input_dir, saved_tmpl)

        yaml_path = build_yaml(
            params,
            job_dir=job_dir,
            settings=settings,
            saved_msa_paths=saved_msa,
            saved_template_paths=saved_tmpl,
        )
        return predict_argv(
            params, job_dir=job_dir, yaml_path=yaml_path, settings=settings
        )

    return app.state.runner.submit(
        build_argv=_build, label="predict_structure",
        input_params=params.model_dump(mode="json"),
    )


@app.post("/api/predict_affinity", response_model=JobInfo)
def post_predict_affinity(
    params: PredictAffinityRequest = Depends(
        model_form_depends(PredictAffinityRequest)
    ),
    msa_files: Optional[list[UploadFile]] = File(None),
    template_files: Optional[list[UploadFile]] = File(None),
    raw_yaml_upload: Optional[UploadFile] = File(None),
) -> JobInfo:
    """Predict structure + ligand binding affinity for one ligand chain."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        if raw_yaml_upload is not None:
            save_upload(raw_yaml_upload, input_dir / "input.yaml")
            params.raw_yaml = (input_dir / "input.yaml").read_text(encoding="utf-8")

        saved_msa = _save_msa_uploads(msa_files, input_dir)
        _resolve_per_chain_msa_uris(params, input_dir, saved_msa)

        saved_tmpl = _save_template_uploads(template_files, input_dir)
        _resolve_template_uris(params, input_dir, saved_tmpl)

        yaml_path = build_yaml(
            params,
            job_dir=job_dir,
            settings=settings,
            saved_msa_paths=saved_msa,
            saved_template_paths=saved_tmpl,
        )
        return predict_argv(
            params, job_dir=job_dir, yaml_path=yaml_path, settings=settings
        )

    return app.state.runner.submit(
        build_argv=_build, label="predict_affinity",
        input_params=params.model_dump(mode="json"),
    )


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/predict_structure", response_model=JobInfo)
    def post_predict_structure_task(
        request: Request,
        params: PredictStructureRequest = Depends(model_form_depends(PredictStructureRequest)),
        msa_files: Optional[list[UploadFile]] = File(None),
        template_files: Optional[list[UploadFile]] = File(None),
        raw_yaml_upload: Optional[UploadFile] = File(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Predict structure as a single atomic task (blocks until completion).

        Designed for FC Async Task Mode (X-Fc-Invocation-Type: Async).  For the
        submit/poll interface, use POST /api/predict_structure instead.
        """
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        saved_state: dict[str, dict] = {"msa": {}, "tmpl": {}}

        def _save(_req, input_dir: Path) -> None:
            if raw_yaml_upload is not None:
                save_upload(raw_yaml_upload, input_dir / "input.yaml")
                params.raw_yaml = (input_dir / "input.yaml").read_text(encoding="utf-8")
            saved_state["msa"] = _save_msa_uploads(msa_files, input_dir)
            _resolve_per_chain_msa_uris(params, input_dir, saved_state["msa"])
            saved_state["tmpl"] = _save_template_uploads(template_files, input_dir)
            _resolve_template_uris(params, input_dir, saved_state["tmpl"])

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            yaml_path = build_yaml(
                req, job_dir=job_dir, settings=settings,
                saved_msa_paths=saved_state["msa"],
                saved_template_paths=saved_state["tmpl"],
            )
            return predict_argv(req, job_dir=job_dir, yaml_path=yaml_path, settings=settings)

        return execute_task(
            request,
            job_id=job_id,
            label="predict_structure",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )

    @app.post("/api/tasks/predict_affinity", response_model=JobInfo)
    def post_predict_affinity_task(
        request: Request,
        params: PredictAffinityRequest = Depends(model_form_depends(PredictAffinityRequest)),
        msa_files: Optional[list[UploadFile]] = File(None),
        template_files: Optional[list[UploadFile]] = File(None),
        raw_yaml_upload: Optional[UploadFile] = File(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Predict structure + ligand affinity as a single atomic task."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        saved_state: dict[str, dict] = {"msa": {}, "tmpl": {}}

        def _save(_req, input_dir: Path) -> None:
            if raw_yaml_upload is not None:
                save_upload(raw_yaml_upload, input_dir / "input.yaml")
                params.raw_yaml = (input_dir / "input.yaml").read_text(encoding="utf-8")
            saved_state["msa"] = _save_msa_uploads(msa_files, input_dir)
            _resolve_per_chain_msa_uris(params, input_dir, saved_state["msa"])
            saved_state["tmpl"] = _save_template_uploads(template_files, input_dir)
            _resolve_template_uris(params, input_dir, saved_state["tmpl"])

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            yaml_path = build_yaml(
                req, job_dir=job_dir, settings=settings,
                saved_msa_paths=saved_state["msa"],
                saved_template_paths=saved_state["tmpl"],
            )
            return predict_argv(req, job_dir=job_dir, yaml_path=yaml_path, settings=settings)

        return execute_task(
            request,
            job_id=job_id,
            label="predict_affinity",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )


# Mount MCP server — must be AFTER all POST routes are registered so the
# auto-discovery walk sees the full surface.
attach_mcp(app)
