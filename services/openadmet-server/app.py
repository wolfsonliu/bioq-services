"""FastAPI app for openadmet-server.

Exposes:

* ``POST /api/predict``          — submit-poll predict job
* ``POST /api/tasks/predict``    — FC async task variant
* ``POST /api/compare``          — submit-poll compare job
* ``POST /api/tasks/compare``    — FC async task variant
* ``GET  /api/models``           — enumerate NAS-registered model_dirs

Job lifecycle (`/api/jobs/*`), manifest, healthz, OpenAPI come from
`bioagent_service.create_app`.

Design doc: engineering/decisions/2026-07-05-openadmet-server-design.md
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
from fastapi import Depends, File, Header, HTTPException, Request, UploadFile

from .adapter import OpenAdmetAdapter
from .models import CompareRequest, PredictRequest
from .settings import OpenAdmetSettings
from .tools import (
    archive_request,
    augment_csv_with_aliases,
    build_predict_shell,
    compare_argv_mode_a,
    compare_argv_mode_b,
    predict_composite_argv,
    sniff_smiles_column,
    split_inline_smiles,
    write_alias_csv,
)
from .uris import resolve_input, save_upload

logger = logging.getLogger(__name__)

settings = OpenAdmetSettings()
adapter = OpenAdmetAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="OpenADMET Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---------------------------------------------------------------------------
# /healthz/detail override — probe NAS weights + CheMeleon foundation.
# ---------------------------------------------------------------------------


def _strip_route(router, path: str, method: str) -> None:
    router.routes = [
        r for r in router.routes
        if not (
            getattr(r, "path", None) == path
            and method in getattr(r, "methods", set())
        )
    ]
    for r in router.routes:
        inner = getattr(r, "original_router", None)
        if inner is not None:
            _strip_route(inner, path, method)


_strip_route(app.router, "/healthz/detail", "GET")


@app.get("/healthz/detail")
def healthz_detail(request: Request) -> dict:
    """Probe weights_dir mount + CheMeleon foundation + registered models."""
    models = settings.list_models()
    expected = {
        "weights_dir": settings.weights_dir,
        "models_root": settings.models_root,
        "chemeleon_foundation": settings.chemeleon_foundation,
    }
    missing = {k: str(p) for k, p in expected.items() if not p.exists()}
    foundation_present = settings.chemeleon_foundation.exists()

    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "weights_dir": str(settings.weights_dir),
        # Service usable iff all mounts present AND ≥ 1 model registered AND
        # chemeleon foundation cached (all 6 pre-staged models depend on it).
        "weights_loaded": (not missing) and len(models) > 0 and foundation_present,
        "weights_missing": missing,
        "models_available": [m.name for m in models],
        "models_count": len(models),
        "chemeleon_foundation_present": foundation_present,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


# ---------------------------------------------------------------------------
# GET /api/models — enumerate the NAS model registry
# ---------------------------------------------------------------------------


@app.get("/api/models")
def list_models() -> dict:
    """List NAS-registered anvil model_dirs with their metadata."""
    models = settings.list_models()
    return {
        "models_root": str(settings.models_root),
        "count": len(models),
        "models": [m.to_dict() for m in models],
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _validate_model_names(names: list[str]) -> list:
    """Validate every model_name exists on NAS; return the ModelInfo list.

    Raises 422 for the first missing name.
    """
    available = {m.name: m for m in settings.list_models()}
    if not available:
        raise HTTPException(
            status_code=503,
            detail=(
                "No models registered on NAS. Pre-stage at least one via "
                "scripts/fetch_weights.sh then GET /api/models to verify."
            ),
        )
    resolved = []
    for name in names:
        if name not in available:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Model '{name}' not registered. Available: "
                    f"{sorted(available)}"
                ),
            )
        resolved.append(available[name])
    return resolved


def _prepare_predict_input(
    req: PredictRequest,
    input_csv: Optional[UploadFile],
    input_sdf: Optional[UploadFile],
    job_dir: Path,
    settings: OpenAdmetSettings,
) -> Path:
    """Land the user's SMILES source at ``<job_dir>/input/input.csv``.

    Priorities: (1) inline smiles → write alias CSV; (2) CSV upload/URI →
    augment with alias cols; (3) SDF upload/URI → keep SDF path (upstream
    reads .sdf directly).

    Returns the absolute path to feed ``--input-path`` (either ``.csv`` or ``.sdf``).
    """
    in_dir = job_dir / "input"
    in_dir.mkdir(parents=True, exist_ok=True)

    provided = [
        bool(req.input_smiles),
        input_csv is not None,
        bool(req.input_csv_uri),
        input_sdf is not None,
        bool(req.input_sdf_uri),
    ]
    if sum(provided) == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "Provide exactly one of: `input_smiles`, `input_csv`, "
                "`input_csv_uri`, `input_sdf`, `input_sdf_uri`"
            ),
        )
    if sum(provided) > 1:
        raise HTTPException(
            status_code=422,
            detail="Multiple input sources supplied; pass exactly one.",
        )

    if req.input_smiles:
        try:
            smiles = split_inline_smiles(req.input_smiles, max_n=200)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return write_alias_csv(
            smiles, in_dir / "input.csv", settings.default_input_col_aliases,
        )

    if input_csv is not None or req.input_csv_uri:
        # Land the raw CSV first, then augment with alias columns.
        raw = resolve_input(input_csv, req.input_csv_uri, in_dir / "raw.csv", settings)
        detected = sniff_smiles_column(raw, settings.default_input_col_aliases)
        if detected is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Uploaded CSV has no recognized SMILES column. "
                    f"Expected one of: {settings.default_input_col_aliases}. "
                    f"Set `input_col` explicitly to override."
                ),
            )
        return augment_csv_with_aliases(
            raw, in_dir / "input.csv", settings.default_input_col_aliases, detected,
        )

    # SDF path — upstream reads .sdf natively. No alias trick applies because
    # SDF property blocks carry columns as key-value pairs (not headers).
    return resolve_input(input_sdf, req.input_sdf_uri, in_dir / "input.sdf", settings)


# ---------------------------------------------------------------------------
# POST /api/predict
# ---------------------------------------------------------------------------


@app.post("/api/predict", response_model=JobInfo)
def predict(
    params: PredictRequest = Depends(model_form_depends(PredictRequest)),
    input_csv: Optional[UploadFile] = File(None),
    input_sdf: Optional[UploadFile] = File(None),
) -> JobInfo:
    """Run one or more ADMET predictions against pre-registered models.

    Multiple `model_names` may share the same input_col (single subprocess)
    or differ (split into per-input-col groups; results merged into
    output/predictions.csv).
    """
    resolved_models = _validate_model_names(params.model_names)

    def _build(job_id: str, job_dir: Path) -> list[str]:
        input_path = _prepare_predict_input(
            params, input_csv, input_sdf, job_dir, settings,
        )
        archive_request(job_dir, "predict_request", params.model_dump(mode="json"))
        argvs = predict_composite_argv(
            params,
            input_csv=input_path,
            job_dir=job_dir,
            settings=settings,
            models=resolved_models,
        )
        return build_predict_shell(argvs, output_dir=job_dir / "output")

    return app.state.runner.submit(
        build_argv=_build,
        label="predict",
        input_params=params.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# POST /api/tasks/predict — FC async task mode variant
# ---------------------------------------------------------------------------


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/predict", response_model=JobInfo)
    def predict_task(
        request: Request,
        params: PredictRequest = Depends(model_form_depends(PredictRequest)),
        input_csv: Optional[UploadFile] = File(None),
        input_sdf: Optional[UploadFile] = File(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Predict as a single atomic task (blocks until pipeline completion)."""
        resolved_models = _validate_model_names(params.model_names)
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        state: dict = {}

        def _save(_req: PredictRequest, input_dir: Path) -> None:
            # Land CSV/SDF into <job_dir>/input/, memoize the resolved path.
            job_dir = input_dir.parent
            state["input_path"] = _prepare_predict_input(
                params, input_csv, input_sdf, job_dir, settings,
            )
            archive_request(job_dir, "predict_request", params.model_dump(mode="json"))

        def _build(_req: PredictRequest, _job_id: str, job_dir: Path) -> list[str]:
            argvs = predict_composite_argv(
                params,
                input_csv=state["input_path"],
                job_dir=job_dir,
                settings=settings,
                models=resolved_models,
            )
            return build_predict_shell(argvs, output_dir=job_dir / "output")

        return execute_task(
            request,
            job_id=job_id,
            label="predict",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )


# ---------------------------------------------------------------------------
# POST /api/compare
# ---------------------------------------------------------------------------


def _prepare_compare_inputs(
    req: CompareRequest,
    stats_files: list[UploadFile],
    job_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """Return (model_dirs, stats_file_paths) — exactly one is non-empty per request."""
    in_dir = job_dir / "input"
    in_dir.mkdir(parents=True, exist_ok=True)

    if req.model_names:
        model_infos = _validate_model_names(req.model_names)
        return [m.path for m in model_infos], []

    # Mode B — persist stats JSON uploads.
    if not stats_files:
        raise HTTPException(
            status_code=422,
            detail="compare Mode B: `model_stats_files` upload is required.",
        )
    if len(stats_files) != len(req.labels):
        raise HTTPException(
            status_code=422,
            detail=(
                f"compare Mode B: `model_stats_files` count "
                f"({len(stats_files)}) must equal `labels` count ({len(req.labels)})."
            ),
        )
    stats_paths: list[Path] = []
    for i, uf in enumerate(stats_files):
        dest = in_dir / f"stats_{i}_{Path(uf.filename or f'file{i}').name}"
        save_upload(uf, dest)
        stats_paths.append(dest)
    return [], stats_paths


@app.post("/api/compare", response_model=JobInfo)
def compare(
    params: CompareRequest = Depends(model_form_depends(CompareRequest)),
    model_stats_files: list[UploadFile] = File(default=[]),
) -> JobInfo:
    """Post-hoc comparison of pre-trained models (Mode A) or their stats JSON (Mode B)."""

    def _build(job_id: str, job_dir: Path) -> list[str]:
        archive_request(job_dir, "compare_request", params.model_dump(mode="json"))
        model_dirs, stats_paths = _prepare_compare_inputs(
            params, model_stats_files, job_dir,
        )
        output_dir = job_dir / "output"
        if model_dirs:
            return compare_argv_mode_a(
                params, output_dir=output_dir, model_dirs=model_dirs, settings=settings,
            )
        return compare_argv_mode_b(
            params, output_dir=output_dir, stats_files=stats_paths, settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build,
        label="compare",
        input_params=params.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# POST /api/tasks/compare — FC async task variant
# ---------------------------------------------------------------------------


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/compare", response_model=JobInfo)
    def compare_task(
        request: Request,
        params: CompareRequest = Depends(model_form_depends(CompareRequest)),
        model_stats_files: list[UploadFile] = File(default=[]),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        state: dict = {}

        def _save(_req: CompareRequest, input_dir: Path) -> None:
            job_dir = input_dir.parent
            archive_request(job_dir, "compare_request", params.model_dump(mode="json"))
            state["model_dirs"], state["stats_paths"] = _prepare_compare_inputs(
                params, model_stats_files, job_dir,
            )

        def _build(_req: CompareRequest, _job_id: str, job_dir: Path) -> list[str]:
            output_dir = job_dir / "output"
            if state["model_dirs"]:
                return compare_argv_mode_a(
                    params,
                    output_dir=output_dir,
                    model_dirs=state["model_dirs"],
                    settings=settings,
                )
            return compare_argv_mode_b(
                params,
                output_dir=output_dir,
                stats_files=state["stats_paths"],
                settings=settings,
            )

        return execute_task(
            request,
            job_id=job_id,
            label="compare",
            params=params,
            build_argv=_build,
            save_inputs=_save,
        )


# Must be after all POST routes so MCP auto-discovery sees the full surface.
attach_mcp(app)
