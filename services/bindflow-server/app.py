"""FastAPI app for bindflow-server.

Exposes `/api/calculate/fep` + `/api/calculate/mmpbsa` (both submit/poll).
Task endpoints (`/api/tasks/*`) are NOT registered — BindFlow workflows
routinely exceed FC's 24 h ceiling; the service is HPC-primary and never
deployed to FC (design doc §2.2, §3.3).  See `settings.task_endpoints_enabled`.

Job lifecycle endpoints (/healthz, /api/jobs/*, /api/manifest, /openapi.json)
come from `bioq_service.create_app`.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from bioq_service import (
    JobInfo,
    attach_mcp,
    create_app,
    model_form_depends,
    read_version_file,
)
from fastapi import Depends, File, Form, HTTPException, Request, UploadFile

from .adapter import BindFlowAdapter
from .models import FepCalculateRequest, MmpbsaCalculateRequest
from .settings import BindFlowSettings
from .tools import calculate_argv, list_ligands, sanitize_ligand_filename
from .uris import resolve_dir_zip, resolve_ligands, resolve_single_file

logger = logging.getLogger(__name__)

settings = BindFlowSettings()
adapter = BindFlowAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="BindFlow Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---------------------------------------------------------------------------
# /healthz/detail override — surface GROMACS + snakemake + gmx_MMPBSA presence.
# ---------------------------------------------------------------------------


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


_GMX_VERSION_RE = re.compile(r"GROMACS version:\s*([\d.]+)")
_SUPPORTED_GMX_MAJOR = {"2022", "2023", "2024", "2025"}


def _probe_gmx() -> tuple[bool, Optional[str], bool]:
    """Return (available, version_string, in_supported_range)."""
    gmx = shutil.which("gmx")
    if not gmx:
        return False, None, False
    try:
        result = subprocess.run(
            [gmx, "--version"], capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError):
        return True, None, False
    m = _GMX_VERSION_RE.search(result.stdout or "")
    if not m:
        return True, None, False
    version = m.group(1)
    major = version.split(".")[0]
    return True, version, major in _SUPPORTED_GMX_MAJOR


@app.get("/healthz/detail")
def healthz_detail(request: Request) -> dict:
    """Probe runtime deps.

    BindFlow has no NN weights; instead we surface whether `gmx`, `snakemake`,
    and `gmx_MMPBSA` are on PATH and — for gmx — whether the version is in
    BindFlow's supported [2022, 2026) range.  `weights_loaded` is retained
    for cross-service uniformity and reports the effective readiness signal.
    """
    gmx_avail, gmx_version, gmx_ok = _probe_gmx()
    snakemake_avail = shutil.which("snakemake") is not None
    gmx_mmpbsa_avail = shutil.which("gmx_MMPBSA") is not None

    ready = gmx_avail and gmx_ok and snakemake_avail

    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "gmx_available": gmx_avail,
        "gmx_version": gmx_version,
        "gmx_version_supported": gmx_ok,
        "snakemake_available": snakemake_avail,
        "gmx_mmpbsa_available": gmx_mmpbsa_avail,
        # Retained for cross-service uniformity — see design doc §6.5.
        "weights_dir": str(settings.weights_dir),
        "weights_loaded": ready,
        "weights_missing": {},
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
        "task_endpoints_enabled": settings.task_endpoints_enabled,
    }


# ---------------------------------------------------------------------------
# Shared input staging (used by both endpoints)
# ---------------------------------------------------------------------------


def _stage_inputs(
    input_dir: Path,
    *,
    protein: Optional[UploadFile],
    protein_uri: Optional[str],
    ligands: Optional[list[UploadFile]],
    ligands_zip_uri: Optional[str],
    cofactor: Optional[UploadFile],
    cofactor_uri: Optional[str],
    membrane: Optional[UploadFile],
    membrane_uri: Optional[str],
    custom_ff_zip: Optional[UploadFile],
    custom_ff_zip_uri: Optional[str],
    topology_zip: Optional[UploadFile],
    topology_zip_uri: Optional[str],
) -> dict[str, Path | None]:
    """Persist all input files under `<job_dir>/input/`; return absolute paths."""
    input_dir.mkdir(parents=True, exist_ok=True)

    protein_path = resolve_single_file(
        protein, protein_uri, input_dir / "protein.pdb", settings,
        required=True, field_name="protein",
    )
    ligands_dir = resolve_ligands(
        ligands, ligands_zip_uri, input_dir / "ligands", settings,
    )
    # Post-condition: at least one ligand + all names safe.
    try:
        for lig in list_ligands(ligands_dir):
            sanitize_ligand_filename(lig.name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None

    cofactor_path = resolve_single_file(
        cofactor, cofactor_uri, input_dir / "cofactor.sdf", settings,
        required=False, field_name="cofactor",
    )
    membrane_path = resolve_single_file(
        membrane, membrane_uri, input_dir / "membrane.pdb", settings,
        required=False, field_name="membrane",
    )
    custom_ff_dir = resolve_dir_zip(
        custom_ff_zip, custom_ff_zip_uri, input_dir / "custom_ff", settings,
        field_name="custom_ff",
    )
    topology_dir = resolve_dir_zip(
        topology_zip, topology_zip_uri, input_dir / "topology", settings,
        field_name="topology",
    )
    return {
        "protein_path": protein_path,
        "ligands_dir": ligands_dir,
        "cofactor_path": cofactor_path,
        "membrane_path": membrane_path,
        "custom_ff_dir": custom_ff_dir,
        "topology_dir": topology_dir,
    }


# ---------------------------------------------------------------------------
# /api/calculate/fep
# ---------------------------------------------------------------------------


@app.post("/api/calculate/fep", response_model=JobInfo)
def post_fep(
    params: FepCalculateRequest = Depends(model_form_depends(FepCalculateRequest)),
    protein: Optional[UploadFile] = File(default=None),
    protein_uri: Optional[str] = Form(default=None),
    ligands: Optional[list[UploadFile]] = File(default=None),
    ligands_zip_uri: Optional[str] = Form(default=None),
    cofactor: Optional[UploadFile] = File(default=None),
    cofactor_uri: Optional[str] = Form(default=None),
    membrane: Optional[UploadFile] = File(default=None),
    membrane_uri: Optional[str] = Form(default=None),
    custom_ff_zip: Optional[UploadFile] = File(default=None),
    custom_ff_zip_uri: Optional[str] = Form(default=None),
    topology_zip: Optional[UploadFile] = File(default=None),
    topology_zip_uri: Optional[str] = Form(default=None),
) -> JobInfo:
    """Submit a FEP calculation.  Long-running — poll /api/jobs/<id>."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        staged = _stage_inputs(
            job_dir / "input",
            protein=protein, protein_uri=protein_uri,
            ligands=ligands, ligands_zip_uri=ligands_zip_uri,
            cofactor=cofactor, cofactor_uri=cofactor_uri,
            membrane=membrane, membrane_uri=membrane_uri,
            custom_ff_zip=custom_ff_zip, custom_ff_zip_uri=custom_ff_zip_uri,
            topology_zip=topology_zip, topology_zip_uri=topology_zip_uri,
        )
        return calculate_argv(
            params,
            calculation_type="fep",
            job_dir=job_dir,
            protein_path=staged["protein_path"],
            ligands_dir=staged["ligands_dir"],
            cofactor_path=staged["cofactor_path"],
            membrane_path=staged["membrane_path"],
            custom_ff_dir=staged["custom_ff_dir"],
            topology_dir=staged["topology_dir"],
            settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build,
        label="fep",
        input_params=params.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# /api/calculate/mmpbsa
# ---------------------------------------------------------------------------


@app.post("/api/calculate/mmpbsa", response_model=JobInfo)
def post_mmpbsa(
    params: MmpbsaCalculateRequest = Depends(model_form_depends(MmpbsaCalculateRequest)),
    protein: Optional[UploadFile] = File(default=None),
    protein_uri: Optional[str] = Form(default=None),
    ligands: Optional[list[UploadFile]] = File(default=None),
    ligands_zip_uri: Optional[str] = Form(default=None),
    cofactor: Optional[UploadFile] = File(default=None),
    cofactor_uri: Optional[str] = Form(default=None),
    membrane: Optional[UploadFile] = File(default=None),
    membrane_uri: Optional[str] = Form(default=None),
    custom_ff_zip: Optional[UploadFile] = File(default=None),
    custom_ff_zip_uri: Optional[str] = Form(default=None),
    topology_zip: Optional[UploadFile] = File(default=None),
    topology_zip_uri: Optional[str] = Form(default=None),
) -> JobInfo:
    """Submit an MM(P/G)BSA calculation.  Requires gmx_MMPBSA in the image."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        staged = _stage_inputs(
            job_dir / "input",
            protein=protein, protein_uri=protein_uri,
            ligands=ligands, ligands_zip_uri=ligands_zip_uri,
            cofactor=cofactor, cofactor_uri=cofactor_uri,
            membrane=membrane, membrane_uri=membrane_uri,
            custom_ff_zip=custom_ff_zip, custom_ff_zip_uri=custom_ff_zip_uri,
            topology_zip=topology_zip, topology_zip_uri=topology_zip_uri,
        )
        return calculate_argv(
            params,
            calculation_type="mmpbsa",
            job_dir=job_dir,
            protein_path=staged["protein_path"],
            ligands_dir=staged["ligands_dir"],
            cofactor_path=staged["cofactor_path"],
            membrane_path=staged["membrane_path"],
            custom_ff_dir=staged["custom_ff_dir"],
            topology_dir=staged["topology_dir"],
            settings=settings,
        )

    return app.state.runner.submit(
        build_argv=_build,
        label="mmpbsa",
        input_params=params.model_dump(mode="json"),
    )


# Must come AFTER all POST routes so MCP auto-discovery sees the full surface.
attach_mcp(app)
