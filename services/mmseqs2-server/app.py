"""FastAPI app for mmseqs2-server.

Exposes the ColabFold MSA HTTP protocol (4 endpoints) on top of the framework's
generic `/healthz`, `/api/jobs/*`, and `/api/manifest` routes:

  * POST /ticket/msa          — submit a monomer (unpaired) MSA job
  * POST /ticket/pair         — submit a multimer (paired) MSA job
  * GET  /ticket/{id}         — poll job status (PENDING/RUNNING/COMPLETE/ERROR)
  * GET  /result/download/{id}— stream a tar.gz of the *.a3m output files
  * GET  /healthz/detail      — extended health (db_loaded, gpu_free_mb, ...)

Errors are returned with HTTP 200 + ``{"status": "ERROR"}`` because the
ColabFold client treats 4xx/5xx as fatal protocol errors and refuses to retry.
Sequence content is never logged — only sequence count and mode end up in
``JobInfo.input_params``.
"""

from __future__ import annotations

import io
import logging
import subprocess
import tarfile
from pathlib import Path
from typing import Optional

from bioagent_service import create_app, read_version_file
from bioagent_service.models import JobStatus
from fastapi import Form, Request
from fastapi.responses import StreamingResponse

from .adapter import MMseqs2JobAdapter
from .models import TicketStatusResponse, TicketSubmitResponse
from .settings import MMseqs2Settings
from .tools import (
    ParsedSequence,
    colabfold_search_argv,
    parse_mode_flags,
    parse_query_fasta,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_fasta(records: list[ParsedSequence]) -> str:
    """Render ParsedSequence records back into a FASTA string.

    The orchestrator wants a file on disk; this is the inverse of
    ``parse_query_fasta``. Each record is written as ``>{header}\\n{seq}\\n``.
    """
    return "".join(f">{r.header}\n{r.sequence}\n" for r in records)


def _truncate_error(text: Optional[str], limit: int = 200) -> Optional[str]:
    """Cap error strings to ``limit`` chars to avoid leaking subprocess stack traces."""
    if text is None:
        return None
    return text if len(text) <= limit else text[: limit - 1] + "…"


# JobStatus → ColabFold protocol status. Anything terminal-but-not-COMPLETED
# maps to ERROR; transient states map to PENDING / RUNNING.
_STATUS_MAP = {
    JobStatus.PENDING: "PENDING",
    JobStatus.RUNNING: "RUNNING",
    JobStatus.COMPLETED: "COMPLETE",
    JobStatus.FAILED: "ERROR",
}


def _gpu_free_mb() -> int:
    """Return free GPU memory in MiB, or -1 if nvidia-smi is unavailable.

    Best-effort: any subprocess error (missing binary, no GPU, parse failure)
    falls back to -1 so /healthz/detail never crashes. We only read the first
    line so multi-GPU instances report the primary device.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return -1
        first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        return int(first_line.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return -1


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


settings = MMseqs2Settings()
adapter = MMseqs2JobAdapter(settings=settings)

app = create_app(
    adapter,
    settings,
    title="MMseqs2 Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---------------------------------------------------------------------------
# ColabFold-protocol endpoints
# ---------------------------------------------------------------------------


def _submit_msa_job(
    request: Request,
    *,
    q: str,
    mode: str,
    label: str,
    require_paired: bool,
) -> dict:
    """Shared submission path for /ticket/msa and /ticket/pair.

    Returns a plain dict (rather than the Pydantic model) so error responses
    can omit the `id` field without tripping schema validation. The session
    affinity middleware reads `job_id` if present, so on success we include it
    too (under both `id` and `job_id` — `job_id` is what FC's HeaderField
    affinity middleware looks for, `id` is the ColabFold protocol field).
    """
    # ------- Input validation -------
    try:
        parsed_sequences = parse_query_fasta(q)
        mode_config = parse_mode_flags(mode)
    except ValueError as e:
        logger.info("rejecting %s request: %s", label, e)
        return {"status": "ERROR"}

    is_paired_mode = mode_config.pair_mode == "paired"
    if require_paired and not is_paired_mode:
        logger.info("rejecting /ticket/pair: mode %r is not a paired mode", mode)
        return {"status": "ERROR"}
    if not require_paired and is_paired_mode:
        logger.info("rejecting /ticket/msa: mode %r is paired", mode)
        return {"status": "ERROR"}
    if require_paired and len(parsed_sequences) < 2:
        logger.info(
            "rejecting /ticket/pair: only %d sequence(s) — paired needs >=2",
            len(parsed_sequences),
        )
        return {"status": "ERROR"}

    # ------- Build closure -------
    def _build(_job_id: str, job_dir: Path) -> list[str]:
        input_dir = job_dir / "input"
        output_dir = job_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = input_dir / "query.fasta"
        fasta_path.write_text(_serialize_fasta(parsed_sequences), encoding="utf-8")
        return colabfold_search_argv(
            query_path=fasta_path,
            output_dir=output_dir,
            mode_config=mode_config,
            settings=settings,
        )

    # ------- Submit -------
    #
    # Privacy note: deliberately do NOT include `q` (the query sequence) in
    # input_params. JobInfo is persisted to NAS, so anything we put here lives
    # well past the subprocess lifetime. Sequence length + count is enough for
    # operational debugging.
    try:
        job = app.state.runner.submit(
            build_argv=_build,
            label=label,
            input_params={
                "mode": mode,
                "sequence_count": len(parsed_sequences),
                "total_residues": sum(len(s.sequence) for s in parsed_sequences),
            },
        )
    except Exception:
        logger.exception("submit failed for %s (mode=%s)", label, mode)
        return {"status": "ERROR"}

    return {"id": job.job_id, "job_id": job.job_id, "status": "PENDING"}


@app.post("/ticket/msa", response_model=TicketSubmitResponse)
def post_ticket_msa(
    request: Request,
    q: str = Form(...),
    mode: str = Form(...),
) -> dict:
    """Submit a monomer (unpaired) MSA job. ColabFold protocol."""
    return _submit_msa_job(request, q=q, mode=mode, label="msa", require_paired=False)


@app.post("/ticket/pair", response_model=TicketSubmitResponse)
def post_ticket_pair(
    request: Request,
    q: str = Form(...),
    mode: str = Form(...),
) -> dict:
    """Submit a multimer (paired) MSA job. ColabFold protocol."""
    return _submit_msa_job(request, q=q, mode=mode, label="pair", require_paired=True)


@app.get("/ticket/{job_id}", response_model=TicketStatusResponse)
def get_ticket_status(request: Request, job_id: str) -> dict:
    """Poll job status, translated to ColabFold's protocol vocabulary.

    Returns HTTP 200 with `{"status": "ERROR"}` if the job is missing — the
    ColabFold client doesn't handle 404 gracefully so we collapse "not found"
    and "failed" into the same response shape.
    """
    info = request.app.state.job_store.get(job_id)
    if info is None:
        return {"id": job_id, "status": "ERROR", "error": "job not found"}

    cf_status = _STATUS_MAP.get(info.status, "ERROR")
    response: dict = {"id": job_id, "status": cf_status}
    if cf_status == "ERROR":
        response["error"] = _truncate_error(info.error_summary) or info.message or "job failed"
    return response


@app.get("/result/download/{job_id}")
def get_result_download(request: Request, job_id: str):
    """Stream a tar.gz of the job's `.a3m` files.

    The ColabFold client parses the response body as a tarball regardless of
    Content-Type (``tarfile.open(fileobj=response.raw, mode='r|gz')``), but we
    still set ``application/x-tar`` for honest HTTP semantics. Errors return
    HTTP 503 with ``{"status": "ERROR"}`` per the design doc.
    """
    info = request.app.state.job_store.get(job_id)
    if info is None or info.status != JobStatus.COMPLETED:
        return StreamingResponse(
            iter([b'{"status": "ERROR"}']),
            status_code=503,
            media_type="application/json",
        )

    out_dir = adapter.output_dir(adapter.job_dir(job_id))
    a3m_files = [p for p in out_dir.rglob("*.a3m") if p.is_file() and p.stat().st_size > 0]
    if not a3m_files:
        return StreamingResponse(
            iter([b'{"status": "ERROR"}']),
            status_code=503,
            media_type="application/json",
        )

    # Build the tarball in-memory; .a3m payloads are typically a few MiB even
    # for multimers, so this is fine without a temp file. Stream the buffer
    # back to the client.
    buf = io.BytesIO()
    out_resolved = out_dir.resolve()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path in a3m_files:
            tf.add(path, arcname=str(path.resolve().relative_to(out_resolved)))
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/x-tar",
        headers={"Content-Disposition": f"attachment; filename={job_id}.tar.gz"},
    )


@app.get("/healthz/detail")
def healthz_detail(request: Request) -> dict:
    """Extended health: ColabFold-protocol-specific signals.

    Overrides the framework's generic `/healthz/detail` so the agent sees
    mmseqs2-specific signals (db presence, GPU memory) instead of the generic
    disk-usage view. All optional probes are best-effort.
    """
    # Database loaded? The mmseqs index file is the cheapest probe.
    db_idx_path = settings.db_dir / f"{settings.default_db}.idx"
    db_loaded = db_idx_path.exists()

    return {
        "service": adapter.name,
        "version": request.app.version,
        "db_loaded": db_loaded,
        "db_dir": str(settings.db_dir),
        "default_db": settings.default_db,
        "env_db": settings.env_db,
        "gpu_enabled": settings.gpu_enabled,
        "gpu_free_mb": _gpu_free_mb(),
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }
