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

from bioagent_service import (
    JobInfo,
    create_app,
    execute_task,
    read_version_file,
    resolve_task_id,
)
from bioagent_service.models import JobStatus
from fastapi import Form, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from .adapter import MMseqs2JobAdapter
from .models import MSAJobSummary, TicketStatusResponse
from .settings import MMseqs2Settings
from .tools import (
    ModeConfig,
    ParsedSequence,
    colabfold_search_argv,
    parse_mode_flags,
    parse_query_fasta,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_fasta(
    records: list[ParsedSequence],
    *,
    paired: bool = False,
) -> str:
    """Render ParsedSequence records back into a FASTA string.

    The orchestrator wants a file on disk; this is the inverse of
    ``parse_query_fasta``.

    Serialization depends on whether this is a paired-multimer submission:

    * ``paired=False`` (default, /ticket/msa + /api/tasks/msa):
      each record is written as its own ``>{header}\\n{seq}\\n`` block —
      the orchestrator's ``get_queries`` treats each record as an
      independent monomer query and emits one MSA per record.

    * ``paired=True`` (/ticket/pair + /api/tasks/pair): chains are
      **collapsed into a single record** with sequences joined by ``:``.
      This is the ColabFold-canonical complex-query format that
      ``get_queries`` requires to flag ``is_complex=True`` (see
      ``services/mmseqs2-server/_colabfold_helpers.py:260-268``) — without
      it, orchestrator would classify 2+ records as 2+ independent
      monomers and skip the paired search entirely.

    ColabFold's own client (``opensource/ColabFold/colabfold/colabfold.py``
    line 82-84) submits multi-record FASTA to ``/ticket/pair`` and the
    ColabFold server does the same collapse internally; we mirror that
    behaviour so the wire protocol stays drop-in compatible.
    """
    if paired and len(records) >= 2:
        merged_seq = ":".join(r.sequence for r in records)
        # Preserve the first chain's header as the complex jobname — mmseqs
        # uses it only as a filename prefix, so exact value doesn't matter,
        # but keeping the first record's header makes log lines readable.
        return f">{records[0].header}\n{merged_seq}\n"
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

# Remove framework's generic /healthz/detail so we can override it with
# mmseqs2-specific signals (db_loaded, gpu_free_mb, ...). FastAPI uses
# first-match routing, so without this our handler below would be shadowed.
#
# Since FastAPI 0.115+, `app.include_router(...)` wraps each router in an
# `_IncludedRouter` object that owns the actual routes via `.original_router`.
# We have to descend into those wrappers to drop the framework's `/healthz/detail`.
def _strip_route(router, path: str, method: str) -> None:
    router.routes = [
        r
        for r in router.routes
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


# ---------------------------------------------------------------------------
# ColabFold-protocol endpoints
# ---------------------------------------------------------------------------


def _validate_msa_request(
    q: str,
    mode: str,
    *,
    require_paired: bool,
) -> tuple[list[ParsedSequence], ModeConfig]:
    """Parse + cross-validate the ColabFold-protocol ``q`` and ``mode`` fields.

    Returns the parsed FASTA records and the resolved mode config. Raises
    ``ValueError`` on any input rejection — callers translate that to the
    protocol-appropriate response (200 + ``{"status": "ERROR"}`` for ticket
    endpoints, HTTP 422 for task endpoints).
    """
    parsed_sequences = parse_query_fasta(q)
    mode_config = parse_mode_flags(mode)

    is_paired_mode = mode_config.pair_mode == "paired"
    if require_paired and not is_paired_mode:
        raise ValueError(f"mode {mode!r} is not a paired mode")
    if not require_paired and is_paired_mode:
        raise ValueError(f"mode {mode!r} is paired; use /ticket/pair or /api/tasks/pair")
    if require_paired and len(parsed_sequences) < 2:
        raise ValueError(
            f"paired mode needs >=2 sequences, got {len(parsed_sequences)}"
        )

    return parsed_sequences, mode_config


def _make_search_build(
    parsed_sequences: list[ParsedSequence],
    mode_config: ModeConfig,
):
    """Return a ``build_argv`` closure shared by ticket + task endpoints.

    The closure writes the query FASTA to ``<job_dir>/input/query.fasta`` and
    returns the argv to invoke ``server.orchestrator``. The shape matches
    both ``runner.submit``'s ``BuildArgv`` (``(job_id, job_dir) -> argv``)
    and ``execute_task``'s ``BuildArgvForTask`` (``(params, job_id, job_dir)
    -> argv``) by accepting both call signatures via ``*args``.
    """

    is_paired = mode_config.pair_mode == "paired"

    def _build(*args) -> list[str]:
        job_dir: Path = args[-1]
        input_dir = job_dir / "input"
        output_dir = job_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = input_dir / "query.fasta"
        fasta_path.write_text(
            _serialize_fasta(parsed_sequences, paired=is_paired),
            encoding="utf-8",
        )
        return colabfold_search_argv(
            query_path=fasta_path,
            output_dir=output_dir,
            mode_config=mode_config,
            settings=settings,
        )

    return _build


def _msa_summary(
    mode: str,
    parsed_sequences: list[ParsedSequence],
) -> dict:
    """input_params payload used by ticket endpoints (matches MSAJobSummary).

    Privacy: deliberately omits ``q`` — JobInfo is persisted to NAS.
    """
    return {
        "mode": mode,
        "sequence_count": len(parsed_sequences),
        "total_residues": sum(len(s.sequence) for s in parsed_sequences),
    }


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
    try:
        parsed_sequences, mode_config = _validate_msa_request(
            q, mode, require_paired=require_paired,
        )
    except ValueError as e:
        logger.info("rejecting %s request: %s", label, e)
        return {"status": "ERROR"}

    try:
        job = request.app.state.runner.submit(
            build_argv=_make_search_build(parsed_sequences, mode_config),
            label=label,
            input_params=_msa_summary(mode, parsed_sequences),
        )
    except Exception:
        logger.exception("submit failed for %s (mode=%s)", label, mode)
        return {"status": "ERROR"}

    return {"id": job.job_id, "job_id": job.job_id, "status": "PENDING"}


@app.post("/ticket/msa")
def post_ticket_msa(
    request: Request,
    q: str = Form(default=""),
    mode: str = Form(default=""),
) -> dict:
    """Submit a monomer (unpaired) MSA job. ColabFold protocol."""
    return _submit_msa_job(request, q=q, mode=mode, label="msa", require_paired=False)


@app.post("/ticket/pair")
def post_ticket_pair(
    request: Request,
    q: str = Form(default=""),
    mode: str = Form(default=""),
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


def _concat_a3ms_with_null_separator(paths: list[Path]) -> bytes:
    """Concatenate a3m files with ``\\x00`` separators (ColabFold wire format).

    Both upstream ColabFold's ``run_mmseqs2`` and Boltz's ``run_mmseqs2`` fork
    parse the downloaded a3m as a **single file** where distinct query
    sections are separated by a null byte (see ``colabfold.py:301-315`` /
    ``boltz/data/msa/mmseqs2.py:268-284``).  The parser flips ``update_M``
    on ``\\x00`` and then reads the next ``>{numeric_id}`` header as the start
    of a new query.

    Ordering: we sort by filename so numeric-jobname queries (e.g. ``101.a3m``,
    ``102.a3m``) come out in the order the client submitted them.  Empty
    files are dropped — an empty a3m entry would otherwise leave the parser
    with an unresolvable ``M`` key.
    """
    payload = b""
    for i, p in enumerate(sorted(paths)):
        chunk = p.read_bytes()
        if not chunk:
            continue
        if payload:
            payload += b"\x00"
        payload += chunk
    return payload


def _build_colabfold_tarball(out_dir: Path) -> bytes | None:
    """Repack the orchestrator's output as a ColabFold-canonical tar.gz.

    Contract enforced (mirrors ``api.colabfold.com`` layout so
    ``boltz --use_msa_server=<us>`` and upstream ColabFold notebooks work
    against us as a drop-in replacement):

    * ``uniref.a3m`` — concatenation of every ``<jobname>.a3m`` (unpaired,
      i.e. NOT ``.paired.a3m``) with ``\\x00`` separators.  Present iff at
      least one unpaired MSA was produced.
    * ``pair.a3m``   — concatenation of every ``<jobname>.paired.a3m`` (and
      ``<jobname>.env.paired.a3m`` for env-pair mode) with ``\\x00``
      separators.  Present iff at least one paired MSA was produced.

    Returns the raw tar.gz bytes, or ``None`` if ``out_dir`` has no a3m
    outputs at all.

    NOTE: ``mode=env`` unpaired output is currently merged into the single
    ``<jobname>.a3m`` file at the orchestrator layer (upstream ColabFold
    keeps ``uniref.a3m`` + ``bfd.mgnify30.metaeuk30.smag30.a3m`` separate).
    Clients that pass ``use_env=True`` and expect the second file will need
    a follow-up orchestrator change; today they still get all the sequence
    data, just concatenated into ``uniref.a3m``.
    """
    if not out_dir.is_dir():
        return None

    all_a3m = [
        p for p in out_dir.rglob("*.a3m")
        if p.is_file() and p.stat().st_size > 0
    ]
    if not all_a3m:
        return None

    paired = [p for p in all_a3m if p.name.endswith(".paired.a3m")]
    unpaired = [p for p in all_a3m if not p.name.endswith(".paired.a3m")]

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for arcname, sources in (
            ("uniref.a3m", unpaired),
            ("pair.a3m", paired),
        ):
            if not sources:
                continue
            payload = _concat_a3ms_with_null_separator(sources)
            if not payload:
                continue
            info = tarfile.TarInfo(name=arcname)
            info.size = len(payload)
            info.mode = 0o644
            tf.addfile(info, io.BytesIO(payload))
    buf.seek(0)
    return buf.getvalue()


@app.get("/result/download/{job_id}")
def get_result_download(request: Request, job_id: str):
    """Stream a tar.gz of the job's a3m outputs in ColabFold-canonical layout.

    The tarball contains ``uniref.a3m`` and/or ``pair.a3m`` at the top level
    with per-query MSAs concatenated using ``\\x00`` separators — the wire
    format both upstream ColabFold and Boltz expect from
    ``https://api.colabfold.com/result/download/<id>``.  See
    ``_build_colabfold_tarball`` for the full contract.

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
    payload = _build_colabfold_tarball(out_dir)
    if payload is None:
        return StreamingResponse(
            iter([b'{"status": "ERROR"}']),
            status_code=503,
            media_type="application/json",
        )

    return StreamingResponse(
        io.BytesIO(payload),
        media_type="application/x-tar",
        headers={"Content-Disposition": f"attachment; filename={job_id}.tar.gz"},
    )


# ---------------------------------------------------------------------------
# Task endpoints — FC Async Task Mode (sibling to /ticket/* + /result/*)
#
# Same argv builder as the ticket path, but the HTTP request blocks until the
# orchestrator subprocess completes. Designed to be invoked with
# `X-Fc-Invocation-Type: Async` so FC's platform layer manages queueing and
# the function instance stays alive for the full computation. ColabFold
# clients keep using /ticket/* — these are for agents / boltz-server that
# speak our JobInfo protocol.
# ---------------------------------------------------------------------------


def _task_msa_handler(
    request: Request,
    q: str,
    mode: str,
    label: str,
    require_paired: bool,
    x_bioagent_job_id: Optional[str],
    x_fc_async_task_id: Optional[str],
) -> JobInfo:
    """Shared handler body for /api/tasks/msa and /api/tasks/pair.

    Validation failures surface as HTTP 422 (ColabFold's 200-with-status-ERROR
    convention is for browser-style clients; agent-style callers want the real
    HTTP status). Subprocess failures still flow through `execute_task` and
    return 200 + ``JobInfo.status="failed"``.
    """
    job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
    try:
        parsed_sequences, mode_config = _validate_msa_request(
            q, mode, require_paired=require_paired,
        )
    except ValueError as e:
        logger.info("rejecting task %s (job_id=%s): %s", label, job_id, e)
        raise HTTPException(status_code=422, detail=str(e)) from e

    summary = MSAJobSummary(
        mode=mode,
        sequence_count=len(parsed_sequences),
        total_residues=sum(len(s.sequence) for s in parsed_sequences),
    )
    return execute_task(
        request,
        job_id=job_id,
        label=label,
        params=summary,
        build_argv=_make_search_build(parsed_sequences, mode_config),
    )


if settings.task_endpoints_enabled:

    @app.post("/api/tasks/msa", response_model=JobInfo)
    def post_tasks_msa(
        request: Request,
        q: str = Form(default=""),
        mode: str = Form(default=""),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Monomer MSA as a single atomic task (blocks until completion)."""
        return _task_msa_handler(
            request,
            q=q, mode=mode, label="msa", require_paired=False,
            x_bioagent_job_id=x_bioagent_job_id,
            x_fc_async_task_id=x_fc_async_task_id,
        )

    @app.post("/api/tasks/pair", response_model=JobInfo)
    def post_tasks_pair(
        request: Request,
        q: str = Form(default=""),
        mode: str = Form(default=""),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Multimer paired MSA as a single atomic task (blocks until completion)."""
        return _task_msa_handler(
            request,
            q=q, mode=mode, label="pair", require_paired=True,
            x_bioagent_job_id=x_bioagent_job_id,
            x_fc_async_task_id=x_fc_async_task_id,
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
