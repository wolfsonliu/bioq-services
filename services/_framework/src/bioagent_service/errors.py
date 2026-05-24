"""Subprocess log parsing + job finalization.

`extract_error_summary` was originally inlined in each service's tasks.py; lifting
it here means every framework user gets the same triage-quality FAILED responses
(error_summary + error_tail) for free.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from bioagent_service.models import FailureKind, JobStatus, utcnow

if TYPE_CHECKING:
    from bioagent_service.adapter import JobAdapter
    from bioagent_service.jobs import JobStore


def _output_summary(output_dir: Path) -> tuple[int, int]:
    """Walk *output_dir* and return ``(file_count, total_bytes)``."""
    count = total = 0
    if output_dir.exists():
        for f in output_dir.rglob("*"):
            if f.is_file():
                count += 1
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    return count, total


# Matches the last meaningful exception line in a Python traceback, e.g.:
#   ValueError: bad input
#   torch.cuda.OutOfMemoryError: CUDA out of memory ...
#   genie3.cli.errors.SomethingError: ...
_EXC_LINE_RE = re.compile(
    r"^([A-Za-z_][\w.]*(?:Error|Exception)|Exception)\s*:\s*(.+)$"
)


def extract_error_summary(
    log_path: Path, tail_chars: int = 4000
) -> tuple[str | None, str | None]:
    """Scan the log file and return (one-line exception summary, trailing slice).

    - The summary is the LAST line that looks like `ExceptionType: message`
      (walking the file bottom-up, since stack traces print the exception last).
    - The tail is the final `tail_chars` characters of the file, useful for clients
      that want context without a separate /log fetch.
    - Returns (None, None) if the file is missing; (None, "...") if it exists but
      no exception line matches.
    """
    if not log_path.exists():
        return None, None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, "(log file unreadable)"

    tail = text[-tail_chars:] if len(text) > tail_chars else text

    summary: str | None = None
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        m = _EXC_LINE_RE.match(stripped)
        if m:
            summary = stripped
            break

    # Fallback: if nothing matched the exception regex but the log isn't empty,
    # surface the last non-empty line so the client gets *something* useful.
    if summary is None:
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if stripped:
                summary = stripped
                break

    return summary, tail


def finalize_job(
    store: "JobStore",
    adapter: "JobAdapter",
    job_id: str,
    rc: int,
    label: str,
    *,
    error_tail_chars: int = 4000,
) -> None:
    """Transition a finished subprocess into COMPLETED or FAILED in the store.

    - rc == 0 AND adapter.detect_outputs(...) → COMPLETED
    - rc != 0                                  → FAILED, failure_kind=SUBPROCESS_ERROR
    - rc == 0 but no outputs                   → FAILED, failure_kind=NO_OUTPUTS
    Both failure cases attach error_summary + error_tail from the log.
    """
    job = store.get(job_id)
    if job is None:
        return  # caller already cleaned up

    now = utcnow()
    duration = (
        (now - job.started_at).total_seconds() if job.started_at else None
    )
    job_dir = adapter.job_dir(job_id)
    out_count, out_bytes = _output_summary(adapter.output_dir(job_dir))

    if rc == 0 and adapter.detect_outputs(job_dir):
        store.update(
            job_id,
            status=JobStatus.COMPLETED,
            message=f"{label} completed",
            completed_at=now,
            duration_seconds=duration,
            output_count=out_count or None,
            output_total_bytes=out_bytes or None,
            error_summary=None,
            error_tail=None,
            failure_kind=None,
        )
        return

    log_path = adapter.log_path(job_dir)
    summary, tail = extract_error_summary(log_path, tail_chars=error_tail_chars)

    if rc != 0:
        kind = FailureKind.SUBPROCESS_ERROR
        base_msg = f"{label} failed (rc={rc})"
    else:
        kind = FailureKind.NO_OUTPUTS
        base_msg = f"{label} exited 0 but produced no outputs"

    store.update(
        job_id,
        status=JobStatus.FAILED,
        message=f"{base_msg}: {summary}" if summary else base_msg,
        completed_at=now,
        duration_seconds=duration,
        output_count=out_count or None,
        output_total_bytes=out_bytes or None,
        error_summary=summary,
        error_tail=tail,
        failure_kind=kind,
    )


__all__ = ["FailureKind", "extract_error_summary", "finalize_job"]
