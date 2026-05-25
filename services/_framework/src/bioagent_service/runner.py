"""Subprocess execution + background job submission.

`SubprocessRunner` runs an argv synchronously, tee-ing stdout/stderr to a log
file so `finalize_job` can extract the last exception. `JobRunner` is the higher
level wrapper: each service-side endpoint calls `submit(build_argv=...)`, the
runner allocates a job dir, invokes the callback to get the actual argv, runs
the subprocess on its executor, and finalizes the JobStore entry.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from concurrent.futures import Executor
from pathlib import Path
from typing import Any, Callable

from bioagent_service.adapter import JobAdapter
from bioagent_service.errors import finalize_job
from bioagent_service.jobs import JobStore, cleanup_job, evict_finished_until_under_limit
from bioagent_service.models import FailureKind, JobInfo, JobStatus, utcnow
from bioagent_service.settings import ServiceSettings

logger = logging.getLogger(__name__)


BuildArgv = Callable[[str, Path], list[str]]
"""Endpoint-supplied closure: (job_id, job_dir) → argv.

Called synchronously inside `JobRunner.submit` after the job dir is created but
before the subprocess is scheduled, so it's the right place to:
  * Stream uploaded files into `<job_dir>/input/...`
  * Write per-job config files (e.g., YAML) into `<job_dir>/`
  * Resolve external URIs (oss://, http://, job://) to local paths
…and produce an argv whose output paths point into `<job_dir>/output/`.
"""


class SubprocessRunner:
    """Synchronous argv → rc executor that tees output to a log file."""

    @staticmethod
    def run(
        argv: list[str],
        log_path: Path,
        *,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        check_interval_s: float = 30.0,
    ) -> int:
        """Run `argv`, write merged stdout/stderr to `log_path`, return rc.

        Uses ``proc.wait(timeout=check_interval_s)`` in a loop instead of a
        blocking ``proc.wait()`` so the thread remains responsive.  On each
        iteration a ``proc.poll()`` cross-check catches edge cases where the
        child has exited but ``wait()`` did not return (zombie reap race).

        On Popen failure (e.g., binary missing) the exception is written to the
        log and rc=127 is returned, so callers can rely on the log being present
        when finalize_job runs.
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)
        full_env = {**os.environ, **(env or {})}
        try:
            with open(log_path, "wb") as logf:
                proc = subprocess.Popen(
                    argv,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    env=full_env,
                    cwd=str(cwd) if cwd else None,
                )
                while True:
                    try:
                        return proc.wait(timeout=check_interval_s)
                    except subprocess.TimeoutExpired:
                        rc = proc.poll()
                        if rc is not None:
                            logger.warning(
                                "proc.poll() returned %d but wait() timed out "
                                "(pid=%d); treating as exited",
                                rc, proc.pid,
                            )
                            return rc
        except FileNotFoundError as e:
            log_path.write_text(f"failed to spawn {argv[0]!r}: {e}\n", encoding="utf-8")
            return 127
        except OSError as e:
            log_path.write_text(f"OSError launching subprocess: {e}\n", encoding="utf-8")
            return 1


class JobRunner:
    """Glue between an HTTP handler and the JobStore + subprocess execution.

    The adapter is bound at construction time because it's service-wide; only
    per-job concerns (the callback that produces argv, optional env / cwd
    overrides) are passed to `submit`.
    """

    def __init__(
        self,
        store: JobStore,
        executor: Executor,
        settings: ServiceSettings,
        adapter: JobAdapter,
    ) -> None:
        self.store = store
        self.executor = executor
        self.settings = settings
        self.adapter = adapter
        self._active_count = 0
        self._active_lock = threading.Lock()

    @property
    def active_job_count(self) -> int:
        """Number of jobs currently queued or running in the executor."""
        return self._active_count

    def submit(
        self,
        *,
        build_argv: BuildArgv,
        label: str | None = None,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        input_params: dict[str, Any] | None = None,
    ) -> JobInfo:
        """Create a job, build the argv via the callback, schedule the subprocess.

        Returns immediately with a JobInfo in PENDING (about to flip to RUNNING).

        `label` is used in status messages — typically the endpoint name
        (`"rfdiffusion"`, `"proteinmpnn"`). Defaults to `adapter.name`.
        `env` / `cwd` override the adapter defaults for this one job.
        `input_params` is an opaque dict echoed back in JobInfo for debugging.
        """
        adapter = self.adapter
        # Best-effort disk hygiene before scheduling new work.
        evict_finished_until_under_limit(
            self.store, self.settings.jobs_base_dir, self.settings.disk_limit_mb
        )

        job = self.store.create(input_params=input_params)
        job_id = job.job_id
        job_dir = adapter.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "output").mkdir(exist_ok=True)
        adapter.log_path(job_dir).parent.mkdir(parents=True, exist_ok=True)

        # Build argv inside the request handler so upload streams + path lookups
        # can use the just-created job_dir. If the callback raises (bad zip,
        # missing input, fastapi.HTTPException for validation errors, ...),
        # clean up the half-created job so callers don't leave PENDING
        # stragglers in the store and on disk.
        try:
            argv = build_argv(job_id, job_dir)
            if not argv:
                raise ValueError("build_argv returned an empty argv")
        except Exception:
            cleanup_job(self.store, self.settings.jobs_base_dir, job_id)
            raise

        resolved_label = label or adapter.name
        resolved_env = {**adapter.subprocess_env(), **(env or {})}
        resolved_cwd = cwd if cwd is not None else adapter.subprocess_cwd()

        def _run() -> None:
            try:
                self.store.update(
                    job_id,
                    status=JobStatus.RUNNING,
                    message=f"{resolved_label} running",
                    started_at=utcnow(),
                )
                rc = SubprocessRunner.run(
                    argv,
                    adapter.log_path(job_dir),
                    env=resolved_env,
                    cwd=resolved_cwd,
                )
                finalize_job(
                    self.store,
                    adapter,
                    job_id,
                    rc,
                    resolved_label,
                    error_tail_chars=self.settings.error_tail_chars,
                )
            except Exception:
                logger.exception("unhandled error in _run for job %s", job_id)
                try:
                    self.store.update(
                        job_id,
                        status=JobStatus.FAILED,
                        message=f"{resolved_label} failed: internal runner error",
                        completed_at=utcnow(),
                        failure_kind=FailureKind.SUBPROCESS_ERROR,
                        error_summary="internal runner error (see server logs)",
                    )
                except Exception:
                    logger.exception("failed to mark job %s as FAILED", job_id)
            finally:
                with self._active_lock:
                    self._active_count -= 1

        with self._active_lock:
            self._active_count += 1
        self.executor.submit(_run)
        # Return the freshly-created PENDING info; status may already be RUNNING by the
        # time the client reads it back from /api/jobs/{id}.
        return self.store.get(job_id) or job


__all__ = ["BuildArgv", "JobRunner", "SubprocessRunner"]
