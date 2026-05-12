"""JobAdapter — service-wide policy for jobs.

An adapter encapsulates everything that's *uniform* across a service's endpoints:
filesystem layout, output detection, log location, subprocess env / cwd, and how
to rebuild a job's state from disk after a restart. It deliberately does NOT
know about specific request shapes or argv-building — those are per-endpoint
concerns. A service may expose multiple endpoints (e.g. `/api/rfdiffusion`,
`/api/proteinmpnn`, `/api/rf2`), each constructing its own argv before calling
`runner.submit`. The adapter is the same for all of them.

This split keeps two assumptions clean:

  1. A service is one Docker image with one set of weights / dependencies. The
     adapter captures that singularity.
  2. The pipeline (chaining different tools / FCs together) belongs to the
     orchestrator one level up, not inside an individual FC service.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from bioagent_service.jobs import get_job_dir
from bioagent_service.models import JobInfo, JobStatus
from bioagent_service.settings import ServiceSettings

if TYPE_CHECKING:
    from bioagent_service.manifest import EndpointExample


class JobAdapter:
    """Service-wide policy hook surface.

    Subclasses set `name` and override any of the optional methods whose
    defaults don't match the underlying tool. There is nothing abstract here —
    a service that's happy with every default can use `JobAdapter` directly,
    passing only `name` and `settings`.
    """

    name: str = ""

    def __init__(self, settings: ServiceSettings) -> None:
        if not self.name:
            raise TypeError(f"{type(self).__name__}.name must be set")
        self.settings = settings

    # ---- Filesystem layout (override if tool's defaults differ) ----

    def job_dir(self, job_id: str) -> Path:
        """Absolute path to this job's working directory."""
        return get_job_dir(self.settings.jobs_base_dir, job_id)

    def output_dir(self, job_dir: Path) -> Path:
        """Directory that `GET /api/jobs/{id}/download` will zip + `/files` will list."""
        return job_dir / "output"

    def log_path(self, job_dir: Path) -> Path:
        """Path of the subprocess log; framework tees stdout+stderr here."""
        return job_dir / "logs" / "run.log"

    # ---- Lifecycle hooks (override as needed) ----

    def detect_outputs(self, job_dir: Path) -> bool:
        """Return True iff this job produced its expected outputs.

        Used by `finalize_job`: if rc==0 but this returns False, the job is marked
        FAILED with `failure_kind=NO_OUTPUTS`. Default: `output_dir` is non-empty.

        Multi-endpoint services (e.g., rfantibody) typically override to check
        for the presence of any of the per-tool output files.
        """
        out = self.output_dir(job_dir)
        return out.is_dir() and any(out.iterdir())

    def subprocess_env(self) -> dict[str, str]:
        """Extra environment variables to merge into the subprocess. Default: none."""
        return {}

    def subprocess_cwd(self) -> Path | None:
        """Working directory for the subprocess. Default: process inherits caller's."""
        return None

    def endpoint_examples(self) -> dict[str, list["EndpointExample"]]:
        """Copy-pasteable examples per endpoint, keyed by URL path.

        Returned by `GET /api/manifest` as part of each endpoint's `examples`
        list. Each `EndpointExample` carries a `curl` / `python` / `body` /
        `notes` snippet. Strongly recommended to populate at least one example
        per endpoint — agents can usually translate a working curl call into a
        valid SDK invocation without further hints.

        Default: empty. Override to add examples. See
        `services/rfantibody-server/adapter.py` for a worked example.
        """
        return {}

    def manifest_extras(self) -> dict:
        """Service-specific fields to expose on `GET /api/manifest`.

        The framework's manifest already includes the service name, version,
        the list of HTTP endpoints, job lifecycle protocol, and NAS layout. Use
        this hook to add anything an LLM agent would need to *call the service
        effectively* but that the framework can't infer:

          * Output filename conventions (where each tool's results land)
          * Supported input URI schemes (`job://`, `file://`, `oss://`, ...)
          * Cross-tool hints (e.g., "use input_uri=job://<prev>/<file> instead
            of re-uploading bytes")
          * Required env vars, GPU requirements, weight files, etc.

        Default: empty dict. **Strongly recommended to override** so the
        manifest is informative — see `services/rfantibody-server/adapter.py`
        for a working example.
        """
        return {}

    def infer_job_from_dir(self, job_dir: Path) -> JobInfo:
        """Construct a JobInfo for a job dir that has no `job.json` sidecar.

        Called by `reload_from_disk` for legacy directories created before
        sidecar persistence was wired in (or by services migrating from a
        stateless setup).

        Default heuristic: if `detect_outputs` says outputs are present, mark
        COMPLETED; otherwise FAILED. Subclasses with richer per-step state
        should override to populate `progress` or message-style summary.
        """
        job_id = job_dir.name
        if self.detect_outputs(job_dir):
            return JobInfo(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                message="Recovered from disk (outputs present)",
            )
        return JobInfo(
            job_id=job_id,
            status=JobStatus.FAILED,
            message="Recovered from disk (no outputs)",
        )


__all__ = ["JobAdapter"]
