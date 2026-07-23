# bioq-service-framework

Reusable HTTP / job / error-handling layer shared by all bioagent algorithm services.

A new service is ~50 lines: one `JobAdapter` subclass (how to build the subprocess argv)
plus one `ServiceSettings` subclass (env-driven config). The framework supplies the
FastAPI app, job store, error extraction, upload/download endpoints, and OpenAPI schema.

## New service?

**Start here:** [engineering/guides/adding-a-new-service.md](../../engineering/guides/adding-a-new-service.md)
— canonical cookbook with the 10-file checklist, copy-pasteable skeleton, and verification steps.

## Design

See [engineering/decisions/2026-05-12-service-framework-design.md](../../engineering/decisions/2026-05-12-service-framework-design.md).

## Install (in a service's Dockerfile)

```dockerfile
COPY framework /tmp/service-framework
RUN pip install /tmp/service-framework         # or: uv pip install /tmp/service-framework
```

## Minimal service

```python
# services/echo-server/app.py
from pathlib import Path
from bioq_service import JobAdapter, ServiceSettings, create_app
from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict


class EchoRequest(BaseModel):
    message: str


class EchoSettings(ServiceSettings):
    model_config = SettingsConfigDict(env_prefix="ECHO_", env_file=".env", extra="ignore")


class EchoAdapter(JobAdapter[EchoRequest]):
    name = "echo"
    request_model = EchoRequest

    def build_command(self, job_id: str, job_dir: Path, request: EchoRequest) -> list[str]:
        out = job_dir / "output"
        out.mkdir(parents=True, exist_ok=True)
        return ["bash", "-c", f"echo {request.message!r} > {out}/out.txt"]


settings = EchoSettings()
adapter = EchoAdapter(settings=settings)
app = create_app(adapter, settings, title="Echo Service")


@app.post("/api/echo")
def echo(request: EchoRequest):
    return app.state.runner.submit(adapter, request)
```

## Endpoints (auto-registered)

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness |
| GET | `/healthz/detail` | jobs_dir presence + disk usage |
| GET | `/api/manifest` | **Agent-friendly protocol description** (service identity, endpoint list, job lifecycle, NAS layout, service extras) |
| GET | `/openapi.json` | Full JSON Schema (request/response models) |
| GET | `/api/jobs/{job_id}` | Job status (JobInfo, includes error_summary/error_tail) |
| GET | `/api/jobs/{job_id}/files` | List output files |
| GET | `/api/jobs/{job_id}/log` | Full subprocess log |
| GET | `/api/jobs/{job_id}/download` | Zip of output dir |
| GET | `/api/jobs/{job_id}/file/{path:path}` | Single file download (path-traversal safe) |
| DELETE | `/api/jobs/{job_id}` | Delete job (clears dir + store) |

## Conventions

- All request/response types are `pydantic.BaseModel` — `dict[str, Any]` is banned.
- All runtime config goes through `pydantic_settings.BaseSettings` subclasses —
  no `os.getenv` calls inside the framework or in adapter code.
- **Every service must populate both adapter introspection hooks**:
  - `JobAdapter.manifest_extras()` for service-specific protocol knowledge
    (output filename conventions, supported input URI schemes, chaining hints,
    config gotchas).
  - `JobAdapter.endpoint_examples()` returning at least one ready-to-run curl
    (and ideally a Python snippet) per service endpoint. The framework already
    flattens request body fields with required / file markers, but an example
    is what lets the agent write a *correct* first call without hallucinating.

  Together they make `/api/manifest` self-contained for an agent: content-type,
  schema refs, field list, copy-pasteable invocations. See
  `services/rfantibody-server/adapter.py` and `services/genie3-server/adapter.py`
  for worked implementations.

## Helpers for service code & tests

- **`read_version_file(__file__, default=...)`** — read a service's `VERSION`
  file (sibling of `app.py`), strip a leading `v`, fall back to `default` if
  missing. Use it in `create_app(..., version=read_version_file(__file__))`
  so the HTTP version cannot drift from the Docker image tag.

- **`bioq_service.fc_testing`** — helpers for tests that hit the deployed
  Function Compute URLs:
  - `fc_url(service_name, start=Path(__file__))` resolves the URL from
    [services/aliyun_fc_url.md](../aliyun_fc_url.md) (the single source of
    truth — update that file when you redeploy).
  - `poll_job(client, base_url, job_id, timeout_s=1800)` blocks until terminal.
  - `register_fc_marker(config)` + `skip_fc_tests_unless_enabled(config, items)`
    drop into a service's `tests/conftest.py` so `@pytest.mark.fc` tests are
    skipped by default, opted in with `pytest -m fc` or `RUN_FC_TESTS=1`.

  Pattern: `services/<svc>/tests/test_fc.py` covers smoke (health / manifest /
  openapi / 404) plus one minimal-cost job per endpoint. Fixtures (`PDB`,
  `JSON`, etc.) live in `tests/data/` — never reference `opensource/*` (it's
  gitignored). End-to-end usage and failure-mode tables in
  [engineering/guides/testing-fc-services.md](../../engineering/guides/testing-fc-services.md).
