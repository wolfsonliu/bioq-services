# Skeleton — service source files

English | [中文](skeleton.zh.md)

> ← Back to the [Adding a service cookbook overview](./index.md)

This page covers the service's source-file skeleton: `__init__.py` / `settings.py` / `models.py` /
`adapter.py` / `app.py` (incl. task endpoint) / `__main__.py` / `pyproject.toml` / `VERSION` /
`README.md`. The Dockerfile is in [dockerfile](./dockerfile.md); tests are in [testing](./testing.md).

## 5-minute echo skeleton

Below is the minimal runnable starting point for a new service. Replace every `<svc>` with your
service name (lowercase, hyphenated) and every `<Svc>` with the CamelCase form.

### 1. `services/<svc>/__init__.py`

Empty file.

### 2. `services/<svc>/settings.py`

```python
"""Env-driven config for <svc>. All values via pydantic-settings (no os.getenv)."""

from pathlib import Path
from bioq_service import ServiceSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class <Svc>Settings(ServiceSettings):
    model_config = SettingsConfigDict(env_prefix="<SVC>_", env_file=".env", extra="ignore")

    # Override the framework default so FC deployment can use NAS layout.
    jobs_base_dir: Path = Field(default=Path("/data/<svc>_jobs"))

    # Service-specific knobs (env: <SVC>_ROOT, <SVC>_WEIGHTS_DIR, ...)
    root: Path = Field(default=Path("/opt/<svc>"))

    # Weights dir — the **unified convention** defaults to the NAS mount point /data/models/<svc>/.
    # FC auto-mounts it; SIF / local docker needs --bind /scratch/models/<svc>:/data/models/<svc>.
    # Do not bake weights into the image.
    weights_dir: Path = Field(default=Path("/data/models/<svc>"))
    # ... add what your tool needs
```

### 3. `services/<svc>/models.py`

```python
"""Per-endpoint pydantic request models. Re-export framework's JobInfo for compat."""

from bioq_service import JobInfo, JobStatus, FailureKind  # noqa: F401
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    n_samples: int = Field(default=4, ge=1, le=10000)
    seed: int | None = Field(default=None)
```

### 4. `services/<svc>/adapter.py`

```python
"""Service-wide policy: name + output detection + manifest_extras + endpoint_examples."""

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter, JobInfo, JobStatus

from .settings import <Svc>Settings


class <Svc>Adapter(JobAdapter):
    name = "<svc>"

    settings: <Svc>Settings  # narrow for IDEs

    def __init__(self, settings: <Svc>Settings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        """Tighten the default `output_dir non-empty` check to your real artifact."""
        out = self.output_dir(job_dir) / "result.txt"
        return out.exists() and out.stat().st_size > 0

    def subprocess_cwd(self) -> Path | None:
        return self.settings.root

    def manifest_extras(self) -> dict:
        """Service-specific protocol knowledge an agent needs to call this service."""
        return {
            "tool_outputs": {"generate": "output/result.txt"},
            "input_uri_schemes": {"upload": "multipart/form-data"},
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        """One copy-pasteable curl per endpoint (mandatory)."""
        return {
            "/api/generate": [
                EndpointExample(
                    title="basic generation",
                    curl="curl -X POST $URL/api/generate -F n_samples=4",
                    notes="Smallest call. See request_fields in manifest for full params.",
                ),
            ],
        }
```

### 5. `services/<svc>/app.py`

```python
"""FastAPI app + service-specific POST routes. Framework provides /healthz / /api/jobs / /api/manifest."""

from pathlib import Path
from typing import Optional

from bioq_service import JobInfo, attach_mcp, create_app, model_form_depends, read_version_file
from fastapi import Depends, File, Form, Request, UploadFile

from .adapter import <Svc>Adapter
from .models import GenerateRequest
from .settings import <Svc>Settings

settings = <Svc>Settings()
adapter = <Svc>Adapter(settings=settings)

app = create_app(
    adapter, settings,
    title="<Svc> Server",
    version=read_version_file(__file__, default="0.0.1"),
)


# ---- /healthz/detail override: report whether NAS weights are in place ----
# Services that use NAS weights **must** customize /healthz/detail so agents can
# discover a missing mount before the first inference crashes. The framework's
# default /healthz/detail is a generic disk report, and FastAPI uses first-match
# routing, so strip the framework's route before registering your own.
# FastAPI >=0.115 wraps the included router in an _IncludedRouter — strip recursively.

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
    """Weights-in-place probe: list the expected key files under weights_dir; when
    missing, return weights_loaded=false.

    Do not raise when NAS is not mounted (let the service start); expose the state
    via /healthz/detail.
    """
    expected = {
        "main_ckpt": settings.weights_dir / "main.ckpt",
        # ... list this service's key weight files
    }
    missing = {k: str(p) for k, p in expected.items() if not p.exists()}
    return {
        "status": "ok",
        "service": adapter.name,
        "version": request.app.version,
        "weights_dir": str(settings.weights_dir),
        "weights_loaded": not missing,
        "weights_missing": missing,
        "active_jobs": request.app.state.runner.active_job_count,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }


@app.post("/api/generate", response_model=JobInfo)
def generate(
    params: GenerateRequest = Depends(model_form_depends(GenerateRequest)),
    input_pdb: Optional[UploadFile] = File(None),
    input_pdb_uri: Optional[str] = Form(None),
) -> JobInfo:
    """Basic generation endpoint."""

    def _build(_job_id: str, job_dir: Path) -> list[str]:
        out = job_dir / "output"
        out.mkdir(exist_ok=True)
        return ["bash", "-c", f"echo 'generated {params.n_samples} samples' > {out}/result.txt"]

    return app.state.runner.submit(
        build_argv=_build, label="generate",
        input_params=params.model_dump(mode="json"),
    )


# Must come AFTER all POST routes so MCP auto-discovery sees the full surface.
attach_mcp(app)
```

> **Note**: when an endpoint accepts both a pydantic model (form fields) and an `UploadFile`, you
> **must** use `Depends(model_form_depends(Model))` rather than `Annotated[Model, Form()]`.
> The latter doesn't correctly expand the model's fields in FastAPI, which breaks uploading-request
> parsing. If the endpoint takes no file upload, `Annotated[Model, Form()]` also works.

#### Task endpoint (synchronous blocking, for FC async task mode)

Every submit/poll endpoint should have a corresponding `/api/tasks/<name>` task endpoint.
The two share the `argv builder` and `save_inputs` logic — they differ only in their execution model:

| | submit/poll `/api/<name>` | task endpoint `/api/tasks/<name>` |
|---|---|---|
| HTTP response | returns `JobInfo(status=pending)` immediately | returns `JobInfo(status=completed/failed)` only after finishing |
| Suited for | local / Slurm / clients that want to poll immediately | FC async task mode |
| Instance occupancy | subprocess runs in the background while HTTP has ended → FC may recycle the instance | the HTTP request lives and dies with the computation → no recycle |
| Concurrency control | controlled by the client | managed by the FC platform layer |

See [deploy.md](./deploy.md) (FC async task mode console config) for details.

**Two registration approaches:**

**(a) No file upload**: use the `register_task_endpoint` helper directly (see
`services/immunebuilder-server/app.py`):

```python
from bioq_service import register_task_endpoint

def _generate_build(req, _job_id, job_dir):
    return generate_argv(req, job_dir, settings)

register_task_endpoint(
    app,
    path="/api/tasks/generate",
    label="generate",
    request_model=GenerateRequest,
    build_argv=_generate_build,
)
```

Internally, `register_task_endpoint` checks `settings.task_endpoints_enabled` (default True) and is
a no-op when disabled.

**(b) With file upload** (most GPU services): define the endpoint yourself, call `execute_task`, and
wrap it in the `if settings.task_endpoints_enabled:` guard (see `services/boltzgen-server/app.py`):

```python
from bioq_service import execute_task, resolve_task_id
from fastapi import Header, Request

if settings.task_endpoints_enabled:

    @app.post("/api/tasks/generate", response_model=JobInfo)
    def generate_task(
        request: Request,
        params: GenerateRequest = Depends(model_form_depends(GenerateRequest)),
        input_pdb: Optional[UploadFile] = File(None),
        input_uri: Optional[str] = Form(None),
        x_bioagent_job_id: Optional[str] = Header(default=None, alias="X-Bioagent-Job-Id"),
        x_fc_async_task_id: Optional[str] = Header(default=None, alias="X-Fc-Async-Task-Id"),
    ) -> JobInfo:
        """Generate as a single atomic task. Blocks until pipeline completion."""
        job_id = resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)
        # closure-shared dict bridges upload persistence in _save to argv build in _build
        paths: dict[str, Path] = {}

        def _save(_req, input_dir: Path) -> None:
            paths["pdb"] = resolve_input(input_pdb, input_uri, input_dir / "input.pdb", settings)

        def _build(req, _job_id: str, job_dir: Path) -> list[str]:
            return generate_argv(req, paths["pdb"], job_dir, settings)

        return execute_task(
            request, job_id=job_id, label="generate", params=params,
            build_argv=_build, save_inputs=_save,
        )
```

**Key patterns and conventions:**

| Item | Convention |
|---|---|
| Endpoint path | `/api/tasks/<same-name-as-submit-poll>` |
| Dual headers | accept `X-Bioagent-Job-Id` (business) + `X-Fc-Async-Task-Id` (FC platform); priority `X-Bioagent-Job-Id > X-Fc-Async-Task-Id > UUID` |
| job_id resolution | call `resolve_task_id(...)` for uniform handling |
| Closure-shared upload path | use a dict / list rather than a magic string — `_save` writes `paths["..."]`, `_build` reads it |
| Settings guard | required `if settings.task_endpoints_enabled:` wrapper (for custom endpoints); `register_task_endpoint` carries it automatically |
| `attach_mcp(app)` | after any task endpoint registration, at end of file — so MCP can discover the new endpoint |
| Exception semantics | `build_argv` / `save_inputs` raise → framework cleanup_job + 5xx; subprocess non-zero rc → 200 + `status=failed` |

**Idempotency (dedup) — two-layer defense:**

- **FC platform layer**: a repeated invoke with the same `X-Fc-Async-Task-Id` returns HTTP 409
  Conflict and the request **does not reach the function** (built into console async task mode)
- **Framework layer**: `execute_task` checks `JobStore.get(job_id)` on entry and, if present,
  returns the existing JobInfo directly (for LocalDispatcher / curl paths that bypass the FC platform)

Both layers coexist without conflict.

### 6. `services/<svc>/__main__.py` (CLI batch entry point)

Every service needs a `__main__.py` so the same Docker image supports the
`python -m server <endpoint> ...` one-shot batch mode. The CLI mode reuses the `tools.py` argv
builder, `adapter.py` output detection, and `settings.py` config — it does not start FastAPI/uvicorn.

See [framework-api.md](../topics/framework-api.md) (CLI batch) for the detailed hook signatures.

```python
"""CLI batch-mode entry point for <svc>-server.

Usage::

    python -m server generate \
        --input-pdb /data/input.pdb \
        --output-dir /scratch/results/
"""

from __future__ import annotations

from pathlib import Path

from bioq_service.cli import CLIEndpoint, create_cli

from .adapter import <Svc>Adapter
from .models import GenerateRequest
from .settings import <Svc>Settings
from .tools import generate_argv

settings = <Svc>Settings()
adapter = <Svc>Adapter(settings=settings)


def _generate_build(req, inputs, job_dir, settings):
    return generate_argv(
        req,
        job_dir=job_dir,
        input_pdb=inputs["input_pdb"],
        settings=settings,
    )


endpoints = {
    "generate": CLIEndpoint(
        name="generate",
        help="Run generation on an input PDB",
        request_model=GenerateRequest,
        build_argv=_generate_build,
        inputs={"input_pdb": ("Input PDB file", True)},
    ),
}

create_cli(adapter, settings, endpoints, version="0.0.1")
```

**Key points**:

- **`CLIEndpoint.inputs`** declares the required input files. Each entry becomes a `--<name>` CLI
  flag; the framework automatically validates that the file exists, resolves the absolute path, and
  passes it into the `build_argv` `inputs` dict
- **The `build_argv` callback** has signature `(request, inputs, job_dir, settings) → list[str]`, the
  same logic as the `tools.py` call inside `app.py`, just wrapped in a `CLIEndpoint` shell
- **Endpoints with no file input** (e.g. immunebuilder-server receives sequence params): leave
  `inputs={}` empty — all params become argparse flags auto-generated from the pydantic model
- **Complex-type fields** (`dict`, `list[Model]`, etc. that can't be mapped to an argparse flag):
  pass them via `--params-json '{"key": value}'`. CLI flags take priority over `--params-json`
- **`inputs` / model-field name collisions**: the framework automatically skips fields already
  declared in `inputs`, so there is no duplicate argparse definition

Invocation:

```bash
# Docker — override CMD to run CLI mode
docker run --rm -v /data:/data <svc>-server \
    .venv/bin/python -m server generate \
    --input-pdb /data/input.pdb \
    --output-dir /data/results/

# Singularity / Apptainer (sbatch)
apptainer exec --nv <svc>-server.sif \
    .venv/bin/python -m server generate \
    --input-pdb /data/input.pdb \
    --output-dir /scratch/$SLURM_JOB_ID/

# --params-json suits scripted calls / complex params
python -m server generate \
    --input-pdb input.pdb \
    --params-json '{"n_samples": 10, "seed": 42}' \
    --output-dir ./output/
```

### 7. `services/<svc>/pyproject.toml` (optional)

`pyproject.toml` declares only the pip dependencies needed for **offline tests** + the framework path
dependency — heavy runtime deps (torch/cuda/conda/rdkit etc.) live in each Dockerfile. Each service
depends on the framework via a relative editable path, no publishing needed:

```toml
[project]
name = "<svc>-server"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = [
    "bioq-service-framework",
    # lightweight pip deps for offline tests (not the heavy algorithm deps)
]

# relative path to the framework, no PyPI publishing needed
[tool.uv.sources]
bioq-service-framework = { path = "../../framework", editable = true }

# offline test environment only — don't build this service's package
[tool.uv]
package = false

[dependency-groups]
dev = ["pytest", "pytest-asyncio"]
```

Run offline unit tests: `cd services/<svc> && uv run --group dev python -m pytest tests/ -q`.

**Exception — when the server code needs `pip install -e .`** (in the uv venv skeleton the Dockerfile
uses `uv pip install -e .`): drop `package = false` and add back the build-backend declaration —

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
```

**If the server code is injected only via `COPY` and all algorithm deps are resolved in a conda env
or a separate pip install, you can skip pyproject.toml** (deeprank-ab-server, jwt, etc. do this).


### 9. `services/<svc>/VERSION`

```
v0.0.1
```

`Makefile` reads this file as the image tag. To release a new version:
`echo v0.0.2 > services/<svc>/VERSION`.


### 13. `services/<svc>/README.md`

At minimum include:
- A top architecture diagram (client → FastAPI + framework → subprocess → NAS)
- A curl example per endpoint (kept in sync with `endpoint_examples()`)
- A config table (all env vars + defaults)
- Local-dev commands + Docker build + FC deploy sub-section

See [services/rfantibody-server/README.md](../../services/rfantibody-server/README.md) or
[services/genie3-server/README.md](../../services/genie3-server/README.md).