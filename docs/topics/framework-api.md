# Framework API

English | [中文](framework-api.zh.md)

> **Read when**: you write endpoint / adapter / settings / CLI code against the framework, or need the exact signature of a hook.
> **Source**: `framework/src/bioq_service/` — each module's docstring and `framework/tests/` are the authoritative details; this page is a quick index.
> **Refresh/remove when**: a framework API changes; this page should then shrink further, not grow (prefer pointing at the source).

Import name is **`bioq_service`** (`from bioq_service import ...`); distribution name is
`bioq-service-framework`.

## JobAdapter (`adapter.py`)

One `JobAdapter` subclass per service holds the service-wide policy (file layout, output detection,
logging, subprocess env/cwd, restart recovery). It does **not** know request shapes / argv (that's
per-endpoint). Overridable hooks:

| Hook | Default | Purpose |
|---|---|---|
| `name: str` | required | service name; used by `JobInfo.service` |
| `job_dir(job_id)` | `<jobs_base_dir>/<job_id>` | job working directory |
| `output_dir(job_dir)` | `<job_dir>/output` | directory `/download` zips and `/files` lists |
| `log_path(job_dir)` | `<job_dir>/logs/run.log` | where subprocess stdout+stderr is tee'd |
| `detect_outputs(job_dir)` | `output/` non-empty | rc==0 but returns False → FAILED (`failure_kind=NO_OUTPUTS`). Multi-endpoint services should override to recognize their tools' products |
| `subprocess_env()` | `{}` | extra env injected into the subprocess |
| `subprocess_cwd()` | None | subprocess working directory |
| `infer_job_from_dir(job_dir)` | has outputs → COMPLETED | recovery heuristic for legacy dirs without `job.json` |
| `manifest_extras()` | `{}` | **override strongly recommended**: at least `tool_outputs` + `input_uri_schemes` |
| `endpoint_examples()` | `{}` | **override strongly recommended**: ≥1 copy-paste curl per endpoint (python snippet also nice) |

## ServiceSettings (`settings.py`)

All runtime config is a `pydantic_settings.BaseSettings` subclass — **no `os.getenv`** in the
framework or adapters. Give each service its own `env_prefix`:

```python
class DockQSettings(ServiceSettings):
    model_config = SettingsConfigDict(env_prefix="DOCKQ_", env_file=".env", extra="ignore")
```

Base fields (env `<PREFIX>_*`; defaults in `framework/src/bioq_service/settings.py`): `jobs_base_dir`
(`/data/jobs`), `uploads_base_dir` (`/data/uploads`), `oss_output_mount` (`/mnt/oss`), `oss_region`
(`cn-hangzhou`), `disk_limit_mb` (`8000`, evict finished jobs past it), `port` (`9000`),
`max_concurrent_jobs` (`2`, 503 beyond it), `keep_alive_sec`, `keepalive_interval_s` / `keepalive_url`
(FC self-keepalive), `session_header_name` (FC session affinity), `task_endpoints_enabled` (True),
`task_job_id_header` (`X-Bioagent-Job-Id`).

## Assembling the app (`app.py`)

```python
settings = DockQSettings()
adapter = DockQAdapter(settings=settings)
app = create_app(adapter, settings, title="...", version=read_version_file(__file__))
```

`create_app` mounts the common routes (healthz / manifest / openapi / jobs family) and exposes
`adapter` / `settings` / `job_store` / `runner` on `app.state`. A service only adds its own POST
endpoints:

- receive form params via `params: Model = Depends(model_form_depends(Model))`;
- trigger submit/poll via `app.state.runner.submit(build_argv=_build, label="...", input_params=...)`;
- optionally, after all POST routes, call `attach_mcp(app)` to mirror the HTTP surface at `/mcp`
  (needs `bioq-service-framework[mcp]`).

`read_version_file(__file__, default=...)` reads the sibling `VERSION`, strips a leading `v`, so the
HTTP version never drifts from the image tag.

## Task endpoint (`task_endpoint.py`)

- `resolve_task_id(x_bioagent_job_id, x_fc_async_task_id)` — header takes priority, else a UUID.
- `execute_task(request, *, job_id, label, params, build_argv, save_inputs=None, oss_prefix=None)`
  — runs the full pipeline synchronously (idempotent dedup, disk eviction, PENDING→RUNNING→finalize,
  failure = 200 + FAILED, OSS output-sink attempted on success and failure). Endpoints with
  `UploadFile` / custom Form use this and hand-write their handler.
- `register_task_endpoint(app, *, path, label, request_model, build_argv, save_inputs=None)` — a
  convenience wrapper for no-upload scenarios.

**Do not** add `from __future__ import annotations` to this module (PEP 563 string annotations break
FastAPI's `get_type_hints`); keep it that way when editing.

## CLI batch (`cli.py`)

Each service's `__main__.py`:

```python
endpoints = {"score": CLIEndpoint(name="score", help="...", request_model=ScoreRequest,
                                  build_argv=_score_build, inputs={"model": ("help", True)})}
create_cli(adapter, settings, endpoints, version="0.0.1")
```

Each `CLIEndpoint.inputs` key becomes a `--<name>` local-file flag; `build_argv(req, inputs, job_dir,
settings)` is a thin callback to `tools.py`. `create_cli` generates argparse flags from the pydantic
model, resolves inputs, runs the subprocess, and picks the exit code from rc + `detect_outputs`.

## Input resolution (`uris.py`)

| scheme | semantics |
|---|---|
| multipart upload | client sends the file directly |
| `job://<id>/<file>` | pull `output/` from a prior job on the same NAS (chaining) |
| `file:///abs/path` or bare `/abs/path` | copy a NAS-local path |
| `oss://<bucket>/<key>` | download via the OSS SDK (needs `alibabacloud-oss-v2`) |
| `http(s)://...` | stream from a remote URL (incl. OSS signed URLs) |

API: `resolve_input(upload, input_uri, dest, settings, field_name=None)` (URI wins if both given;
both missing → 422; `field_name` locates the missing field in the 422 detail),
`maybe_resolve_input(...)` (returns None when both missing — for inline SMILES/sequence cases),
`resolve_uri(uri, dest, settings)`, `save_upload(upload, dest)`.

## Common endpoints (auto-registered)

`GET /healthz`, `GET /healthz/detail`, `GET /api/manifest`, `GET /openapi.json`,
`GET /api/jobs/{id}`, `GET /api/jobs/{id}/files`, `GET /api/jobs/{id}/log`,
`GET /api/jobs/{id}/download`, `GET /api/jobs/{id}/file/{path}`, `DELETE /api/jobs/{id}`.
