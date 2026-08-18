# Mental Model

English | [中文](mental-model.zh.md)

> **Read when**: you need the core concepts shared by every service (job dir = unit of state, submit/poll vs task endpoint, dual-mode) before writing service code.
> **Source**: condensed from `framework/src/bioq_service/` (job runner, task endpoint, uris) and the per-service Dockerfiles; verify against the framework source when in doubt.
> **Refresh/remove when**: the framework's job lifecycle or input-resolution contract changes.

## One service = one image + HTTP endpoints + a CLI entry

- Each service is a dual-mode Docker image: HTTP (FastAPI, FC) by default, plus
  `python -m server <endpoint>` for Slurm/sbatch one-shot execution.
- Both modes share `tools.py` (argv construction), `adapter.py` (output detection), `settings.py` (config).

## Heavy dependencies live only in the Dockerfile

torch/cuda/conda/rdkit are mutually exclusive across services and cannot share one environment —
that's exactly why each service has its own Dockerfile. `pyproject.toml` declares only the
lightweight deps needed for offline tests plus the framework path source:

```toml
[tool.uv.sources]
bioq-service-framework = { path = "../../framework", editable = true }
```
(`gateway/` sits at the top level, so it uses `path = "../framework"`.)

## The job directory is the unit of state

A job lives at `<jobs_base_dir>/<job_id>/`:

- `input/` — self-contained inputs
- `output/` — products; `/download` zips this directory
- `logs/run.log` — subprocess stdout+stderr
- `job.json` — status sidecar used for restart recovery

## Two call patterns (both in one image)

1. **submit/poll** — POST returns `job_id` immediately; a background `ThreadPoolExecutor` runs the
   subprocess; clients poll `GET /api/jobs/{id}`. Use when you don't need to hold the instance.
2. **task endpoint** (`/api/tasks/<name>`) — POST blocks until the subprocess finishes. Built for
   FC async-task mode (`X-Fc-Invocation-Type: Async`) so the instance stays occupied during compute.
   The preferred entry for modern GPU services. The two are always provided in pairs.

## One input-resolution scheme

All services share a single URI scheme (details in [framework-api.md](./framework-api.md)):
multipart upload · `job://<id>/<file>` (chaining from a prior job's output) · `file:///abs/path`
or bare `/abs/path` · `oss://<bucket>/<key>` · `http(s)://...`.

## Where the concerns split

- The **adapter** owns service-wide policy (file layout, output detection, logout); it never sees
  request shapes or argv.
- Each **endpoint** owns its request model + argv construction (via `tools.py`).
- The **framework** owns the HTTP/job/CLI plumbing both of the above plug into.

That split is what lets a single image serve both HTTP and CLI modes with the same adapter.
