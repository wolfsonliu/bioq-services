# AGENTS.md — bioq-services

Guidance for agents developing inside this repository. This file is intentionally short and
routing-oriented: it holds only what every task needs. Detailed guidance lives in the `docs/topics/`
topic documents — load one only when its trigger applies. Each topic document has an English
original (`*.md`) and a Chinese version (`*.zh.md`). All paths are relative to the repository root;
never reference paths outside this repository.

---

## Project Overview

An AI drug discovery (AIDD) algorithm-service fleet plus a shared service framework. Each
`services/<name>-server/` wraps a third-party bioinformatics/AIDD tool as a **dual-mode Docker
image** — a FastAPI HTTP service deployed on Alibaba Cloud FC, and a `python -m server <endpoint>`
CLI batch mode for Slurm/sbatch — while `framework/` supplies the shared HTTP / job-lifecycle /
persistence / manifest / CLI / upload-download layers.

## Quick Start

There is no global `make setup` or `make test` (each component is its own uv project). First-run
orientation and verification:

```bash
make help                 # every make target, explained
make list                 # discovered deployments (image names)
make local-up             # control plane + one worker locally (kind + OpenFaaS)

# offline tests, per layer (run inside each directory)
cd services/<svc>-server && uv run --group dev python -m pytest tests/ -q
cd framework              && uv run --extra dev python -m pytest tests/ -q
cd gateway                && uv run --with pytest --with pytest-asyncio python -m pytest tests/ -v
```

## Hard Constraints

Non-negotiable. Full rationale (why / when it applies / when it can be removed) and the complete
change checklist are in [`docs/topics/conventions.md`](docs/topics/conventions.md) and
[`docs/adding-a-new-service/index.zh.md`](docs/adding-a-new-service/index.zh.md).

1. All request/response types are `pydantic.BaseModel`; never use `dict[str, Any]` as a request body.
2. Runtime config goes through `pydantic-settings` only — no `os.getenv` in `settings.py`, `framework/`, or adapters.
3. Naming is fixed: import `bioq_service`, distribution `bioq-service-framework`; legacy `X-Bioagent-*` HTTP headers stay unchanged.
4. Build every service on `framework/` — never re-implement HTTP / job lifecycle / persistence / manifest / CLI / upload-download.
5. Every submit/poll endpoint has a paired `/api/tasks/<name>` task endpoint, registered under `if settings.task_endpoints_enabled:`.
6. URI upload fields must be named exactly `<upload_field>_uri` (e.g. `model` → `model_uri`); a mismatch silently produces a 422.
7. Weights live on NAS `/data/models/<svc>/`, not baked into images; `/healthz/detail` reports `weights_loaded` and never raises at import time.
8. Vendor upstream into `services/<svc>-server/upstream/` at a pinned SHA via `scripts/vendor.sh`; the Dockerfile COPYs from `upstream/` — no in-image `git clone`, no `COPY opensource/`.
9. Install the framework with `COPY framework /tmp/service-framework` (COPY, not bind-mount) + `pip install` (or `uv pip install`).
10. Each service releases independently from its own `VERSION` (`make bump-<svc>`); tags are never coordinated globally.
11. `manifest_extras()` provides `tool_outputs` + `input_uri_schemes`; `endpoint_examples()` provides ≥1 runnable curl per endpoint.
12. Endpoints receive form params via `Depends(model_form_depends(Model))` — never a bare `params: Model`.
13. Do not add `from __future__ import annotations` to `framework/src/bioq_service/task_endpoint.py` (PEP 563 string annotations break FastAPI `get_type_hints`).
14. `upstream/` and `weights/` are git-ignored build artifacts — never commit them.
15. Every path in docs and config is repo-root-relative; never reference paths outside this repository.

## Topic Docs

One-line pointer plus the condition under which to load it.

| Topic | Load it when… |
|---|---|
| [repository-layout](docs/topics/repository-layout.md) | you need the directory map, layer responsibilities, or `services.yaml` registry semantics. |
| [mental-model](docs/topics/mental-model.md) | you need the core concepts (job dir = unit of state, submit/poll vs task endpoint, dual-mode). |
| [service-anatomy](docs/topics/service-anatomy.md) | you create or modify files inside a `services/<svc>-server/`. |
| [framework-api](docs/topics/framework-api.md) | you write endpoint / adapter / settings / CLI code against the framework. |
| [conventions](docs/topics/conventions.md) | you need the full rationale (why / when / removal) and checklist behind the hard constraints. |
| [gateway](docs/topics/gateway.md) | you change the control plane under `gateway/`. |
| [testing](docs/topics/testing.md) | you run or write tests at any layer. |
| [build-deploy](docs/topics/build-deploy.md) | you build / tag / push / bump / SIF an image. |
| [local-dev](docs/topics/local-dev.md) | you use the local kind + OpenFaaS stack (`make local-*`). |
| [adding-a-service](docs/adding-a-new-service/index.zh.md) | you add a new service end-to-end. |

## Everyday Norms

- **Code, identifiers, and commit messages in English; prose docs and code comments in Chinese** (matching the existing README/docs). This `AGENTS.md` and its English topic documents are the deliberate exception.
- Do not add AI co-author trailers (e.g. `Co-Authored-By`) to commits.
- A change that touches an endpoint signature / upload field / manifest must update that service's `README.md` and `endpoint_examples()` in the same change.