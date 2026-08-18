# Conventions & Constraints Rationale

English | [中文](conventions.zh.md)

> **Read when**: you want the *why* behind a hard constraint, or the full checklist before submitting a service change.
> **Source**: accumulated from cross-service bugs (the 422 URI-field mismatch, the bind-mount output-sink fix) and the FC/NAS deployment model; the numbered items match the 15 rules in `AGENTS.md`.
> **Refresh/remove when**: a rule's underlying framework/contract changes — then delete it, don't let it linger.

The terse list lives in [`../../AGENTS.md`](../../AGENTS.md#hard-constraints). The complete change
checklist is in [`../adding-a-new-service/index.zh.md`](../adding-a-new-service/index.zh.md).

## Why / when / remove — per rule

1. **pydantic-only types, no `dict[str, Any]` request body.**
   *Why*: the framework derives manifest, CLI argparse flags, and OpenAPI from pydantic schemas; a
   `dict[str, Any]` body defeats validation and auto-generated flags. *When*: defining any request/
   response model. *Remove when*: the framework switches to a different schema system.

2. **Config via `pydantic-settings`, no `os.getenv`.**
   *Why*: one validated config source of truth; scattered `os.getenv` hides config and breaks tests.
   *When*: adding or reading runtime config. *Remove when*: the framework abandons pydantic-settings.

3. **Fixed names: `bioq_service` / `bioq-service-framework` / `X-Bioagent-*` headers.**
   *Why*: import and distribution names were fixed when services moved into this repo; the headers are
   a cross-service historical contract read by clients and the task endpoint. *When*: writing imports,
   `pyproject.toml`, or touching HTTP headers. *Remove when*: a coordinated repo-wide rename lands.

4. **Build on `framework/`, don't re-implement common layers.**
   *Why*: HTTP/job lifecycle/persistence/manifest/CLI/upload-download are solved once; forking them
   breaks consistency. *When*: starting a new service. *Remove when*: a shared framework replaces it.

5. **Paired task endpoint, guarded by `if settings.task_endpoints_enabled:`.**
   *Why*: FC async-task mode needs a blocking endpoint to keep the instance occupied; the guard lets the
   same image serve non-FC deploys. *When*: adding any submit/poll endpoint. *Remove when*: FC
   async-task mode is no longer a target.

6. **URI fields named exactly `<upload_field>_uri`.**
   *Why*: the client's `--file <field>=<path>` maps an upload field to `<field>_uri`; a mismatch makes
   FastAPI drop the field, leaving `upload=None, uri=None` → 422. *When*: defining a file/URI dual
   input. *Remove when*: the mapping convention changes (see
   `../specs/2026-08-18-cross-service-uri-field-naming-design.md`).

7. **Weights on NAS; `/healthz/detail` reports `weights_loaded`, no import-time raise.**
   *Why*: weights are large and shared — baking them bloats images; an import-time raise kills the
   probe. *When*: adding or relocating model weights. *Remove when*: a service legitimately needs small
   in-image weights (<100 MB, documented in a comment).

8. **Vendor upstream at a pinned SHA; no in-image `git clone`, no `COPY opensource/`.**
   *Why*: reproducible, offline builds; `COPY opensource/` is a legacy path. *When*: writing
   `vendor.sh` / `Dockerfile`. *Remove when*: the build abandons Docker or the vendoring scheme changes.

9. **Install framework via `COPY framework /tmp/service-framework` + `pip install`.**
   *Why*: a bind-mount doesn't ship into the image, so runtime fixes (e.g. output-sink) would be missing
   in production. *When*: installing the framework in a `Dockerfile`. *Remove when*: the framework is
   pinned and published to a package registry instead.

10. **Per-service `VERSION`, independent releases.**
    *Why*: services evolve on independent cadences; global tagging forces coupling. *When*: cutting a
    release. *Remove when*: release policy changes globally.

11. **`manifest_extras` (`tool_outputs` + `input_uri_schemes`); `endpoint_examples` ≥1 curl.**
    *Why*: agents/clients call services from the manifest without reading source. *When*: finishing a
    service or changing an endpoint. *Remove when*: manifest consumers go away.

12. **Endpoints use `Depends(model_form_depends(Model))`.**
    *Why*: multipart form parsing needs the dependency; a bare `params: Model` breaks file/form parsing.
    *When*: adding any POST endpoint with form params. *Remove when*: FastAPI form handling changes.

13. **No `from __future__ import annotations` in `framework/src/bioq_service/task_endpoint.py`.**
    *Why*: PEP 563 string annotations break FastAPI `get_type_hints` on runtime classes. *When*: editing
    that file. *Remove when*: FastAPI supports PEP 563.

14. **`upstream/` and `weights/` are git-ignored.**
    *Why*: they're vendor/download build artifacts. *When*: any new service. *Remove when*: never.

15. **Repo-root-relative paths only.**
    *Why*: avoids broken references wherever the repo is checked out. *When*: writing docs/config.
    *Remove when*: never.

## Other norms

- **Docs & comments in Chinese; code/identifiers/commits in English** (matches existing README/docs).
  `AGENTS.md` and the English `docs/topics/*.md` are the deliberate exception — plus their `*.zh.md`
  companions.
- **No AI co-author trailers** (e.g. `Co-Authored-By`) in commits.
- **Protocol changes are self-contained**: a change to an endpoint signature / upload field / manifest
  must update that service's `README.md` and `endpoint_examples()`, and (where useful) add a manifest
  regression test (see rfantibody's `test_quiver_uri_field_matches_upload_field`).
- **Need the framework's full behavior?** Read `framework/src/bioq_service/` (thorough docstrings) and
  the matching `framework/tests/`.
