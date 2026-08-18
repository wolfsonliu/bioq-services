# Adding a Service

English | [中文](adding-a-service.zh.md)

> **Read when**: you add a new service end-to-end in this repo.
> **Source**: points at the authoritative cookbook (currently Chinese) — [`../adding-a-new-service/index.zh.md`](../adding-a-new-service/index.zh.md) and its sub-pages.
> **Refresh/remove when**: the cookbook structure changes.

## Flow (overview)

1. **Write the design doc first** (before any code): a `YYYY-MM-DD-<svc>-server-design.md` archived in
   [`../specs/`](../specs/), with the required sections listed in
   [`../adding-a-new-service/index.zh.md`](../adding-a-new-service/index.zh.md#0-先写设计文档开工前必做).
2. **Build the skeleton** from the cookbook sub-pages:
   [`skeleton`](../adding-a-new-service/skeleton.zh.md) ·
   [`dockerfile`](../adding-a-new-service/dockerfile.zh.md) ·
   [`conda-pitfalls`](../adding-a-new-service/conda-pitfalls.zh.md) ·
   [`testing`](../adding-a-new-service/testing.zh.md) ·
   [`deploy`](../adding-a-new-service/deploy.zh.md);
   naming / required files / verification / registration / checklist are all in
   [`index.zh.md`](../adding-a-new-service/index.zh.md).
   File layout + starter references: [service-anatomy.md](./service-anatomy.md).
3. **Register + wire through**: add a `<svc>-server:` entry to `services.yaml` (add `oss_mount: true`
   for file-input services), and a `TestEndToEnd<Svc>` e2e class in `gateway/tests/test_fc.py` when
   the service is called through the gateway.
4. **Pass the hard constraints** (`AGENTS.md`) and the submission checklist in
   [`../adding-a-new-service/index.zh.md`](../adding-a-new-service/index.zh.md).

## Quick verification (before committing)

```bash
cd services/<svc>-server && uvx ruff check . && uv run --group dev python -m pytest tests/test_app.py tests/test_cli.py -q
./scripts/vendor.sh && ls upstream/ | head
cd ../.. && make build-<svc>-server
```

## Related

- Conventions rationale: [conventions.md](./conventions.md)
- Build/verify loop: [build-deploy.md](./build-deploy.md) · [testing.md](./testing.md)