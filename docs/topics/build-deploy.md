# Build & Deploy Images

English | [中文](build-deploy.zh.md)

> **Read when**: you build / tag / push / bump / SIF an image, or cut a release.
> **Source**: `Makefile` (auto-discovery + per-service versioning) and `docs/adding-a-new-service/` verification flow.
> **Refresh/remove when**: the Makefile targets or versioning policy change.

The `Makefile` auto-discovers buildable images across layers (`services/*/Dockerfile` +
`gateway/Dockerfile` + `edge/*/Dockerfile`; `framework/` has none so it's skipped). Image name = the
last directory segment (workers keep `-server`). Build context is the **repo root**
(`docker build -f <svc-dir>/Dockerfile .`).

```bash
make help                     # every target, explained
make list                     # list discovered services (image names)
make version                  # print each service's current tag
make build-<service>          # build one image at its VERSION
make build-<svc> TAG=v0.0.5   # override the tag for one build
make push-<service>           # build + tag + push to harbor (REGISTRY overridable)
make bump-<service>           # patch version +1 (v0.0.5 → v0.0.6)
make sif-<service>            # Docker → Apptainer SIF (HPC/Slurm)
make login-harbor             # docker login harbor.ruosheng.bio (before first push)
make clean-<service>          # remove one local image
make clean                    # remove all local service images
```

## Versioning

Each service releases independently. Tag priority:

1. `TAG=vX.Y.Z` CLI override (wins over everything)
2. `<svc-dir>/VERSION` file (the normal case)
3. `git describe --tags --always --dirty` (unversioned local builds only)

Never coordinate tags globally.

## SIF

`make sif-<service>` needs `apptainer` on PATH; output lands in `SIF_DIR` (default `sif/`). Clean with
`make clean-sif-<service>` / `make clean-sif`.

## After adding / changing a service

Run the verification checklist in [`../adding-a-new-service/index.zh.md`](../adding-a-new-service/index.zh.md)
(vendor → local docker build → `/api/manifest` sanity → `python -m server --help` → task-route sanity),
then after FC deploy run `test_fc` / `test_fc_task` (see [testing.md](./testing.md)).

## Related

- Local kind deployment: [local-dev.md](./local-dev.md)
- Full new-service flow: [adding-a-new-service cookbook](../adding-a-new-service/index.zh.md)
