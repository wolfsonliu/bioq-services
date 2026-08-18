# Repository Layout

English | [中文](repository-layout.zh.md)

> **Read when**: you need the directory map, what each top-level layer owns, or the meaning of fields in `services.yaml`.
> **Source**: distilled from the repo's actual top-level tree and `services.yaml` (both readable directly); re-derive from the tree if this drifts.
> **Refresh/remove when**: the directory structure or the `services.yaml` schema changes enough that this map misleads.

## Top-level tiers

```
framework/   — shared service framework: a library, no Dockerfile, ships no image.
│              PyPI name bioq-service-framework; import name bioq_service.
gateway/     — control plane: auth / upload negotiation (OSS presign) / async dispatch /
│              status & download proxy (ECS, docker-compose / kind).
edge/        — non-worker edge components: jwt/ (JWT signing), protein-design-mcp/ (MCP adapter).
services/    — compute workers, one FC/OpenFaaS function image each, keep the -server suffix.
deploy/      — deploy targets: ecs/ (prod ECS+FC), compose/ (local full-stack),
│              openfaas/ (local kind+OpenFaaS), config/ (generated shared non-secret config).
docs/        — adding-a-new-service/ (cookbook) + specs/ (design docs) + topics/ (this bilingual topic set).
services.yaml   — fleet registry (url / tier / function / oss_mount per deployed service).
Makefile        — build/push/bump/SIF + local kind integration (make local-*).
scripts/        — misc scripts (e.g. bench_concurrency.py).
```

## Docs organization

| dir | purpose |
|---|---|
| `docs/topics/` | flat topic set — one topic per bilingual pair (`<topic>.md` + `<topic>.zh.md`) |
| `docs/adding-a-new-service/` | multi-page cookbook — `index.md` + sub-pages |
| `docs/specs/` | dated `YYYY-MM-DD-*` design docs |
| `docs/plans/` | dated plan / decision notes |

**Flat vs subdirectory:** keep a topic flat under `docs/topics/` unless it genuinely splits into
several sub-pages. Only then promote it to `docs/<area>/` with an `index.md` (each page `.md` +
`.zh.md`). `docs/adding-a-new-service/` is the sole such example today.

## Workers (38)

alphafold / bindflow / boltz / boltzgen / chembounce / deeprank-ab / diamond / diffdock /
diffdock-pp / diffusion-hopping / dockq / drughive / ensemble / esmfold2 / flowmol / genie3 /
haddock3 / iggm / immunebuilder / lasermpnn / lightdock / megalodon / mmseqs2 / odesign /
openadmet / openbpmd / plip / pocketxmol / ppiflow / promera / proteinmpnn / qligfep / reinvent /
rfantibody / rfdiffusion / rfdiffusion2 / semlaflow / turbohopp.

## services.yaml semantics

The authoritative list of deployed services. Per entry, only `url` is required:

| field | meaning |
|---|---|
| `url` | VPC HTTP trigger URL |
| `region` | Aliyun region (default `cn-hangzhou`) |
| `tier` | `hot` keep-warm / `warm` scale-to-zero (default) / `cold` batch-only |
| `function` | FC function name the gateway uses for async status polling |
| `gpu` | GPU card class (optional, e.g. `fc.gpu.tesla.1`) |
| `oss_mount: true` | service takes file input → gateway rewrites `oss://` input to `/mnt/oss/...` |

Undeployed services appear as **comment entries** (placeholders).

## Where to look next

- Concepts that don't depend on the tree: [mental-model.md](./mental-model.md)
- Per-service file layout: [service-anatomy.md](./service-anatomy.md)
- Control-plane internals: [gateway.md](./gateway.md)
