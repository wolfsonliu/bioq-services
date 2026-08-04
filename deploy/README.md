# deploy — bioq-services deployment targets

Three deployment paths for the gateway + workers, one shared config layer. Pick the
one that matches where you're running:

| Dir | Target | Dispatch / storage | Run it |
|---|---|---|---|
| [`ecs/`](ecs/) | Production single host (ECS) + Alibaba **FC** workers | `fc` / `oss` | on the ECS host: `cd deploy/ecs && ./deploy.sh` |
| [`compose/`](compose/README.md) | Local full-stack, single machine, no FC/OSS | `http` / `file` | `cd deploy/compose && docker compose up -d` |
| [`openfaas/`](openfaas/README.md) | Local kind + OpenFaaS (elastic, scale-to-zero) | `openfaas` / `file` | from the repo root: `make local-up` |
| [`config/`](config/README.md) | Shared **non-secret** topology (checked in) | — | consumed by all three (not run) |

**Why `ecs`/`compose` run in their own dir but `openfaas` runs via `make`:** the
first two are `docker compose up -d` with a dir-local `.env` + relative mounts —
you operate them on the target host, in the dir. OpenFaaS orchestration (create
kind cluster, install OpenFaaS, load images, ConfigMap/Secret, port-forward,
Keycloak user/service-account management) is a tight local dev loop, so it lives as
`make local-*` targets in the repo-root `Makefile` (`make help` lists them).

## Config: 3 layers, one schema

All targets feed the same `GATEWAY_*` pydantic-settings schema
(`gateway/settings.py`). Precedence (later wins):

1. **Defaults** — `gateway/settings.py` + Dockerfile `ENV`.
2. **Non-secret topology** — `config/gateway.common.env` + `config/gateway.<target>.env` (checked in).
3. **Secrets + site overrides** — each target's gitignored `.env` (compose/ecs) or
   `.env.local` (openfaas). Secrets (OSS AK/SK, `ALI_SK`, Postgres pw, OIDC client
   secret, session secret, external DB URL) live **only** here — never in `config/`.

Design rationale: monorepo ADR `engineering/decisions/2026-08-04-deploy-config-layering.md`.

## Auth / IdP

Every target authenticates via OIDC/JWT (Keycloak) with a VPC/localhost bypass for
internal callers. `compose` and `openfaas` bundle Keycloak by default; `ecs` can
also bundle it as a quickstart via the optional `idp` compose profile
(`COMPOSE_PROFILES=idp` — see `ecs/.env.example`), or point at a managed/corp IdP.
The bundled realm ships **throwaway dev credentials** — harden before production.
