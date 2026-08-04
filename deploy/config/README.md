# deploy/config — non-secret gateway config (checked in)

Layered configuration shared by all three deploy targets (`../ecs`, `../compose`,
`../openfaas`). See the design rationale in the monorepo ADR
`engineering/decisions/2026-08-04-deploy-config-layering.md`.

Three layers, one schema (`gateway/settings.py`, `GATEWAY_*`):

1. **Defaults** — `gateway/settings.py` + Dockerfile `ENV` (stable paths).
2. **Topology / presentation (non-secret)** — the files here:
   - `gateway.common.env` — values identical across all targets.
   - `gateway.ecs.env` / `gateway.compose.env` / `gateway.openfaas.env` — per-target.
3. **Secrets + site overrides** — each target's gitignored `.env` (compose/ecs) /
   `.env.local` (openfaas). **Never put secrets in this folder.**

Precedence (later wins): defaults → common → per-target → `.env`. Injection:
- docker compose (`ecs`, `compose`): `env_file:` list, `.env` last.
- kind (`openfaas`): a `gateway-config` ConfigMap (these files) + a
  `gateway-secrets` Secret (`.env.local`), `envFrom` with the Secret last.

Note (compose only): `${VAR}` interpolation in a `docker-compose.yml` reads the
compose dir's `.env`, **not** `env_file:` — so image tags / ports / host paths /
`POSTGRES_PASSWORD` live in that `.env`, while `GATEWAY_*` container config comes
from these files.
