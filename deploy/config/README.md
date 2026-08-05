# deploy/config — non-secret gateway config (generated, checked in)

One **complete** non-secret config file per deploy target, **generated from the
`gateway/settings.py` schema** so every knob and its default is visible in one
place. Secrets never live here. See the ADRs
`engineering/decisions/2026-08-04-deploy-config-layering.md` (layering) and
`engineering/decisions/2026-08-05-gateway-config-generation.md` (generation).

- `gateway.ecs.env` / `gateway.compose.env` / `gateway.openfaas.env` — each lists
  **every** operator-relevant `GATEWAY_*` knob with its default + a doc comment,
  with that target's values (fc/oss vs http/file vs openfaas) baked in. Secrets
  appear only as **commented placeholders**.

## Editing

**Do not hand-edit these files' structure** — they are generated. To change a
value that is a per-target default, edit `gateway/config_spec.py` (`PROFILES` /
`COMMON` / `DOCS`) and regenerate:

```bash
make gen-config     # (re)write all three files
make check-config   # CI gate: fail if a committed file is stale
```

For a **one-off site value** (e.g. a different `GATEWAY_OSS_BUCKET`), don't edit
here — set it in the target's gitignored `.env` (it wins). Secrets always go in
`.env` / `.env.local`.

## How each target consumes it

- docker compose (`ecs`, `compose`): `env_file: [../config/gateway.<t>.env, .env]`
  — `.env` last, so secrets/site overrides win.
- kind (`openfaas`): a `gateway-config` ConfigMap built from `gateway.openfaas.env`
  + a `gateway-secrets` Secret from `.env.local`, `envFrom` with the Secret last.

Precedence (later wins): schema defaults → this file → `.env`/`.env.local`.

Note (compose only): `${VAR}` interpolation in a `docker-compose.yml` reads the
compose dir's `.env`, **not** `env_file:` — so image tags / ports / host paths /
`POSTGRES_PASSWORD` / `COMPOSE_PROFILES` live in that `.env`, while `GATEWAY_*`
container config comes from these generated files.

## Inspecting the effective config

Inside a running gateway, `python -m server config` prints the **resolved** config
(secrets redacted, default/overridden labelled) and `python -m server check`
validates it — the same fail-fast validation the gateway runs at startup.
