# gateway-server

Persistent API gateway (ECS) fronting the FC GPU services. Single public
entry: auth + user/key management (SQLite) + presigned-OSS upload + async
dispatch to downstream + status/download proxy. See design + plan:
`engineering/decisions/2026-07-09-unified-service-access-cli.md`,
`engineering/decisions/2026-07-09-gateway-server-plan.md`.

## Endpoints (`/v1/*`)
| Method | Path | Purpose |
|---|---|---|
| GET | /v1/services | list downstream services |
| GET | /v1/services/{svc} | manifest + openapi |
| POST | /v1/run/{svc}/{endpoint} | submit (async) → job_id |
| GET | /v1/jobs/{job_id} | status |
| GET | /v1/jobs/{job_id}/download | result zip |
| POST | /v1/jobs/{job_id}/cancel | cancel (MVP: local mark) |
| POST | /v1/uploads/presign | presigned OSS PUT for a large input |
| GET | /healthz | health |

## Auth
Three-layer: VPC bypass (internal) → JWT (`Authorization: Bearer`) → API key
(`X-API-Key`, looked up in the DB). `key_id` = principal.

## Local dev
```bash
GATEWAY_DB_URL=sqlite:///$PWD/gw.db \
GATEWAY_REGISTRY_PATH=$PWD/services/aliyun_fc_url.md \
uv run python -m uvicorn server.app:app --port 9000
```

## Deploy (ECS)
Deploy assets live in `deploy/` (docker-compose, persistent `/data` volume,
`restart: always`, healthcheck). On the ECS host:

```bash
cd services/gateway-server/deploy
cp .env.example .env          # fill in OSS creds, bucket, optional JWT JWKS URL
./deploy.sh --build           # build image from repo root, then `docker compose up -d`
```

`.env` supplies secrets + config: `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET`,
`GATEWAY_OSS_BUCKET`, `GATEWAY_OSS_REGION`, and (for external JWT access)
`GATEWAY_AUTH__JWT_JWKS_URL`. `.env` is gitignored. The image already sets the
DB/registry/session defaults, so those need not be repeated.

## Users & API keys
Users and API keys live in the gateway's SQLite DB (`/data/gateway/gateway.db`,
tables `users` / `api_keys`). `key_id` **is** the principal jobs are owned by;
the secret is stored only as a sha256 hash. There is no admin UI (MVP) — seed
keys with the stdlib-only helper `scripts/seed_key.py`.

The gateway must have started at least once (so it created the schema) before
seeding. All commands below run against the bind-mounted DB file (default
`deploy/data/`).

**Create a new user** — there is no separate "create user" step: a user is
created as a side effect of issuing their first key (a user with no key can't
authenticate). Just pass a new `--principal`:

```bash
python services/gateway-server/scripts/seed_key.py \
    --db services/gateway-server/deploy/data/gateway/gateway.db \
    --principal alice
# inserts user "alice" (INSERT OR IGNORE) + a new API key;
# prints key_id + secret  (store the secret — only its sha256 hash is persisted)
```

**Add another key to an existing user** (rotation / per-client keys) — same
principal, a distinct `--key-id`:

```bash
python services/gateway-server/scripts/seed_key.py \
    --db services/gateway-server/deploy/data/gateway/gateway.db \
    --principal alice --key-id gk_alice_ci
# user row is INSERT OR IGNORE'd (no-op); a second key is added
```

Options: `--secret` (default: random), `--key-id` (default: `gk_<random>`),
`--display-name`. Then authenticate:

```bash
curl -H "X-API-Key: <secret>" https://<gateway-host>/v1/services
```

Internal (VPC) callers hitting the gateway's `*-vpc.fcapp.run` / localhost host
are auto-bypassed and need no key.

## Tests
```bash
uv run python -m pytest services/gateway-server/tests/ -v
```
