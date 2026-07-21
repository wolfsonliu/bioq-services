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
# create/upgrade the schema first (alembic; app startup no longer create_all()s)
cd services/gateway-server
GATEWAY_DB_URL=sqlite:///$PWD/gw.db uv run alembic upgrade head
cd -

GATEWAY_DB_URL=sqlite:///$PWD/gw.db \
GATEWAY_REGISTRY_PATH=$PWD/services/services.yaml \
uv run python -m uvicorn server.app:app --port 9000
```

## Database & migrations
The user/credential + job store is SQLAlchemy. The **docker-compose deployment
bundles a PostgreSQL 18 service** and points the gateway at it by default (URL
auto-composed from the `POSTGRES_*` vars in `.env`; data persists on the host
under `deploy/pgdata/`). To use a **cloud/managed PostgreSQL** (or sqlite for a
single node) instead, set `GATEWAY_DB_URL` in `.env` — it overrides the bundled URL:

```
GATEWAY_DB_URL=postgresql+psycopg://<user>:<pw>@<host>:5432/<db>?sslmode=require
```

For **local dev** the default is single-file SQLite (`GATEWAY_DB_URL=sqlite:///...`).

The store auto-applies connection-pool tuning (pre-ping + recycle, sized to the
threadpool) for non-sqlite URLs. Schema is managed by **Alembic**, not
`create_all()`: the container entrypoint runs `alembic upgrade head` (idempotent)
before uvicorn, so deploys and restarts self-migrate. Add a schema change with:

```bash
cd services/gateway-server
GATEWAY_DB_URL=sqlite:///$PWD/gw.db uv run alembic revision --autogenerate -m "<change>"
# review the generated migrations/versions/*.py, then it applies on next deploy
```

## Deploy (ECS)
Deploy assets live in `deploy/` (docker-compose, persistent `/data` volume,
`restart: always`, healthcheck). On the ECS host:

```bash
cd services/gateway-server/deploy
cp .env.example .env          # set POSTGRES_PASSWORD + OSS creds, bucket, optional JWT JWKS URL
./deploy.sh --build           # build image from repo root, then `docker compose up -d`
```

`.env` supplies secrets + config: `POSTGRES_PASSWORD` (for the bundled postgres),
`OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET`, `GATEWAY_OSS_BUCKET`,
`GATEWAY_OSS_REGION`, and (for external JWT access) `GATEWAY_AUTH__JWT_JWKS_URL`.
`.env` is gitignored. `docker compose up` starts a **PostgreSQL 18** service, waits
for it to be healthy, then the gateway entrypoint runs `alembic upgrade head`
before uvicorn — so the schema is created/updated automatically. Point
`GATEWAY_DB_URL` at an external DB in `.env` to bypass the bundled postgres (see
[Database & migrations](#database--migrations)).

## Users & API keys
Users and API keys live in the gateway's DB (tables `users` / `api_keys`).
`key_id` **is** the principal jobs are owned by; the secret is stored only as a
sha256 hash. There is no admin UI (MVP) — seed keys with `scripts/seed_key.py`.
The schema is created by `alembic upgrade head` (run automatically by the
container entrypoint on start) before seeding.

**With the bundled PostgreSQL** (docker-compose default) the DB lives in a
container volume, so seed from *inside* the gateway container — it has the DB
URL in `$GATEWAY_DB_URL`, so no `--db`/`--db-url` needed:

```bash
cd services/gateway-server/deploy
docker compose exec gateway python scripts/seed_key.py --principal alice
# inserts user "alice" (if new) + a new API key;
# prints key_id + secret  (store the secret — only its sha256 hash is persisted)

# another key for an existing user (rotation / per-client) — distinct --key-id:
docker compose exec gateway python scripts/seed_key.py --principal alice --key-id gk_alice_ci
```

**With a SQLite DB** (local dev / single node) seed against the DB file directly
(stdlib-only, no container needed):

```bash
python services/gateway-server/scripts/seed_key.py \
    --db services/gateway-server/deploy/data/gateway/gateway.db \
    --principal alice
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
