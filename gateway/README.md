# gateway-server

Persistent API gateway (ECS) fronting the FC GPU services. Single public
entry: auth + user/key management (SQLite) + storage-backed upload (OSS presign
or gateway-proxied file) + async
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
| POST | /v1/uploads/prepare | mint an upload target (OSS presigned PUT or gateway /v1/files) for an input |
| GET | /healthz | health |

## Auth
Two layers: **VPC bypass** (internal/localhost, break-glass) → **JWT/OIDC**
(`Authorization: Bearer <token>`). API keys were retired — humans use OIDC device
flow / SSO, machines use OIDC client-credentials.

**OIDC / JWT**: point `GATEWAY_AUTH__JWT_JWKS_URL` at an IdP's JWKS (Keycloak/Dex/
corp SSO). Verified tokens authenticate as `account_id = sub`; the user is
provisioned just-in-time and its `role` is derived from a groups claim
(`GATEWAY_AUTH__JWT_ADMIN_GROUP`, default `bioq-admins` → admin). Production MUST
set `GATEWAY_AUTH__JWT_ISSUER` so tokens from other realms are rejected, and keep
`GATEWAY_AUTH__BYPASS_VPC=false` unless the VPC host is genuinely trusted. The
admin console (`/admin`) logs in via SSO (or VPC bypass internally). See the local
IdP spike in `deploy/keycloak/` and the in-cluster Keycloak in `deploy/openfaas/`.

## Local dev
```bash
# create/upgrade the schema first (alembic; app startup no longer create_all()s)
cd gateway
GATEWAY_DB_URL=sqlite:///$PWD/gw.db uv run alembic upgrade head
cd -

GATEWAY_DB_URL=sqlite:///$PWD/gw.db \
GATEWAY_REGISTRY_PATH=$PWD/services.yaml \
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
cd gateway
GATEWAY_DB_URL=sqlite:///$PWD/gw.db uv run alembic revision --autogenerate -m "<change>"
# review the generated migrations/versions/*.py, then it applies on next deploy
```

## Deploy (ECS)
Deploy assets live in `deploy/` (docker-compose, persistent `/data` volume,
`restart: always`, healthcheck). On the ECS host:

```bash
cd gateway/deploy
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

## Users & roles
Users live in the gateway's DB (table `users`): an account (`account_id` = the
JWT `sub`) with a `role` (`user` | `admin`). **There is no user/key CRUD in the
gateway** — identity is owned by the IdP. Users are **provisioned just-in-time**
on first authenticated request, and their `role` is derived from the token's
groups claim (`GATEWAY_AUTH__JWT_ADMIN_GROUP`, default `bioq-admins` → admin).

Create/manage users in the IdP:
- **Local (in-cluster Keycloak)**: `make local-user ACCOUNT=alice PASSWORD=pw [ADMIN=1]`
  (see the repo README's local-dev section).
- **Production**: manage users/groups in the managed IdP (which may front LDAP/AD).

Authenticate with a Bearer token (see [Auth](#auth)):
```bash
curl -H "Authorization: Bearer <OIDC token>" https://<gateway-host>/v1/services
```
Internal (VPC) callers hitting `*-vpc.fcapp.run` / localhost are auto-bypassed.

## Admin console
A server-side-rendered, terminal-styled management UI at `/admin` (read-only
dashboard, accounts, jobs, services + write ops: create account / cancel job /
reload `services.yaml`). Browser auth is a cookie session established via **OIDC
SSO**: navigate to `/admin/login` → "Sign in with SSO" (requires `oidc_issuer` /
`oidc_client_id` / `oidc_client_secret` configured); only `bioq-admins` users get
in. Internal VPC hosts are bypassed and land on `/admin` directly. Write forms are
CSRF-protected; the session cookie is `SameSite=lax` and signed with
`GATEWAY_SESSION_SECRET` (set it explicitly for multi-instance).

## Tests
```bash
uv run python -m pytest gateway/tests/ -v
```
