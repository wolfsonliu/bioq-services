# Gateway (Control Plane)

English | [中文](gateway.zh.md)

> **Read when**: you change anything under `gateway/` (auth, dispatch, storage, schema, config).
> **Source**: `gateway/app.py`, `gateway/settings.py`, `gateway/dispatchers/`, `gateway/README.md`.
> **Refresh/remove when**: a gateway endpoint, backend, or env var changes.

`gateway/` is the always-on API gateway (ECS / compose / kind) in front of the downstream
FC/HTTP/OpenFaaS services. Entry points: `python -m server` (in-container) / `python -m gateway`
(in-repo). The FastAPI app is `gateway/app.py`; the single config schema source is
`gateway/settings.py` (`GatewaySettings`, `env_prefix="GATEWAY_"`, nested keys use `GATEWAY_AUTH__...`).

## Endpoints (`/v1/*`)

`GET /v1/services`, `GET /v1/services/{svc}`, `POST /v1/run/{svc}/{endpoint}`,
`GET /v1/jobs/{job_id}`, `GET /v1/jobs/{job_id}/download`, `POST /v1/jobs/{job_id}/cancel`,
`POST /v1/uploads/prepare`, `GET /healthz`.

## Auth

Two layers: VPC bypass (localhost / in-network break-glass) → OIDC/JWT (`Authorization: Bearer`).
The api key is retired. Humans use OIDC device flow / SSO; machines use OIDC client-credentials.
Users are JIT-provisioned; role derives from the token's groups claim
(`GATEWAY_AUTH__JWT_ADMIN_GROUP`, default `bioq-admins` → admin). Production must set
`GATEWAY_AUTH__JWT_ISSUER` and keep `bypass_vpc=false`. Admin console at `/admin` (SSO, CSRF).

## Dispatch backends

`GATEWAY_DISPATCH_BACKEND` = `fc` (Aliyun FC async-task mode, polled via FC OpenAPI `GetAsyncTask`;
AK/SK from `ALI_AK`/`ALI_SK`) / `http` (compose, direct submit/poll against each service's in-process
runner) / `openfaas` (kind + OpenFaaS). Dispatcher protocol lives in `gateway/dispatchers/`
(`base.py` + `fc.py` / `local.py` / `openfaas.py`).

## Storage

`GATEWAY_STORAGE_BACKEND` = `oss` (presigned direct upload, default bucket `bioagent-inputs`) or
`file` (`/v1/files` over a shared volume `GATEWAY_FILE_BASE_DIR`).

## Database / migrations

Users + jobs use SQLAlchemy; the schema is managed by **Alembic** (not `create_all()`). The container
entrypoint runs `alembic upgrade head` before uvicorn. Local sqlite
(`GATEWAY_DB_URL=sqlite:///<path>`); production Postgres (`postgresql+psycopg://...`). To add a schema
change:

```bash
cd gateway && GATEWAY_DB_URL=sqlite:///$PWD/gw.db uv run alembic revision --autogenerate -m "<change>"
```

## Deploy config

Non-secret topology lives in checked-in `deploy/config/gateway.<target>.env` (`make gen-config`
regenerates from the schema; `make check-config` is a CI drift gate). Secrets live only in each
target's git-ignored `.env`. The gateway validates config on boot and refuses to start on fatal
misconfiguration.

## Related

- Local kind deployment and Keycloak: [local-dev.md](./local-dev.md)
- Building/tagging the gateway image: [build-deploy.md](./build-deploy.md)
