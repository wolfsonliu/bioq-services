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
Run the image persistently with a volume at `/data` (SQLite + scratch). Set
`OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET`, `GATEWAY_OSS_BUCKET`, and (for
external access) `GATEWAY_AUTH__JWT_JWKS_URL`. Seed users/keys via the DB.

## Tests
```bash
uv run python -m pytest services/gateway-server/tests/ -v
```
