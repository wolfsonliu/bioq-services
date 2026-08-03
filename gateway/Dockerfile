# Build context = repo root:  docker build -f gateway/Dockerfile .
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
WORKDIR /opt/gateway
RUN uv venv .venv --python python3.11
ENV PATH="/opt/gateway/.venv/bin:${PATH}"

# framework + runtime deps (framework installed from source via bind mount)
RUN --mount=type=bind,source=framework,target=/tmp/service-framework \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" \
        "httpx>=0.28" "pyjwt[crypto]>=2.8" "sqlalchemy>=2.0" \
        "jinja2>=3.1" "itsdangerous>=2.1" "python-multipart>=0.0.9" \
        "alembic>=1.13" "psycopg[binary]>=3.1" alibabacloud-oss-v2 \
        "alibabacloud_fc20230330>=4.0" "alibabacloud_tea_openapi>=0.3.0" alibabacloud_tea_util

# gateway code into /opt/gateway/server/  (imported as `server.app`).
# Glob all top-level modules (+ VERSION) so a new module can't be forgotten;
# `*.py` is non-recursive, so tests/ / scripts/ are excluded. auth/ + db/ +
# dispatchers/ are subpackages, copied explicitly.
COPY gateway/*.py gateway/VERSION /opt/gateway/server/
COPY gateway/auth/ /opt/gateway/server/auth/
COPY gateway/db/ /opt/gateway/server/db/
COPY gateway/dispatchers/ /opt/gateway/server/dispatchers/
# admin console (server-side rendered): code + Jinja templates + static CSS.
COPY gateway/admin/ /opt/gateway/server/admin/

# schema migrations (alembic upgrade head runs in entrypoint before uvicorn).
# alembic.ini + migrations/ live at /opt/gateway (the WORKDIR alembic runs from).
COPY gateway/alembic.ini /opt/gateway/alembic.ini
COPY gateway/migrations/ /opt/gateway/migrations/
COPY gateway/entrypoint.sh /opt/gateway/entrypoint.sh
RUN chmod +x /opt/gateway/entrypoint.sh

# admin helper: seed users/API keys (docker compose exec gateway python scripts/seed_key.py)
COPY gateway/scripts/ /opt/gateway/scripts/

# downstream registry (svc -> ServiceRecord)
COPY services.yaml /opt/gateway/services.yaml

ENV GATEWAY_SESSION_HEADER_NAME=bioagent-session-id
ENV GATEWAY_REGISTRY_PATH=/opt/gateway/services.yaml
ENV GATEWAY_DB_URL=sqlite:////data/gateway/gateway.db
ENV GATEWAY_JOBS_BASE_DIR=/data/gateway_jobs
ENV PORT=9000
EXPOSE 9000
# Persistent (ECS): mount a volume at /data for the SQLite DB + job scratch.
# entrypoint runs `alembic upgrade head` (idempotent) then execs uvicorn.
CMD ["/opt/gateway/entrypoint.sh"]
