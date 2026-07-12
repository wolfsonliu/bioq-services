# Build context = repo root:  docker build -f services/gateway-server/Dockerfile .
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
WORKDIR /opt/gateway
RUN uv venv .venv --python python3.11
ENV PATH="/opt/gateway/.venv/bin:${PATH}"

# framework + runtime deps (framework installed from source via bind mount)
RUN --mount=type=bind,source=services/_framework,target=/tmp/service-framework \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" \
        "httpx>=0.28" "pyjwt[crypto]>=2.8" "sqlalchemy>=2.0" alibabacloud-oss-v2

# gateway code into /opt/gateway/server/  (imported as `server.app`)
COPY services/gateway-server/__init__.py services/gateway-server/settings.py \
     services/gateway-server/adapter.py services/gateway-server/app.py \
     services/gateway-server/models.py services/gateway-server/registry.py \
     services/gateway-server/discover.py services/gateway-server/dispatch.py \
     services/gateway-server/presign.py services/gateway-server/VERSION \
     /opt/gateway/server/
COPY services/gateway-server/auth/ /opt/gateway/server/auth/
COPY services/gateway-server/db/ /opt/gateway/server/db/

# downstream registry (svc -> ServiceRecord)
COPY services/services.yaml /opt/gateway/services.yaml

ENV GATEWAY_SESSION_HEADER_NAME=bioagent-session-id
ENV GATEWAY_REGISTRY_PATH=/opt/gateway/services.yaml
ENV GATEWAY_DB_URL=sqlite:////data/gateway/gateway.db
ENV GATEWAY_JOBS_BASE_DIR=/data/gateway_jobs
ENV PORT=9000
EXPOSE 9000
# Persistent (ECS): mount a volume at /data for the SQLite DB + job scratch.
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "9000", "--timeout-keep-alive", "900"]
