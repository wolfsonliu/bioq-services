# Build from bioagent project root:
#   docker build --platform linux/amd64 -t ensemble-server -f services/ensemble-server/Dockerfile .
#
# ensemble-server — multi-method biomolecular task aggregator (CPU only).
# Unlike most bioagent services, this image does NOT bundle a GPU model;
# it orchestrates remote FC calls via pipelines.framework.FCDispatcher.
#
# Base: uv's pre-built python3.12 bookworm slim image (project convention,
# saves a `COPY --from=ghcr.io/astral-sh/uv` step vs `python:3.12-slim`).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Huawei Cloud pip mirror (~10× faster from China than pypi.org).
    UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/ \
    UV_HTTP_TIMEOUT=600 \
    # PYTHONPATH lets `import server.app` (via /opt/ensemble) and
    # `import pipelines.framework.fc_dispatcher` (via /opt) work without
    # an editable install of the project root.
    PYTHONPATH=/opt:/opt/ensemble

# curl is needed for FC healthz probes; ca-certificates for HTTPS calls
# the FCDispatcher makes to underlying GPU services.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ensemble
RUN uv venv /opt/.venv --python python3.12
ENV PATH=/opt/.venv/bin:$PATH

# Shared bioagent service framework (provides ServiceSettings, read_version_file).
# Copied early so subsequent server-code changes don't bust the framework layer.
COPY services/_framework /opt/_framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/.venv/bin/python /opt/_framework

# Runtime deps for the aggregator: FastAPI stack + Alibaba Cloud FC OpenAPI
# (used by FCDispatcher to invoke underlying GPU services in async-task mode).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/.venv/bin/python \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        "pydantic>=2.7" \
        "pydantic-settings>=2.4" \
        "httpx>=0.28" \
        "alibabacloud_fc20230330>=4.0" \
        "alibabacloud_tea_openapi>=0.3" \
        "alibabacloud_tea_util>=0.3"

# pipelines.framework.fc_dispatcher — imported by server.app.  Pure-Python,
# no install needed (sits on PYTHONPATH).  Only the framework subdir is needed;
# everything else under pipelines/ is excluded via .dockerignore.
COPY pipelines/__init__.py /opt/pipelines/__init__.py
COPY pipelines/framework /opt/pipelines/framework

# Server source — copied last so iterating on routes/adapters doesn't bust
# the heavier framework / deps layers above.
COPY services/ensemble-server /opt/ensemble/server

ENV PORT=9000
EXPOSE 9000

# WORKDIR is /opt/ensemble so `python -m server` resolves /opt/ensemble/server
# as the `server` package (CPython prepends CWD to sys.path).  __main__.py
# starts uvicorn on 0.0.0.0:9000.
CMD ["python", "-m", "server"]
