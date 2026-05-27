# Build from bioagent project root:
#   docker build --platform linux/amd64 -t dockq-server -f services/dockq-server/Dockerfile .
#
# DockQ source must be at opensource/DockQ/ (CPU-only Python + Cython extension).
# Final image ~350 MB — DockQ + bioagent-service-framework + numpy/biopython wheels.
#
# Base: uv's pre-built python3.11 slim image (Debian bookworm + uv pre-installed).
# Chosen over python:3.11-slim to skip the `COPY --from=ghcr.io/astral-sh/uv` step,
# and over Alpine because DockQ requires numpy<2.0 + Cython compile — Alpine's musl
# libc breaks every manylinux wheel and would balloon build time to 8+ min.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror (project-wide convention, ~10× faster from China than
# pypi.org).
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/

# build-essential is needed only for DockQ's Cython compile during install.
# We keep it for now (~150 MB); a future optimization could do a multistage
# build and drop it from the runtime stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create the venv up front so subsequent COPYs do not overlay it.
WORKDIR /opt/dockq
RUN uv venv .venv --python python3.11

# Heavy deps first (numpy + biopython + Cython) — large layer, rarely changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "numpy<2.0" biopython networkx parallelbar cython

# DockQ source + editable install. COPY overlays files but preserves .venv/.
COPY opensource/DockQ /opt/dockq
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python -e .

# Shared bioagent service framework + remote-fetch deps (late layer so the
# framework changing doesn't invalidate the DockQ install layer above).
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# Server code last — explicit file list so test/doc changes don't bust cache.
COPY services/dockq-server/__init__.py \
     services/dockq-server/VERSION \
     services/dockq-server/app.py \
     services/dockq-server/adapter.py \
     services/dockq-server/batch_score.py \
     services/dockq-server/models.py \
     services/dockq-server/settings.py \
     services/dockq-server/tools.py \
     services/dockq-server/uris.py \
     /opt/dockq/server/

# DockQ CLI installed under .venv/bin/DockQ — make it the resolved binary.
ENV DOCKQ_BINARY=/opt/dockq/.venv/bin/DockQ
ENV DOCKQ_ROOT=/opt/dockq
ENV DOCKQ_JOBS_BASE_DIR=/data/dockq_jobs
RUN mkdir -p /data/dockq_jobs

ENV PORT=9000
EXPOSE 9000
# WORKDIR is /opt/dockq (set above) — CPython prepends CWD to sys.path,
# making the `server/` subdir importable as `server.app`. Without this
# uvicorn fails with `ModuleNotFoundError: No module named 'server'`.
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
