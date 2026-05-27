# Build from bioagent project root:
#   docker build --platform linux/amd64 -t rfantibody-server -f services/rfantibody-server/Dockerfile .
#
# RFantibody source must be at opensource/RFantibody/ (weights baked into image).
# Base: cudnn-runtime (not -devel). RFantibody installs entirely from prebuilt
# wheels via `uv pip install`, no nvcc / CUDA headers needed at build or runtime.
# Save ~5 GB vs the -devel variant on this CUDA 11.8 line.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
# CN build hosts frequently time out on multi-MB wheels from files.pythonhosted.org
# (biotraj/biotite/torch). Default uv timeout is 30s — bump it and trim parallel
# downloads so each wheel gets more bandwidth.
ENV UV_HTTP_TIMEOUT=600
ENV UV_CONCURRENT_DOWNLOADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# --- Install RFantibody ---
WORKDIR /opt/rfantibody

# Create the venv up-front so subsequent `uv pip install` invocations target it
# consistently (uv pip auto-detects .venv in CWD, but being explicit is safer).
RUN uv venv .venv --python python3.10

# Switched from `uv sync --frozen` to `uv pip install -e .` because uv.lock
# pins each wheel to a hardcoded `files.pythonhosted.org` URL, which bypasses
# the Huawei mirror set via UV_INDEX_URL above. From CN build hosts those URLs
# time out on multi-MB wheels (biotraj/biotite/torch). `uv pip install` honors
# UV_INDEX_URL for the default index and reads [tool.uv.sources] +
# [[tool.uv.index]] from pyproject.toml for torch (pytorch index) / dgl
# (direct wheel URL). Trade-off: lockfile reproducibility lost; acceptable
# because the alternative (build failures behind a flaky CN→pypi link) is
# strictly worse.
COPY opensource/RFantibody/ .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python -e .

# Shared bioagent service framework (HTTP/job/error layer, pydantic models, NAS-aware
# read-through cache). Installing it as a separate layer keeps cache hit rates high
# when only the framework's source changes.
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# Server code last — explicit file list so test/doc changes don't bust cache.
COPY services/rfantibody-server/__init__.py \
     services/rfantibody-server/VERSION \
     services/rfantibody-server/app.py \
     services/rfantibody-server/adapter.py \
     services/rfantibody-server/models.py \
     services/rfantibody-server/settings.py \
     services/rfantibody-server/tools.py \
     services/rfantibody-server/uris.py \
     /opt/rfantibody/server/

# Env-driven settings — see services/rfantibody-server/settings.py (RFantibodySettings).
# The RFANTIBODY_ prefix is read by pydantic-settings; defaults match the image layout.
ENV RFANTIBODY_ROOT=/opt/rfantibody
ENV RFANTIBODY_WEIGHTS_DIR=/opt/rfantibody/weights
ENV RFANTIBODY_SCRIPTS_DIR=/opt/rfantibody/scripts
ENV RFANTIBODY_JOBS_BASE_DIR=/data/rfantibody_jobs
RUN mkdir -p /data/rfantibody_jobs

# --- Alibaba Cloud FC configuration ---
# CAPort: FC default listening port is 9000
# Keep-alive: FC requires >= 15 minutes (900s)
# Startup: HTTP server must be ready within 120s
ENV RFANTIBODY_MAX_CONCURRENT_JOBS=1
ENV PORT=9000
EXPOSE 9000

CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
