# Build from bioagent project root:
#   docker build --platform linux/amd64 -t rfantibody-server -f services/rfantibody-server/Dockerfile .
#
# RFantibody source must be at opensource/RFantibody/ (weights baked into image).
# Base: cudnn-runtime (not -devel). RFantibody installs entirely from prebuilt
# wheels via `uv sync`, no nvcc / CUDA headers needed at build or runtime.
# Save ~5 GB vs the -devel variant on this CUDA 11.8 line.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# --- Install RFantibody ---
WORKDIR /opt/rfantibody

# Dependencies first (cache-friendly layer order)
COPY opensource/RFantibody/pyproject.toml opensource/RFantibody/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Full RFantibody source + weights
COPY opensource/RFantibody/ .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Shared bioagent service framework (HTTP/job/error layer, pydantic models, NAS-aware
# read-through cache). Installing it as a separate layer keeps cache hit rates high
# when only the framework's source changes.
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install /tmp/service-framework \
    && uv pip install httpx alibabacloud-oss-v2

# --- Copy server code as "server" package ---
COPY services/rfantibody-server/ /opt/rfantibody/server/

# Env-driven settings — see services/rfantibody-server/settings.py (RFantibodySettings).
# The RFANTIBODY_ prefix is read by pydantic-settings; defaults match the image layout.
ENV RFANTIBODY_ROOT=/opt/rfantibody
ENV RFANTIBODY_WEIGHTS_DIR=/opt/rfantibody/weights
ENV RFANTIBODY_SCRIPTS_DIR=/opt/rfantibody/scripts
ENV RFANTIBODY_JOBS_BASE_DIR=/data/rfantibody_jobs

# --- Alibaba Cloud FC configuration ---
# CAPort: FC default listening port is 9000
# Keep-alive: FC requires >= 15 minutes (900s)
# Startup: HTTP server must be ready within 120s
ENV PORT=9000
EXPOSE 9000

CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
