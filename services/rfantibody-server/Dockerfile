# Build from bioagent project root:
#   docker build --platform linux/amd64 -t rfantibody-server -f services/rfantibody-server/Dockerfile .
#
# Pre-build (once; re-run to bump pinned upstream commit):
#   ./services/rfantibody-server/scripts/vendor.sh
#
# RFantibody source lives at services/rfantibody-server/upstream/ (vendored
# at pinned SHA via scripts/vendor.sh).  Weights/ is NOT in the upstream
# repo — must be pre-staged at services/rfantibody-server/weights/ before
# building.
# Base: cudnn-runtime (not -devel). RFantibody installs entirely from prebuilt
# wheels via `uv pip install`, no nvcc / CUDA headers needed at build or runtime.
# Save ~5 GB vs the -devel variant.
#
# CUDA 12.1 (was 11.8): upstream RFantibody pyproject.toml pins torch/dgl to
# cu118 wheels, whose cubins don't cover sm_89 (RTX 4090). We override with
# services/rfantibody-server/rfantibody-pyproject.toml (cu121 wheels). 5090
# (sm_120 / Blackwell) still unsupported — needs CUDA ≥ 12.8 + torch ≥ 2.6.
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

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

# ---------------------------------------------------------------------------
# RFantibody source — selective COPY for cache isolation.
#
# weights/ (1.6 GB, very stable) is a separate layer so code edits don't bust
# the weight cache. test/ (29 MB), CLAUDE.md, uv.lock are skipped.
#
# Switched from `uv sync --frozen` to `uv pip install -e .` because uv.lock
# pins each wheel to a hardcoded `files.pythonhosted.org` URL, which bypasses
# the Huawei mirror set via UV_INDEX_URL above. Trade-off: lockfile
# reproducibility lost; acceptable because the alternative (build failures
# behind a flaky CN→pypi link) is strictly worse.
# ---------------------------------------------------------------------------
# Weights live on NAS at /data/models/rfantibody/weights/ — NOT baked into
# the image (~1.6 GB savings).  FC mounts the NAS; SIF / HPC bind via:
#   apptainer run --bind /scratch/models/rfantibody:/data/models/rfantibody \
#       rfantibody-server.sif python -m server design ...
# Pre-stage to local: ./services/rfantibody-server/scripts/fetch_weights.sh
# (or set WEIGHTS_DST to download directly to the NAS / HPC target).
# See engineering/decisions/2026-06-26-service-weights-externalization.md.

# RFantibody source — from vendored upstream/ (see scripts/vendor.sh).
# pyproject.toml is the cu121-patched override from services/rfantibody-server/
# (see rfantibody-pyproject.toml header for rationale) and is COPYed *after*
# the upstream so it wins.
COPY services/rfantibody-server/upstream/README.md ./README.md
COPY services/rfantibody-server/upstream/src       ./src
COPY services/rfantibody-server/upstream/scripts   ./scripts
COPY services/rfantibody-server/upstream/include   ./include
COPY services/rfantibody-server/rfantibody-pyproject.toml ./pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python -e .

# Shared bioagent service framework (HTTP/job/error layer, pydantic models, NAS-aware
# read-through cache). Bind mount avoids a COPY layer — framework dir is available
# during RUN but never stored in the image.
RUN --mount=type=bind,source=services/_framework,target=/tmp/service-framework \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# Server code last — explicit file list so test/doc changes don't bust cache.
COPY services/rfantibody-server/__init__.py \
     services/rfantibody-server/__main__.py \
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
ENV PYTHONPATH=/opt/rfantibody
ENV RFANTIBODY_ROOT=/opt/rfantibody
ENV RFANTIBODY_WEIGHTS_DIR=/data/models/rfantibody/weights
ENV RFANTIBODY_SCRIPTS_DIR=/opt/rfantibody/scripts
ENV RFANTIBODY_JOBS_BASE_DIR=/data/rfantibody_jobs
RUN mkdir -p /data/rfantibody_jobs

# --- Alibaba Cloud FC configuration ---
# CAPort: FC default listening port is 9000
# Keep-alive: FC requires >= 15 minutes (900s)
# Startup: HTTP server must be ready within 120s
ENV RFANTIBODY_MAX_CONCURRENT_JOBS=1
# Keepalive pings this URL through FC's gateway so the instance stays alive
# during long GPU jobs. Override at deploy time if the URL changes.
# ENV RFANTIBODY_KEEPALIVE_URL=https://fc-rfantibody-guekbpucdo.cn-hangzhou-vpc.fcapp.run
ENV RFANTIBODY_SESSION_HEADER_NAME=bioagent-session-id
ENV PORT=9000
EXPOSE 9000

CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
