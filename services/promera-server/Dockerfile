# Build from bioagent project root:
#   docker build --platform linux/amd64 -t promera-server -f services/promera-server/Dockerfile .
#
# Prerequisites:
#   1. ./services/promera-server/scripts/fetch_weights.sh   (Promera + tinyprot + LigandMPNN weights)
#
# Promera, tinyprot, and LigandMPNN sources are fetched from GitHub at pinned
# SHAs (see *_SHA ARGs below).  Promera: Python 3.12 + torch 2.9 + CUDA 12
# (cuequivariance-ops-torch-cu12).  All deps available as PyPI wheels — uv
# venv skeleton, no conda needed.

# Pinned upstream commits — bump SHAs to upgrade individual components.
ARG TINYPROT_REPO=https://github.com/bjing2016/tinyprot.git
ARG TINYPROT_SHA=e33866bc474c64a9cfc324e0cbb7f9f82ad1e855
ARG PROMERA_REPO=https://github.com/bjing2016/promera.git
ARG PROMERA_SHA=5c7d6a88bbf7f13906274fdc7c7f84e7a666017d
ARG LIGANDMPNN_REPO=https://github.com/dauparas/LigandMPNN
ARG LIGANDMPNN_SHA=26ec57ac976ade5379920dbd43c7f97a91cf82de

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

# Python 3.12 from deadsnakes PPA (Ubuntu 22.04 ships 3.10).
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    ca-certificates curl git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# build-essential is required at runtime by triton: the JIT path
# (cuequivariance_ops_torch → triton.runtime.build._build) shells out to
# the host C compiler to build the NVIDIA driver launcher .so the first
# time a kernel runs.  cuda:*-runtime base images don't include gcc, so
# without this the design endpoint crashes mid-inference with
# "Failed to find C compiler".  See engineering/guides/promera-tinyprot-cache.md.

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

WORKDIR /opt/promera
RUN uv venv .venv --python python3.12

# Heavy deps first — torch 2.9 (~2.5 GB, includes CUDA 12 runtime).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
    torch==2.9.0 triton==3.5.0 numpy

# tinyprot — clone at pinned SHA + pip install.
ARG TINYPROT_REPO
ARG TINYPROT_SHA
RUN set -eux; \
    git clone --filter=blob:none --no-checkout "$TINYPROT_REPO" /tmp/tinyprot; \
    cd /tmp/tinyprot; \
    git checkout "$TINYPROT_SHA"; \
    test "$(git rev-parse HEAD)" = "$TINYPROT_SHA"; \
    rm -rf .git
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python /tmp/tinyprot/ && \
    rm -rf /tmp/tinyprot

# Promera source — clone at pinned SHA + editable install (so `python -m
# promera` resolves).
ARG PROMERA_REPO
ARG PROMERA_SHA
RUN set -eux; \
    git clone --filter=blob:none --no-checkout "$PROMERA_REPO" /opt/promera/promera_src; \
    cd /opt/promera/promera_src; \
    git checkout "$PROMERA_SHA"; \
    test "$(git rev-parse HEAD)" = "$PROMERA_SHA"; \
    rm -rf .git
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python -e /opt/promera/promera_src/

# Promera weights (pre-downloaded via scripts/fetch_weights.sh).
COPY services/promera-server/weights/promera/ /opt/promera/weights/

# tinyprot LMDB caches — must be present at import time of tinyprot.msa,
# otherwise the cofold/design subprocess crashes before it can run.
# scripts/fetch_weights.sh pre-downloads these from HuggingFace.
COPY services/promera-server/weights/tinyprot/ccd.lmdb/      /opt/promera/tinyprot_cache/ccd.lmdb/
COPY services/promera-server/weights/tinyprot/taxonomy.lmdb/ /opt/promera/tinyprot_cache/taxonomy.lmdb/

# LigandMPNN source — clone at pinned SHA, cherry-pick the 4 scripts and the
# openfold/ subtree, drop the rest.  Weights are staged locally.
ARG LIGANDMPNN_REPO
ARG LIGANDMPNN_SHA
RUN set -eux; \
    git clone --filter=blob:none --no-checkout "$LIGANDMPNN_REPO" /tmp/ligandmpnn-src; \
    cd /tmp/ligandmpnn-src; \
    git checkout "$LIGANDMPNN_SHA"; \
    test "$(git rev-parse HEAD)" = "$LIGANDMPNN_SHA"; \
    rm -rf .git; \
    mkdir -p /opt/promera/LigandMPNN; \
    cp run.py model_utils.py data_utils.py score.py /opt/promera/LigandMPNN/; \
    cp -r openfold /opt/promera/LigandMPNN/openfold; \
    rm -rf /tmp/ligandmpnn-src
COPY services/promera-server/weights/ligandmpnn/ /opt/promera/LigandMPNN/model_params/

# Shared bioagent service framework + remote-fetch deps.
RUN --mount=type=bind,source=services/_framework,target=/tmp/service-framework \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
    "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2 pyyaml

# Server code — explicit file list for cache isolation.
COPY services/promera-server/__init__.py \
     services/promera-server/__main__.py \
     services/promera-server/VERSION \
     services/promera-server/app.py \
     services/promera-server/adapter.py \
     services/promera-server/models.py \
     services/promera-server/settings.py \
     services/promera-server/tools.py \
     services/promera-server/uris.py \
     /opt/promera/server/

ENV PYTHONPATH=/opt/promera
ENV PROMERA_ROOT=/opt/promera
ENV PROMERA_PYTHON=/opt/promera/.venv/bin/python
ENV PROMERA_WEIGHTS=/opt/promera/weights/promera_2606.ckpt
ENV LIGANDMPNN_DIR=/opt/promera/LigandMPNN
ENV PROMERA_LIGANDMPNN_DIR=/opt/promera/LigandMPNN
ENV PROMERA_TEMPLATES_DIR=/opt/promera/promera_src/examples/templates
ENV PROMERA_JOBS_BASE_DIR=/data/promera_jobs
ENV PROMERA_SESSION_HEADER_NAME=bioagent-session-id
# tinyprot/init.py reads TINYPROT_CACHE at import time (falls back to
# ~/.cache/tinyprot which isn't populated on ephemeral FC instances).
ENV TINYPROT_CACHE=/opt/promera/tinyprot_cache
RUN mkdir -p /data/promera_jobs

ENV PORT=9000
EXPOSE 9000
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "9000", "--timeout-keep-alive", "900"]
