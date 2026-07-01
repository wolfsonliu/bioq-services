# Build from bioagent project root:
#   docker build --platform linux/amd64 -t promera-server -f services/promera-server/Dockerfile .
#
# Prerequisites:
#   1. ./services/promera-server/scripts/vendor.sh         (tinyprot + promera + LigandMPNN sources)
#   2. ./services/promera-server/scripts/fetch_weights.sh  (Promera + tinyprot + LigandMPNN weights)
#
# tinyprot, promera, and LigandMPNN sources live at
# services/promera-server/upstream/{tinyprot,promera,LigandMPNN}/ (vendored
# at pinned SHAs via scripts/vendor.sh).  Promera: Python 3.12 + torch 2.9 +
# CUDA 12 (cuequivariance-ops-torch-cu12).  All deps available as PyPI
# wheels — uv venv skeleton, no conda needed.

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

# tinyprot — from vendored upstream/tinyprot/ (see scripts/vendor.sh).
COPY services/promera-server/upstream/tinyprot /tmp/tinyprot
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python /tmp/tinyprot/ && \
    rm -rf /tmp/tinyprot

# Promera source — from vendored upstream/promera/ + editable install (so
# `python -m promera` resolves).
COPY services/promera-server/upstream/promera /opt/promera/promera_src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python -e /opt/promera/promera_src/

# All weights (promera + tinyprot LMDBs + LigandMPNN model_params) live on
# NAS at /data/models/promera/{promera,tinyprot,ligandmpnn}/ — NOT baked
# into the image.  FC mounts the NAS; SIF / HPC bind:
#   apptainer run --bind /scratch/models/promera:/data/models/promera \
#       promera-server.sif python -m server cofold ...
# Pre-stage / upload via scripts/fetch_weights.sh (downloads from HuggingFace
# to services/promera-server/weights/, then rsync to NAS).
# See engineering/decisions/2026-06-26-service-weights-externalization.md.

# LigandMPNN source — cherry-pick from vendored upstream/LigandMPNN/.
# Only the 4 scripts and the openfold/ subtree are needed.
COPY services/promera-server/upstream/LigandMPNN/run.py         /opt/promera/LigandMPNN/run.py
COPY services/promera-server/upstream/LigandMPNN/model_utils.py /opt/promera/LigandMPNN/model_utils.py
COPY services/promera-server/upstream/LigandMPNN/data_utils.py  /opt/promera/LigandMPNN/data_utils.py
COPY services/promera-server/upstream/LigandMPNN/score.py       /opt/promera/LigandMPNN/score.py
COPY services/promera-server/upstream/LigandMPNN/openfold       /opt/promera/LigandMPNN/openfold

# LigandMPNN weights — symlink the model_params/ path the upstream code
# expects to the NAS-mounted dir.
RUN ln -s /data/models/promera/ligandmpnn /opt/promera/LigandMPNN/model_params

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
ENV PROMERA_WEIGHTS=/data/models/promera/promera/promera_2606.ckpt
ENV LIGANDMPNN_DIR=/opt/promera/LigandMPNN
ENV PROMERA_LIGANDMPNN_DIR=/opt/promera/LigandMPNN
ENV PROMERA_TEMPLATES_DIR=/opt/promera/promera_src/examples/templates
ENV PROMERA_JOBS_BASE_DIR=/data/promera_jobs
ENV PROMERA_SESSION_HEADER_NAME=bioagent-session-id
# tinyprot/init.py reads TINYPROT_CACHE at import time (falls back to
# ~/.cache/tinyprot which isn't populated on ephemeral FC instances).
ENV TINYPROT_CACHE=/data/models/promera/tinyprot
RUN mkdir -p /data/promera_jobs

ENV PORT=9000
EXPOSE 9000
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "9000", "--timeout-keep-alive", "900"]
