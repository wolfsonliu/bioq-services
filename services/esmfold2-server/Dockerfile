# Build from bioagent project root:
#   docker build --platform linux/amd64 -t esmfold2-server -f services/esmfold2-server/Dockerfile .
#
# ESM source must be at opensource/esm/ (GPU-required, but built without nvcc).
# Pre-download weights before building:
#   ./services/esmfold2-server/scripts/fetch_weights.sh
#
# Base: uv's pre-built python3.12 slim image (Debian bookworm + uv pre-installed).
# PyTorch wheel bundles CUDA runtime libs, so nvidia base is not needed.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/esmfold2
RUN uv venv .venv --python python3.12

# Heavy deps first (torch). Separate layer for cache efficiency.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "torch>=2.2"

# ESM package source + editable install.
# This pulls in the custom Transformers fork, einops, biotite, rdkit, etc.
COPY opensource/esm/pyproject.toml opensource/esm/README.md /opt/esmfold2/esm/
COPY opensource/esm/esm /opt/esmfold2/esm/esm
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python -e ./esm

# Pre-staged ESMFold2 weights from HuggingFace (pre-download before build).
# See scripts/fetch_weights.sh.
COPY weights/esmfold2/ /opt/esmfold2/weights/

# Build-time import smoke test.
RUN .venv/bin/python -c "\
import torch; \
print('torch', torch.__version__, 'compiled cuda', torch.version.cuda); \
from esm.models.esmfold2 import ESMFold2InputBuilder; \
print('esm.models.esmfold2 OK')"

# Shared bioagent service framework + remote-fetch deps.
RUN --mount=type=bind,source=services/_framework,target=/tmp/service-framework \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# Inference script — standalone, lives alongside server code.
COPY services/esmfold2-server/inference.py /opt/esmfold2/inference.py

# Server code last — explicit file list so test/doc changes don't bust cache.
COPY services/esmfold2-server/__init__.py \
     services/esmfold2-server/__main__.py \
     services/esmfold2-server/VERSION \
     services/esmfold2-server/app.py \
     services/esmfold2-server/adapter.py \
     services/esmfold2-server/models.py \
     services/esmfold2-server/settings.py \
     services/esmfold2-server/tools.py \
     services/esmfold2-server/uris.py \
     /opt/esmfold2/server/

ENV PYTHONPATH=/opt/esmfold2
ENV ESMFOLD2_ROOT=/opt/esmfold2
ENV ESMFOLD2_PYTHON=/opt/esmfold2/.venv/bin/python
ENV ESMFOLD2_INFERENCE_SCRIPT=/opt/esmfold2/inference.py
ENV ESMFOLD2_MODEL_DIR=/opt/esmfold2/weights
ENV ESMFOLD2_CCD_PATH=/opt/esmfold2/weights/ccd.pkl
ENV ESMCFOLD_CCD_PATH=/opt/esmfold2/weights/ccd.pkl
ENV ESMFOLD2_JOBS_BASE_DIR=/data/esmfold2_jobs
RUN mkdir -p /data/esmfold2_jobs

ENV PORT=9000
EXPOSE 9000
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
