# Build from bioagent project root:
#   docker build --platform linux/amd64 -t esmfold2-server -f services/esmfold2-server/Dockerfile .
#
# Pre-build (once; re-run to bump pinned upstream commit):
#   ./services/esmfold2-server/scripts/vendor.sh           # vendor upstream/
#   ./services/esmfold2-server/scripts/fetch_weights.sh    # weights (NAS)
#
# ESM (Biohub fork) lives at services/esmfold2-server/upstream/ (vendored at
# pinned SHA via scripts/vendor.sh).  GPU-required, but built without nvcc.
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

# ESM package source — from vendored upstream/ (see scripts/vendor.sh) +
# editable install.  Pulls in the custom Transformers fork, einops, biotite,
# rdkit, etc.
COPY services/esmfold2-server/upstream/pyproject.toml /opt/esmfold2/esm/pyproject.toml
COPY services/esmfold2-server/upstream/README.md      /opt/esmfold2/esm/README.md
COPY services/esmfold2-server/upstream/esm            /opt/esmfold2/esm/esm
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python -e ./esm

# Weights are NOT bundled in the image — they live on a NAS volume
# mounted at /data/models/ by FC configuration.
# Expected layout:
#   /data/models/esmfold2/   — ESMFold2 checkpoint + ccd.pkl
#   /data/models/esmc/6b/    — ESMC-6B checkpoint

# Build-time import smoke test.
RUN .venv/bin/python -c "\
import torch; \
print('torch', torch.__version__, 'compiled cuda', torch.version.cuda); \
from esm.models.esmfold2 import ESMFold2InputBuilder; \
print('esm.models.esmfold2 OK')"

# Shared bioagent service framework + remote-fetch deps.
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
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
# Prevent any HuggingFace network access at runtime (FC VPC has no internet).
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV ESMFOLD2_ROOT=/opt/esmfold2
ENV ESMFOLD2_PYTHON=/opt/esmfold2/.venv/bin/python
ENV ESMFOLD2_INFERENCE_SCRIPT=/opt/esmfold2/inference.py
ENV ESMFOLD2_MODEL_DIR=/data/models/esmfold2
ENV ESMFOLD2_ESMC_DIR=/data/models/esmc/6b
ENV ESMFOLD2_CCD_PATH=/data/models/esmfold2/ccd.pkl
ENV ESMCFOLD_CCD_PATH=/data/models/esmfold2/ccd.pkl
ENV ESMFOLD2_JOBS_BASE_DIR=/data/esmfold2_jobs
RUN mkdir -p /data/esmfold2_jobs

# Output-sink: 当 gateway 传 X-Bioagent-Oss-Prefix 时，framework 把 job dir 镜像到此挂载点。
# 需在 FC 控制台把数据面 OSS bucket 挂到 /mnt/oss。缺挂载 => no-op（仅 NAS）。
ENV ESMFOLD2_OSS_OUTPUT_MOUNT=/mnt/oss

# Keepalive pings this URL through FC's gateway so the instance stays alive
# during long GPU jobs. Override at deploy time if the URL changes.
# ENV ESMFOLD2_KEEPALIVE_URL=https://fc-esmfold-spayusioug.cn-hangzhou-vpc.fcapp.run
# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance. Requires FC console config too.
ENV ESMFOLD2_SESSION_HEADER_NAME=bioagent-session-id
ENV PORT=9000
EXPOSE 9000
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
