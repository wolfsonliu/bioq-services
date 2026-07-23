# Build from bioagent project root:
#   docker build --platform linux/amd64 -t lasermpnn-server -f services/lasermpnn-server/Dockerfile .
#
# Prerequisites (run once on the host; re-run to bump the pinned upstream SHA):
#   ./services/lasermpnn-server/scripts/vendor.sh          # vendor upstream/ (no model_weights)
#   ./services/lasermpnn-server/scripts/fetch_weights.sh   # stage weights, then rsync to NAS
#
# LASErMPNN source comes from services/lasermpnn-server/upstream/ (vendor.sh
# product) — the Dockerfile does no network access to github. Weights load from
# NAS at /data/models/lasermpnn/ (not baked in). SIF/HPC: apptainer --nv --bind.
#
# GPU runtime base (small ~82 MB model, but wheel stack targets CUDA 12.x).
# `-runtime` (not `-devel`): all deps are prebuilt wheels (torch + PyG
# scatter/cluster from data.pyg.org), no nvcc / compiler needed.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror via uv (project-wide convention; PyPI's default linux
# torch wheel bundles CUDA 12.1, which runs on this 12.4 base).
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
# No git in apt: vendor.sh runs on the host, no in-build clone, and no
# `uv pip install "... @ git+..."` VCS specifiers here.

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

WORKDIR /opt/lasermpnn
RUN uv venv .venv --python python3.10

# Torch first (heavy, rarely changes). PyPI torch 2.4.1 linux wheel bundles cu121.
# numpy<2 keeps ABI compatible with the torch_scatter/torch_cluster wheels and
# with ProDy.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python "torch==2.4.1" "numpy<2"

# PyG scatter/cluster prebuilt wheels — keyed by torch MINOR (2.4.0) + cu121.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        --extra-index-url https://data.pyg.org/whl/torch-2.4.0+cu121.html \
        "torch-scatter==2.1.2" "torch-cluster==1.6.3"

# Remaining inference-chain deps (ProDy for PDB IO, pydssp for SS, pykeops only
# to satisfy the ligandmpnn variant's top-level import — JIT is never hit on the
# inference path so no compiler is needed).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        prody pydssp \
        pandas scipy scikit-learn tqdm \
        matplotlib plotly logomaker pykeops

# LASErMPNN upstream — copied from the vendored tree AS the package `LASErMPNN`
# (upstream uses absolute `from LASErMPNN.utils...` imports and is invoked as
# `python -m LASErMPNN.run_batch_inference` from this parent dir). The small
# files/*.pt reference tensors ride along; model_weights/ is excluded (NAS).
COPY services/lasermpnn-server/upstream /opt/lasermpnn/LASErMPNN

# Shared bioagent service framework + remote-fetch deps (late layer). COPY (not
# --mount=bind) so a framework source change busts this layer's cache.
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# Server code last — explicit file list so test/doc changes don't bust cache.
COPY services/lasermpnn-server/__init__.py \
     services/lasermpnn-server/__main__.py \
     services/lasermpnn-server/VERSION \
     services/lasermpnn-server/app.py \
     services/lasermpnn-server/adapter.py \
     services/lasermpnn-server/models.py \
     services/lasermpnn-server/settings.py \
     services/lasermpnn-server/tools.py \
     /opt/lasermpnn/server/

# Import smoke in the runtime stage (single-stage here) — catch missing runtime
# libs the builder-only tools would otherwise mask.
RUN .venv/bin/python -c "import torch, torch_scatter, torch_cluster, prody, pydssp; print('deps ok')" \
    && PYTHONPATH=/opt/lasermpnn .venv/bin/python -c "from server.app import app; print(app.title)"

ENV PYTHONPATH=/opt/lasermpnn
ENV LASERMPNN_ROOT=/opt/lasermpnn
ENV LASERMPNN_WEIGHTS_DIR=/data/models/lasermpnn
ENV LASERMPNN_JOBS_BASE_DIR=/data/lasermpnn_jobs
ENV LASERMPNN_DEVICE=cuda:0
RUN mkdir -p /data/lasermpnn_jobs

# Output-sink: when the gateway passes X-Bioagent-Oss-Prefix, the framework
# mirrors the finished job dir to this mount. Requires an OSS mount at /mnt/oss
# in the FC console. Absent mount => no-op (NAS-only).
ENV LASERMPNN_OSS_OUTPUT_MOUNT=/mnt/oss
# FC session affinity header (requires FC console config too).
ENV LASERMPNN_SESSION_HEADER_NAME=bioagent-session-id

ENV PORT=9000
EXPOSE 9000
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
