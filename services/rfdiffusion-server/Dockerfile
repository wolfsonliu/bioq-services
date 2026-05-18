# Build from bioagent project root:
#   docker build --platform linux/amd64 -t rfdiffusion-server -f services/rfdiffusion-server/Dockerfile .
#
# RFdiffusion source must be at opensource/RFdiffusion/ (weights baked into image,
# ~4 GB of .pt files; ensure opensource/RFdiffusion/models/ is populated first).
#
# Base: cudnn8-runtime on CUDA 11.8 (no nvcc/headers needed at runtime).
# RFdiffusion + SE3-Transformer install entirely from prebuilt wheels (torch +
# dgl + e3nn + hydra-core). Verified with grep across the source tree — no
# `cpp_extension.load` / `CUDAExtension` paths run at inference time.
# Save ~3-4 GB vs the -devel variant.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror (project-wide convention, ~10× faster than upstream
# pypi.org from China build hosts). UV_INDEX_URL is consumed by `uv pip install`.
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
# DGL chooses its backend at import; pytorch is the only one we ship.
ENV DGLBACKEND=pytorch

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    git wget curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv binary (matches rfantibody-server / genie3-server version pin).
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# ---------------------------------------------------------------------------
# Create the isolated venv RFdiffusion runs from. Mirrors rfantibody-server's
# layout: /opt/<service>/.venv with `uv pip install` driving all installs.
# ---------------------------------------------------------------------------
WORKDIR /opt/rfdiffusion
RUN uv venv .venv --python python3.10

# ---------------------------------------------------------------------------
# Core scientific stack first. Ordering matters for cache hit rates: this layer
# is the largest (torch ~2 GB) and changes least often.
#
# Wheel sources:
#   * torch 2.0.1 + cu118 — from download.pytorch.org (only place these live).
#     We use the standard 11.8 CUDA wheels for compatibility with the base image.
#   * dgl 1.1.3+cu118 — from data.dgl.ai. Upstream RFdiffusion docker pins
#     dgl 1.0.2+cu116; the +cu118 build is the closest match for our base.
#   * everything else from Huawei mirror via UV_INDEX_URL.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        torch==2.0.1+cu118

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        --find-links https://data.dgl.ai/wheels/cu118/repo.html \
        dgl==1.1.3+cu118

# ---------------------------------------------------------------------------
# RFdiffusion's remaining python deps — pinned to the versions the upstream
# Dockerfile uses (verified compatible with torch 2.0.1 and hydra 1.3.2).
# pynvml/decorator/dllogger are required by the NVIDIA SE3-Transformer source.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        e3nn==0.3.3 \
        hydra-core==1.3.2 \
        pyrsistent==0.19.3 \
        decorator==5.1.0 \
        pynvml==11.0.0 \
        wandb==0.12.0 \
        "numpy<2"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "dllogger @ git+https://github.com/NVIDIA/dllogger@v1.0.0"

# ---------------------------------------------------------------------------
# Copy RFdiffusion source + pretrained weights. Heavy COPY late so the
# previous wheel layers stay cached when only source/config changes.
#
# The models/ dir is ~4 GB (8 × 460 MB checkpoints + 1 × 230 MB RF2 weights).
# Make sure `opensource/RFdiffusion/models/` is populated locally before
# building — `bash opensource/RFdiffusion/scripts/download_models.sh
# opensource/RFdiffusion/models` will fetch them.
# ---------------------------------------------------------------------------
COPY opensource/RFdiffusion /opt/rfdiffusion

# NVIDIA SE3-Transformer is vendored under env/SE3Transformer; install it
# (pure-python, no nvcc needed).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python --no-build-isolation \
        /opt/rfdiffusion/env/SE3Transformer

# Editable install of rfdiffusion itself so `from rfdiffusion ...` works inside
# the inference script and any auxiliary code.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python --no-build-isolation --no-deps \
        -e /opt/rfdiffusion

# ---------------------------------------------------------------------------
# Shared bioagent service framework + remote-fetch deps. Late layer so framework
# source changes don't invalidate the torch / dgl wheel layers above.
# ---------------------------------------------------------------------------
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# Server code as `server` package — uvicorn imports `server.app:app`.
COPY services/rfdiffusion-server/ /opt/rfdiffusion/server/

# Settings (env-driven; see services/rfdiffusion-server/settings.py).
ENV RFDIFFUSION_ROOT=/opt/rfdiffusion
ENV RFDIFFUSION_MODELS_DIR=/opt/rfdiffusion/models
ENV RFDIFFUSION_INFERENCE_SCRIPT=/opt/rfdiffusion/scripts/run_inference.py
ENV RFDIFFUSION_PYTHON=/opt/rfdiffusion/.venv/bin/python
ENV RFDIFFUSION_JOBS_BASE_DIR=/data/rfdiffusion_jobs
RUN mkdir -p /data/rfdiffusion_jobs

# Alibaba Cloud FC: CAPort 9000, keep-alive ≥ 15 min, /healthz must be ready < 120 s.
ENV PORT=9000
EXPOSE 9000
CMD ["/opt/rfdiffusion/.venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
