# HTTP wrapper for protein-design-mcp — self-contained build.
#
# Inlines the upstream Dockerfile stages instead of FROM'ing a prebuilt
# protein-design-mcp:latest, then layers the SSE + Streamable-HTTP wrapper
# on top.  The protein-design-mcp upstream source lives at
# services/protein-design-mcp-http/upstream/ (vendored at pinned SHA via
# scripts/vendor.sh).  Wrapper code lives in this directory.
#
# Pre-build (once; re-run to bump pinned upstream commit):
#   ./services/protein-design-mcp-http/scripts/vendor.sh
#
# Build context MUST be the bioagent project root (so services/ is reachable):
#
#   docker build --platform linux/amd64 \
#       -f services/protein-design-mcp-http/Dockerfile \
#       -t protein-design-mcp-http:latest .
#
# Run (CPU):
#   docker run --rm -p 9000:9000 -e DEVICE=cpu protein-design-mcp-http:latest
#
# Run (GPU + persistent model cache):
#   docker run --rm --gpus all -p 9000:9000 \
#       -v $(pwd)/models:/models -v $(pwd)/cache:/cache \
#       protein-design-mcp-http:latest
#
# Smoke-test after start:
#   curl http://localhost:9000/healthz                 # → ok
#   curl -N http://localhost:9000/sse                  # → SSE stream w/ endpoint
#   curl -X POST http://localhost:9000/mcp \
#        -H 'Content-Type: application/json' \
#        -H 'Accept: application/json, text/event-stream' \
#        -d '{"jsonrpc":"2.0","id":1,"method":"initialize",
#             "params":{"protocolVersion":"2025-03-26","capabilities":{},
#                       "clientInfo":{"name":"probe","version":"0.1"}}}'

# =============================================================================
# Stage 1: Base CUDA + Python  (mirrors upstream Dockerfile stage 1)
# =============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror via uv (project-wide convention). The torch install
# in stage 2 pins --index-url to pytorch.org for the +cu118 wheels; everything
# else benefits from Huawei mirror automatically.
# UV_SYSTEM_PYTHON=1 lets `uv pip install` target system Python without
# requiring `--system` on every line (this image has no venv — many editable
# installs target a single shared interpreter).
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# uv binary (matches the pin used by sibling services rfantibody-server /
# proteinmpnn-server / genie3-server).
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# setuptools/wheel still needed for the editable installs below.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --upgrade pip setuptools wheel

# =============================================================================
# Stage 2: Computational backends  (mirrors upstream stage 2)
# =============================================================================
FROM base AS tools

WORKDIR /opt

# PyTorch CUDA 11.8 wheels. +cu118 local-version variants only exist on
# download.pytorch.org — Huawei mirror's default linux torch wheels are cu121.
# --index-url overrides UV_INDEX_URL for this command only.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.0.1+cu118 \
        torchvision==0.15.2+cu118

# --- RFdiffusion ---
RUN git clone --depth 1 https://github.com/RosettaCommons/RFdiffusion.git \
    && mkdir -p RFdiffusion/models

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e RFdiffusion/env/SE3Transformer

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e RFdiffusion

# Pin e3nn==0.5.1 (torch 2.0.x compat, before torch.compiler).
# Install dgl==1.1.3 CUDA 11.8 build (PyPI's dgl is CPU-only — pinned via --find-links).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        hydra-core omegaconf icecream pyrsistent \
        "e3nn==0.5.1" "numpy<2" \
    && uv pip install --find-links https://data.dgl.ai/wheels/cu118/repo.html \
        "dgl==1.1.3"

# --- ProteinMPNN ---
RUN git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git \
    && mkdir -p ProteinMPNN/vanilla_model_weights

# --- ColabFold / AlphaFold2 (API backend by default — no weights baked) ---
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold.git" \
    && mkdir -p /root/.cache/colabfold

RUN ln -sf $(python -c "import colabfold; print(colabfold.__path__[0])")/batch.py \
    /usr/local/bin/colabfold_batch || true

# --- OpenFold v1.0.1 (ESMFold dependency, PYTHONPATH only — no C++ build) ---
RUN git clone --depth 1 --branch v1.0.1 \
    https://github.com/aqlaboratory/openfold.git /opt/openfold \
    && uv pip install biopython dm-tree ml-collections modelcif einops 2>/dev/null || true

# --- OpenMM (energy minimization, AMBER14 force fields) ---
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install openmm

# =============================================================================
# Stage 3: protein-design-mcp source + HTTP wrapper
# =============================================================================
FROM tools AS app

WORKDIR /app

# Upstream protein-design-mcp source — from vendored upstream/ (see
# scripts/vendor.sh).
COPY services/protein-design-mcp-http/upstream/pyproject.toml /app/pyproject.toml
COPY services/protein-design-mcp-http/upstream/src            /app/src
COPY services/protein-design-mcp-http/upstream/docker         /app/docker

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e ".[dev]"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        "fair-esm>=2.0.0" \
        aiohttp \
        tqdm \
    && uv pip install "numpy<2"

# Upstream entrypoint + model downloader scripts (from the clone above).
RUN chmod +x docker/entrypoint.sh docker/download_models.py

# --- Bake model weights into the image (~3.7 GB) ---
# Runs the upstream download_models.py so RFdiffusion (Complex_base_ckpt.pt,
# 1.5 GB), ProteinMPNN (v_48_020.pt, 150 MB) and ESMFold (~2 GB) weights live
# in the image instead of being fetched on first tool call. download_models.py
# also creates the symlinks the upstream code expects under /opt/RFdiffusion/
# and /opt/ProteinMPNN/.
#
# Env vars are set inline because the matching ENV directives appear later in
# this Dockerfile. ESMFold load runs on CPU during build (no GPU); needs
# ~3 GB of build-host RAM. If a download flakes the script logs a warning
# and exits 0, so the build doesn't fail catastrophically — but missing
# weights will surface as runtime errors on the affected tool.
RUN mkdir -p /models /cache /data \
    && MODELS_DIR=/models \
       TORCH_HOME=/models/esm \
       RFDIFFUSION_PATH=/opt/RFdiffusion \
       PROTEINMPNN_PATH=/opt/ProteinMPNN \
       python /app/docker/download_models.py

# --- HTTP wrapper layer (this service) ---
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        "starlette>=0.37,<1.0" \
        "uvicorn[standard]>=0.30,<1.0" \
        "anyio>=4.5,<5.0"

COPY services/protein-design-mcp-http/http_server.py /app/http_server.py

# Tool paths (mirrors upstream stage 4).
ENV RFDIFFUSION_PATH=/opt/RFdiffusion
ENV PROTEINMPNN_PATH=/opt/ProteinMPNN
ENV PYTHONPATH="/opt/openfold:${PYTHONPATH}"

# ColabFold.
ENV COLABFOLD_PATH=/usr/local/bin/colabfold_batch
ENV COLABFOLD_BACKEND=api

# Storage.
ENV MODELS_DIR=/models
ENV CACHE_DIR=/cache
ENV TORCH_HOME=/models/esm
ENV COLABFOLD_WEIGHTS_DIR=/models/colabfold

# Performance.
ENV PYTORCH_JIT_USE_NNC_NOT_NVFUSER=1
ENV DS_BUILD_OPS=0

# Device mode: "auto" / "cuda" / "cpu".
ENV DEVICE=auto

# HTTP wrapper config.
ENV PORT=9000
ENV LOG_LEVEL=info

#VOLUME ["/models", "/data", "/cache"]
EXPOSE 9000

# Reuse upstream entrypoint.sh — it sets DEVICE/MODELS_DIR/etc and falls
# through to `exec "$@"` for non-"server"/"download" CMDs, so uvicorn
# launches after env setup.
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["python", "/app/http_server.py"]

LABEL org.opencontainers.image.title="Protein Design MCP Server (HTTP)"
LABEL org.opencontainers.image.description="protein-design-mcp exposed over MCP SSE + Streamable HTTP transports"
LABEL org.opencontainers.image.source="https://github.com/jasonkim8652/protein-design-mcp"
LABEL org.opencontainers.image.licenses="Apache-2.0"
