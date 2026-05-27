# Build from bioagent project root:
#   docker build --platform linux/amd64 -t genie3-server -f services/genie3-server/Dockerfile .
#
# Genie3 source must be at opensource/genie3/ (pretrained v1 weights baked into image).
# This image only supports `genie3 generate` — evaluation deps (ColabFold/ESMFold/FoldSeek/
# ProteinMPNN/TMscore/DSSP/IPSAE) are intentionally excluded to keep the image small.
# Base: cudnn-runtime (not -devel). genie3 installs from prebuilt torch wheels +
# pip wheels for its other deps; build-essential below is for numpy/Cython
# wheel compilation, not CUDA — that works fine on -runtime too.
# Save ~3.6 GB vs the -devel variant.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror via uv (project-wide convention, ~10× faster from
# China than download.pytorch.org which has no China CDN).
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    git wget curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv binary (matches rfantibody-server's version pin).
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# --- Install Genie3 (generation deps only) ---
# Create isolated venv up front (mirrors rfantibody-server / proteinmpnn-server).
# Genie3 has its own setup.py but no uv.lock, so we drive installs with
# `uv pip install` rather than `uv sync`.
WORKDIR /opt/genie3
RUN uv venv .venv --python python3.10

# Bootstrap build deps before installing the package (some deps need numpy/Cython at build time)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "numpy>=2.0.2,<3" Cython setuptools wheel

# Genie3 generation runtime deps. Keep this list in sync with opensource/genie3/setup.py.
# Notes:
#   - torch==2.7.1 ships with CUDA 12.6 wheels (compatible with the 12.4 base CUDA driver runtime).
#   - PyYAML is required by genie3/cli.py for iterative configs.
#   - We omit wandb/tensorboard server features but keep the python packages so imports work.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        torch==2.7.1 \
        tqdm scipy pandas \
        lightning \
        "biopython<1.86" \
        ml-collections zstandard \
        huggingface_hub \
        wandb tensorboard \
        PyYAML

# Copy genie3 source + pretrained v1 weights (drop training data / examples to slim the image)
COPY opensource/genie3/setup.py /opt/genie3/setup.py
COPY opensource/genie3/src /opt/genie3/src
COPY opensource/genie3/pretrained/v1 /opt/genie3/pretrained/v1

# Apply local patches to upstream genie3. See services/genie3-server/patches/README.md.
COPY services/genie3-server/patches /tmp/genie3-patches
RUN cd /opt/genie3 && for p in /tmp/genie3-patches/*.patch; do \
        echo "Applying $p"; patch -p1 < "$p"; \
    done

# Editable install of genie3 (registers the `genie3` console entry point inside the venv)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python --no-build-isolation -e .

# --- Shared bioagent service framework + remote-fetch deps ---
# Installing the framework as a separate layer keeps cache hits high when only
# the framework's source changes. See services/_framework/ for the package.
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# Server code last — explicit file list so test/patch changes don't bust cache.
COPY services/genie3-server/__init__.py \
     services/genie3-server/VERSION \
     services/genie3-server/app.py \
     services/genie3-server/adapter.py \
     services/genie3-server/configs.py \
     services/genie3-server/datasets.py \
     services/genie3-server/models.py \
     services/genie3-server/settings.py \
     /opt/genie3/server/

# Env-driven settings — see services/genie3-server/settings.py (Genie3Settings).
# The GENIE3_ prefix is read by pydantic-settings; defaults match the image layout.
# GENIE3_BIN points at the venv's console entry, since `.venv/bin` isn't on PATH.
ENV GENIE3_ROOT=/opt/genie3
ENV GENIE3_BIN=/opt/genie3/.venv/bin/genie3
ENV GENIE3_JOBS_BASE_DIR=/data/genie3_jobs
RUN mkdir -p /data/genie3_jobs

# --- Alibaba Cloud FC configuration ---
# CAPort: FC default listening port is 9000
# Keep-alive: FC requires >= 15 minutes (900s)
# Startup: HTTP server must be ready within 120s
ENV GENIE3_MAX_CONCURRENT_JOBS=1
ENV PORT=9000
EXPOSE 9000

CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
