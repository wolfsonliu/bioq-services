# Build from bioagent project root:
#   docker build --platform linux/amd64 -t rfdiffusion-server -f services/rfdiffusion-server/Dockerfile .
#
# RFdiffusion source must be at opensource/RFdiffusion/ (weights baked into image,
# ~3.9 GB of .pt files; ensure opensource/RFdiffusion/models/ is populated via
# `bash opensource/RFdiffusion/scripts/download_models.sh ...` before building).
#
# Multistage layout
# -----------------
#   builder = heavy: apt build-essential + python-dev + git/wget for wheel/git installs.
#   runtime = lean : only python3.10 + ca-certificates; the finished .venv + source +
#                    models are copied over.
#
# Why multistage: build-essential / python3.10-dev / git / wget pull in ~300 MB
# of headers + gcc + libs the runtime never uses. In a single-stage Dockerfile
# `apt-get remove` after install would leave the bytes in the original apt
# install layer — the only way to actually shrink is to never copy them into
# the final image.
#
# Base: cudnn8-RUNTIME variant of nvidia/cuda 11.8 in BOTH stages. RFdiffusion
# + SE3-Transformer install entirely from prebuilt wheels (torch + dgl + e3nn
# + hydra-core); the source tree has no `cpp_extension.load` / `CUDAExtension`
# paths that run at inference time, so the runtime base saves ~3-4 GB vs `-devel`.
ARG BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04


# ===========================================================================
# Stage 1: builder — installs the venv, copies source + models in.
# ===========================================================================
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror (project-wide convention, ~10× faster than upstream
# pypi.org from China build hosts). UV_INDEX_URL is consumed by `uv pip install`.
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/

# Build-only apt deps. `python3.10-dev` + `build-essential` are required by
# legacy setup.py packages (SE3Transformer, rfdiffusion root) that build sdist
# wheels at install time. `git` is needed by the `dllogger @ git+...` install.
# `wget`/`curl`/`ca-certificates` are used by uv's wheel cache fetches.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    git wget curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv binary (matches rfantibody-server / genie3-server version pin).
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# ---------------------------------------------------------------------------
# Create the venv and seed it with setuptools + wheel.
#
# Why seed setuptools/wheel up-front: two later steps install legacy setup.py-
# only packages (`env/SE3Transformer/`, the rfdiffusion source root) with
# `--no-build-isolation`. Without isolation uv looks for the build backend in
# THIS venv and crashes with `ModuleNotFoundError: setuptools`. Tiny layer
# (~2 MB) — keeps the foundation independent of torch's slow cache.
# ---------------------------------------------------------------------------
WORKDIR /opt/rfdiffusion
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv .venv --python python3.10 \
    && uv pip install --python .venv/bin/python setuptools wheel

# ---------------------------------------------------------------------------
# Core scientific stack — one wheel per layer so the slow downloads
# (torch ~2 GB, dgl ~300 MB) cache independently.
#
# Wheel sources:
#   * torch 2.0.1 + cu118 — from download.pytorch.org. CUDA 11.8 wheels for
#     compatibility with the base image.
#   * dgl 1.1.3 + cu118 — from data.dgl.ai. Upstream RFdiffusion pins dgl
#     1.0.2+cu116; +cu118 is the closest match for our base.
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
# Remaining RFdiffusion python deps in ONE layer — versions pinned to what
# upstream Dockerfile uses (verified compatible with torch 2.0.1 + hydra 1.3.2).
# pynvml / decorator / dllogger are pulled in by NVIDIA SE3-Transformer.
#
# Combined into a single uv call instead of 2 RUNs: 4× smaller manifest cost,
# one resolver pass instead of two, and these pins all change in lockstep so
# splitting them doesn't help cache hit rates in practice.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        e3nn==0.3.3 \
        hydra-core==1.3.2 \
        pyrsistent==0.19.3 \
        decorator==5.1.0 \
        pynvml==11.0.0 \
        wandb==0.12.0 \
        "numpy<2" \
        "dllogger @ git+https://github.com/NVIDIA/dllogger@v1.0.0"

# ---------------------------------------------------------------------------
# Source + weights. Split into TWO COPYs to keep the 3.9 GB models layer
# stable across source iterations:
#   1. models/        — pretrained checkpoints, rarely change, big.
#   2. code subdirs   — rfdiffusion package, env/, config/, scripts/.
#
# We deliberately skip `img/` (7 MB of PNG screenshots), `docker/`,
# `tests/` — none of them are referenced by run_inference.py.
#
# `examples/input_pdbs/` MUST be copied: rfdiffusion/inference/model_runners.py
# falls back to `examples/input_pdbs/1qys.pdb` as the placeholder target
# whenever `inference.input_pdb` is None. Endpoints that don't ship a PDB
# (unconditional / symmetry / freeform-without-pdb) crash with FileNotFound
# at sampler init if this directory is missing. The rest of `examples/`
# (shell scripts / output PDBs) is not on any code path → skipped.
# ---------------------------------------------------------------------------
COPY opensource/RFdiffusion/models /opt/rfdiffusion/models
COPY opensource/RFdiffusion/rfdiffusion /opt/rfdiffusion/rfdiffusion
COPY opensource/RFdiffusion/env /opt/rfdiffusion/env
COPY opensource/RFdiffusion/config /opt/rfdiffusion/config
COPY opensource/RFdiffusion/scripts /opt/rfdiffusion/scripts
COPY opensource/RFdiffusion/helper_scripts /opt/rfdiffusion/helper_scripts
COPY opensource/RFdiffusion/examples/input_pdbs /opt/rfdiffusion/examples/input_pdbs
COPY opensource/RFdiffusion/setup.py /opt/rfdiffusion/setup.py

# ---------------------------------------------------------------------------
# Three editable / wheel-from-source installs + shared framework, fused into
# ONE layer with cleanup. None of these change in isolation in practice, and
# combining them lets us strip `__pycache__` / build leftovers in the same
# layer (deletion in a later RUN wouldn't reclaim bytes from earlier layers).
#
#   * SE3-Transformer (vendored under env/SE3Transformer; pure-python, no nvcc)
#   * rfdiffusion editable install (--no-deps so we don't re-resolve the heavy
#     stack we already installed; uses the source at /opt/rfdiffusion/rfdiffusion)
#   * service framework + httpx + alibabacloud-oss-v2
# ---------------------------------------------------------------------------
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python --no-build-isolation \
        /opt/rfdiffusion/env/SE3Transformer \
    && uv pip install --python .venv/bin/python --no-build-isolation --no-deps \
        -e /opt/rfdiffusion \
    && uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2 \
    && rm -rf /tmp/service-framework \
    && find /opt/rfdiffusion/.venv -depth -type d -name '__pycache__' -exec rm -rf {} + \
    && find /opt/rfdiffusion/.venv -depth -type d -name 'tests' -exec rm -rf {} + \
    && find /opt/rfdiffusion/.venv -name '*.pyc' -delete


# ===========================================================================
# Stage 2: runtime — minimal apt surface + the finished venv from builder.
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# DGL chooses its backend at import; pytorch is the only one we ship.
ENV DGLBACKEND=pytorch

# Runtime-only apt: python3.10 (the venv's shebang `.venv/bin/python` symlinks
# to /usr/bin/python3.10) + ca-certificates (httpx HTTPS for OSS / URL inputs).
# No gcc, no headers, no git — that all stayed in the builder stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The whole prepared tree comes over in one COPY: venv + source + models +
# editable install metadata. ~7-8 GB in a single layer, which is what we want
# for registry dedup (harbor stores it once across all FC pulls).
COPY --from=builder /opt/rfdiffusion /opt/rfdiffusion

# WORKDIR MUST be set in the runtime stage (it doesn't propagate from the
# builder stage). Without it `cwd` defaults to `/`, and uvicorn's
# `server.app:app` import looks for `server` under cwd + sys.path → fails with
# `ModuleNotFoundError: No module named 'server'` at FC cold start.
WORKDIR /opt/rfdiffusion

# Server code last — explicit file list so test/doc changes don't bust cache.
COPY services/rfdiffusion-server/__init__.py \
     services/rfdiffusion-server/VERSION \
     services/rfdiffusion-server/app.py \
     services/rfdiffusion-server/adapter.py \
     services/rfdiffusion-server/models.py \
     services/rfdiffusion-server/settings.py \
     services/rfdiffusion-server/tools.py \
     services/rfdiffusion-server/uris.py \
     /opt/rfdiffusion/server/

# Settings (env-driven; see services/rfdiffusion-server/settings.py).
ENV RFDIFFUSION_ROOT=/opt/rfdiffusion
ENV RFDIFFUSION_MODELS_DIR=/opt/rfdiffusion/models
ENV RFDIFFUSION_INFERENCE_SCRIPT=/opt/rfdiffusion/scripts/run_inference.py
ENV RFDIFFUSION_PYTHON=/opt/rfdiffusion/.venv/bin/python
ENV RFDIFFUSION_JOBS_BASE_DIR=/data/rfdiffusion_jobs
RUN mkdir -p /data/rfdiffusion_jobs

# Alibaba Cloud FC: port 9000, keep-alive ≥ 15 min, /healthz must be ready < 120 s.
ENV RFDIFFUSION_MAX_CONCURRENT_JOBS=1
ENV PORT=9000
EXPOSE 9000
CMD ["/opt/rfdiffusion/.venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
