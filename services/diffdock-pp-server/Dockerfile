# Build from bioagent project root:
#   docker build --platform linux/amd64 -t diffdock-pp-server \
#       -f services/diffdock-pp-server/Dockerfile .
#
# Pre-build (once; re-run to bump pinned upstream commit / refresh weights):
#   ./services/diffdock-pp-server/scripts/vendor.sh         # → upstream/
#   ./services/diffdock-pp-server/scripts/fetch_weights.sh  # → weights/ (score + confidence + ESM-2 + esm src)
#
# DiffDock-PP source lives at services/diffdock-pp-server/upstream/
# (vendored at pinned SHA via scripts/vendor.sh).  Weights (~22 MB checkpoints
# + ~2.5 GB ESM-2 + esm src) live on NAS at /data/models/diffdock-pp/ —
# NOT baked into the image (FC NAS mount; SIF / HPC `apptainer run --bind`).
# See engineering/decisions/2026-06-26-service-weights-externalization.md
# and engineering/decisions/2026-07-03-diffdock-pp-server-design.md.
#
# Multistage layout
# -----------------
#   builder = conda env (Python 3.10 + pytorch 1.13.0 + cu117 + PyG 2.2 +
#             e3nn + biopython + fair-esm + tensorboardX) + framework + cleanup.
#   runtime = lean: conda env from builder + DiffDock-PP source + server code.
#
# Base: nvidia/cuda 11.8 -runtime (forward-compatible with conda's
# pytorch-cuda=11.7; pytorch 1.13.0 has no >11.7 wheels).
# CUDA 11.8 driver back-compatible with cu117 binaries — verified on the
# diffusion-hopping-server build (same conda pin).

ARG BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ARG MICROMAMBA_IMAGE=mambaorg/micromamba:2.6.2


# ===========================================================================
# Stage 1: builder — conda env + framework.
# ===========================================================================
FROM ${MICROMAMBA_IMAGE} AS micromamba
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

# NB: no `git` in apt — vendor.sh ran on the host; the image never clones.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
ENV MAMBA_EXE=/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/diffdock_pp/bin:$PATH

# ---------------------------------------------------------------------------
# Mirror configuration (Huawei pip + TUNA conda) for CN-friendly builds.
# Same rationale as diffusion-hopping-server: strict channel_priority is
# essential for old pytorch/pyg resolves.
# ---------------------------------------------------------------------------
RUN mkdir -p /root/.config/pip \
    && printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
       > /root/.config/pip/pip.conf \
    && cat > /root/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
channel_priority: strict
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
# custom_channels: only map channels that TUNA actually hosts.
# Do NOT map `pyg` or `nvidia` — TUNA returns 404 for those; they fall
# through to the upstream conda.anaconda.org URLs which work fine.
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# ---------------------------------------------------------------------------
# Conda env — matches upstream DiffDock-PP README dependencies (Python
# 3.10 + pytorch 1.13.0 + cu117 + PyG 2.2 + torch-scatter/sparse/cluster/
# spline-conv + e3nn + biopython + biopandas). Skip wandb (training-only)
# and matplotlib (visualization plots).
#
# ittapi: same issue as diffusion-hopping-server. pytorch 1.13's
# libtorch_cpu.so weakly references `iJIT_NotifyEvent` (Intel VTune ITT).
# mkl 2024.1+ removed the symbol, so `import torch` blows up unless
# conda-forge's `ittapi` provides libjitprofiling.so independently.
# See engineering/guides/pytorch-conda-ittapi.md.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n diffdock_pp \
        -c pyg -c nvidia -c pytorch -c conda-forge \
        python=3.10 \
        pytorch::pytorch=1.13.0 \
        pytorch-cuda=11.7 \
        torchvision=0.14.0 \
        pyg::pyg=2.2.0 \
        pyg::pytorch-scatter=2.1.0 \
        pyg::pytorch-sparse=0.6.15 \
        pyg::pytorch-cluster \
        pyg::pytorch-spline-conv \
        conda-forge::e3nn=0.5.1 \
        conda-forge::biopython=1.80 \
        conda-forge::biopandas=0.4.1 \
        conda-forge::numpy=1.24 \
        conda-forge::pandas=1.5 \
        conda-forge::scikit-learn \
        conda-forge::scipy \
        conda-forge::pyyaml \
        conda-forge::dill \
        conda-forge::tqdm \
        conda-forge::tensorboardx \
        conda-forge::tensorboard \
        conda-forge::ittapi

# fair-esm from PyPI (not on conda-forge). Deferred so the base env cache
# isn't invalidated when we bump only this.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/diffdock_pp/bin/python \
        fair-esm

# ---------------------------------------------------------------------------
# DiffDock-PP source — from vendored upstream/ (see scripts/vendor.sh).
# Upstream has no setup.py; we import via PYTHONPATH at runtime instead of
# `pip install -e`.  Do a smoke import to catch missing conda deps early.
# ---------------------------------------------------------------------------
COPY services/diffdock-pp-server/upstream /opt/diffdock-pp
RUN /opt/conda/envs/diffdock_pp/bin/python -c \
    "import sys; sys.path.insert(0, '/opt/diffdock-pp/src'); \
     from geom_utils import set_time; print('diffdock_pp OK')"

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env (mirrors diffusion-hopping cleanup).
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/diffdock_pp/bin/pip uninstall -y \
        jupyterlab jupyter_server jupyterlab_server jupyter_client \
        ipywidgets ipykernel notebook nbconvert nbformat || true \
    && find /opt/conda/envs/diffdock_pp/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/diffdock_pp/share/doc \
        /opt/conda/envs/diffdock_pp/share/man \
        /opt/conda/envs/diffdock_pp/share/locale \
    && find /opt/conda/envs/diffdock_pp/lib/python3.10/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/diffdock_pp -depth -type d \
        -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/diffdock_pp -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Shared service framework — independent layer so framework bumps don't
# invalidate the conda env layer above.
# ---------------------------------------------------------------------------
RUN --mount=type=bind,source=services/_framework,target=/tmp/service-framework \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/diffdock_pp/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2


# ===========================================================================
# Stage 2: runtime — conda env + DiffDock-PP source + server code.
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/diffdock_pp/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Conda env from builder.
COPY --from=builder /opt/conda /opt/conda

# DiffDock-PP source from builder (vendored upstream).
COPY --from=builder /opt/diffdock-pp /opt/diffdock-pp

# Server code as `server` package — uvicorn imports `server.app:app`.
# Bundled config yaml lands next to inference.py so DIFFDOCK_PP_CONFIG_YAML
# default resolves without extra COPY commands.
COPY services/diffdock-pp-server/__init__.py \
     services/diffdock-pp-server/__main__.py \
     services/diffdock-pp-server/VERSION \
     services/diffdock-pp-server/app.py \
     services/diffdock-pp-server/adapter.py \
     services/diffdock-pp-server/models.py \
     services/diffdock-pp-server/settings.py \
     services/diffdock-pp-server/tools.py \
     services/diffdock-pp-server/uris.py \
     services/diffdock-pp-server/inference.py \
     services/diffdock-pp-server/single_pair_inference.yaml \
     /opt/diffdock-pp/server/

WORKDIR /opt/diffdock-pp

# PYTHONPATH — upstream repo root (for `server` package) AND src/ (upstream
# uses flat imports like `from args import parse_args` — src/ must be on
# sys.path).  Apptainer/SIF ignores WORKDIR, so PYTHONPATH must contain both
# explicitly.
ENV PYTHONPATH=/opt/diffdock-pp:/opt/diffdock-pp/src
ENV DIFFDOCK_PP_ROOT=/opt/diffdock-pp
ENV DIFFDOCK_PP_PYTHON=/opt/conda/envs/diffdock_pp/bin/python
ENV DIFFDOCK_PP_INFERENCE_SCRIPT=/opt/diffdock-pp/server/inference.py
ENV DIFFDOCK_PP_CONFIG_YAML=/opt/diffdock-pp/server/single_pair_inference.yaml
ENV DIFFDOCK_PP_JOBS_BASE_DIR=/data/diffdock_pp_jobs

# Weights externalized — NAS mount injected by FC config / SIF --bind.
ENV DIFFDOCK_PP_WEIGHTS_DIR=/data/models/diffdock-pp

# Point torch.hub + HuggingFace caches at the NAS-mounted esm_cache dir so
# `torch.hub.load("facebookresearch/esm:main", ...)` runs offline.  These
# echo the settings.torchhub_dir default; also set by inference.py at
# runtime as belt-and-suspenders.
ENV TORCH_HOME=/data/models/diffdock-pp/esm_cache
ENV HF_HOME=/data/models/diffdock-pp/esm_cache/hf_cache

RUN mkdir -p /data/diffdock_pp_jobs

# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance.
ENV DIFFDOCK_PP_SESSION_HEADER_NAME=bioagent-session-id
ENV DIFFDOCK_PP_MAX_CONCURRENT_JOBS=1

ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/diffdock_pp/bin/python", "-m", "uvicorn", \
     "server.app:app", "--host", "0.0.0.0", "--port", "9000", \
     "--timeout-keep-alive", "900"]
