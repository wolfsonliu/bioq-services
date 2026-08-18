# Build from bioagent project root:
#   docker build --platform linux/amd64 -t turbohopp-server \
#       -f services/turbohopp-server/Dockerfile .
#
# Pre-build (once; re-run to bump pinned upstream commit / refresh weights):
#   ./services/turbohopp-server/scripts/vendor.sh          # → upstream/
#   # (no fetch_weights.sh: upstream does not publish public checkpoints;
#   #  deployer supplies them via rsync to NAS — see README.md ## Weights)
#
# TurboHopp source lives at services/turbohopp-server/upstream/ (vendored at
# pinned SHA via scripts/vendor.sh).  The consistency-model .ckpt lives on
# NAS at /data/models/turbohopp/checkpoints/v1/ — NOT baked into the image.
# See engineering/decisions/2026-06-26-service-weights-externalization.md.
#
# Multistage layout
# -----------------
#   builder = conda env from upstream environment.yml (Python 3.10 +
#             pytorch 2.2 + cu121 + PyG 2.5 + rdkit + openmm + pdbfixer) +
#             framework + cleanup.
#   runtime = lean: conda env from builder + TurboHopp source + server code.
#
# Base: nvidia/cuda 12.1 -runtime (matches upstream env's pytorch 2.2 +
# cu121 wheels).  Upstream's environment.yml also lists cudatoolkit=11.8
# in the conda channel section, but the pip section installs cu121 wheels
# for torch/torch-scatter/etc — we align to the pip side.

ARG BASE_IMAGE=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
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
ENV PATH=/opt/conda/bin:$PATH

# ---------------------------------------------------------------------------
# Mirror configuration (Huawei pip + PKU conda) for CN-friendly builds.
# ---------------------------------------------------------------------------
COPY deploy/conda/mirrors.condarc /tmp/mirrors.condarc
RUN mkdir -p /root/.config/pip \
    && printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
       > /root/.config/pip/pip.conf \
    && cat > /root/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
channel_priority: strict
EOF
RUN cat /tmp/mirrors.condarc >> /root/.condarc

# ---------------------------------------------------------------------------
# Conda env — installed via micromamba as a lean subset of upstream's
# environment.yml.  We skip mujoco / d4rl / gymnasium / dm_control etc. —
# those are used by upstream's train_rlcm.py (RL fine-tuning), not the
# inference path.  Same rationale for wandb (we set WANDB_MODE=disabled in
# inference.py).
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba install -y -n base \
        -c conda-forge -c bioconda \
        python=3.10 \
        pip \
        numpy=1.26.4 \
        openmm=8.1.1 \
        pdbfixer=1.9 \
        openbabel=3.1.1 \
    && micromamba clean -a -y

# ---------------------------------------------------------------------------
# pip install: pytorch 2.2 + cu121 + PyG stack + rdkit + supporting libs.
# torch-{scatter,cluster,sparse,spline-conv,pyg-lib} come from PyG's wheel
# index; other pieces come from the Huawei pip mirror.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/bin/python \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.2.2 torchvision==0.17.2

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/bin/python \
        --extra-index-url https://data.pyg.org/whl/torch-2.2.0+cu121.html \
        torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 \
        torch-spline-conv==1.2.2 pyg-lib==0.4.0 \
        torch-geometric==2.5.2

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/bin/python \
        rdkit==2023.9.5 \
        biopandas==0.4.1 biopython==1.83 \
        pytorch-lightning==2.2.1 lightning==2.2.1 lightning-utilities==0.11.2 \
        torchmetrics==1.3.2 \
        pandas==2.2.1 scipy==1.12.0 scikit-learn==1.4.1.post1 \
        matplotlib==3.8.3 \
        omegaconf==2.3.0 hydra-core==1.3.2 easydict==1.13 \
        tqdm==4.66.2 pyyaml==6.0.1 \
        h5py==3.11.0 lmdb==1.4.1 \
        # TurboHopp's sascorer.py imports meeko for some auxiliary utilities.
        meeko==0.5.0

# ---------------------------------------------------------------------------
# TurboHopp source — from vendored upstream/ (see scripts/vendor.sh).
# Upstream has no setup.py / pyproject; we import via PYTHONPATH at runtime.
# ---------------------------------------------------------------------------
COPY services/turbohopp-server/upstream /opt/turbohopp
RUN /opt/conda/bin/python -c \
    "import sys; sys.path.insert(0, '/opt/turbohopp'); \
     import diffusion_hopping; print('diffusion_hopping OK'); \
     from consistency.models_consistency import karras_schedule; \
     print('consistency.models_consistency OK')"

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env.
# ---------------------------------------------------------------------------
RUN /opt/conda/bin/pip uninstall -y \
        jupyterlab jupyter_server jupyterlab_server jupyter_client \
        ipywidgets ipykernel notebook nbconvert nbformat || true \
    && find /opt/conda/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/share/doc \
        /opt/conda/share/man \
        /opt/conda/share/locale \
    && find /opt/conda/lib/python3.10/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda -depth -type d \
        -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Shared service framework — independent layer so framework bumps don't
# invalidate the conda env layer above.
# ---------------------------------------------------------------------------
RUN --mount=type=bind,source=framework,target=/tmp/service-framework \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2


# ===========================================================================
# Stage 2: runtime — conda env + TurboHopp source + server code.
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:$PATH

# openbabel + openmm need libgomp; ca-certificates for OSS / HTTP.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Conda env from builder.
COPY --from=builder /opt/conda /opt/conda

# TurboHopp source from builder (vendored upstream).
COPY --from=builder /opt/turbohopp /opt/turbohopp

# Server code as `server` package — uvicorn imports `server.app:app`.
COPY services/turbohopp-server/__init__.py \
     services/turbohopp-server/__main__.py \
     services/turbohopp-server/VERSION \
     services/turbohopp-server/app.py \
     services/turbohopp-server/adapter.py \
     services/turbohopp-server/models.py \
     services/turbohopp-server/settings.py \
     services/turbohopp-server/tools.py \
     services/turbohopp-server/inference.py \
     /opt/turbohopp-server/server/

WORKDIR /opt/turbohopp

# PYTHONPATH — turbohopp upstream root (so `diffusion_hopping` +
# `consistency` + `utils` imports resolve) plus the server package parent.
# Apptainer/SIF ignores WORKDIR, so both must be in PYTHONPATH explicitly.
ENV PYTHONPATH=/opt/turbohopp:/opt/turbohopp-server
ENV TURBOHOPP_ROOT=/opt/turbohopp
ENV TURBOHOPP_PYTHON=/opt/conda/bin/python
ENV TURBOHOPP_INFERENCE_SCRIPT=/opt/turbohopp-server/server/inference.py
ENV TURBOHOPP_JOBS_BASE_DIR=/data/turbohopp_jobs

# Weights externalized — NAS mount injected by FC config / SIF --bind.
ENV TURBOHOPP_WEIGHTS_DIR=/data/models/turbohopp/checkpoints/v1

# wandb is a hard dep of upstream imports; disable at every entry point.
ENV WANDB_MODE=disabled
ENV WANDB_DISABLED=true
ENV WANDB_SILENT=true

RUN mkdir -p /data/turbohopp_jobs

# FC session affinity + concurrency.
ENV TURBOHOPP_SESSION_HEADER_NAME=bioagent-session-id
ENV TURBOHOPP_MAX_CONCURRENT_JOBS=1

ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/bin/python", "-m", "uvicorn", \
     "server.app:app", "--host", "0.0.0.0", "--port", "9000", \
     "--timeout-keep-alive", "900"]
