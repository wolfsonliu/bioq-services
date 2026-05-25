# Build from bioagent project root:
#   docker build --platform linux/amd64 -t deeprank-ab-server -f services/deeprank-ab-server/Dockerfile .
#
# Prereqs (both gitignored, populate before build):
#   ./services/deeprank-ab-server/scripts/fetch_esm_weights.sh   # downloads ESM-2 weights
# See engineering/decisions/2026-05-24-deeprank-ab-server-design.md.
#
# Multistage layout
# -----------------
#   builder = conda env only: micromamba env create + pip extras + framework +
#             cleanup. No upstream source or weights — those go directly into
#             the runtime stage so builder cache survives source edits.
#   runtime = lean: conda env from builder + upstream source + ESM weights +
#             server code.
#
# Layer ordering (stable → volatile):
#   apt/micromamba → mirrors → conda env → pip extras → ittapi → framework
#   → cleanup ‖ runtime: conda env → upstream source → ESM weights → server .py
#
# Base: CUDA 11.8 -devel (matching pytorch-cuda=11.8 in the conda env).
# Using devel for both stages for safety — runtime would suffice since
# DeepRank-Ab has no JIT CUDA compilation, but devel avoids edge cases
# with torch extensions.
ARG BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG MICROMAMBA_IMAGE=mambaorg/micromamba:2.6.2


# ===========================================================================
# Stage 1: builder — conda env only (no upstream source or weights).
# ===========================================================================
FROM ${MICROMAMBA_IMAGE} AS micromamba
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=micromamba /bin/micromamba /bin/micromamba
ENV MAMBA_EXE=/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/deeprank-ab/bin:$PATH

# ---------------------------------------------------------------------------
# Mirror configuration for pip + conda.
# ---------------------------------------------------------------------------
RUN mkdir -p /root/.config/pip \
    && printf '[global]\nindex-url = https://mirrors.aliyun.com/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
       > /root/.config/pip/pip.conf \
    && cat > /root/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# ---------------------------------------------------------------------------
# Create the conda env from the server's curated environment file.
# (upstream pins python=3.9; this copy upgrades to 3.10, drops unused
#  packages, and relaxes transitive pins — see environment-gpu.yml)
# ---------------------------------------------------------------------------
COPY services/deeprank-ab-server/environment-gpu.yml /tmp/deeprank-ab-env/environment-gpu.yml

RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    micromamba env create -y -n deeprank-ab -f /tmp/deeprank-ab-env/environment-gpu.yml \
    && rm -rf /tmp/deeprank-ab-env

# ---------------------------------------------------------------------------
# Pip extras: ANARCI for CDR annotation.
#
# The upstream annotate.py does `from anarci import anarci` and passes the
# bundled hmmscan binary via hmmerpath=. ANARCI ships its own HMM profiles.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/conda/envs/deeprank-ab/bin/pip install anarci

# PyTorch 2.0.1 (conda) links against libjitprofiling.so (Intel VTune JIT
# profiling API, provides iJIT_NotifyEvent). conda-forge ittapi installs the
# .so into the env's lib/ where the dynamic linker finds it. If the conda
# package is unavailable, fall back to a gcc stub (the symbol is a no-op
# profiling callback).
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba install -n deeprank-ab -c conda-forge -y ittapi \
    || (echo 'int iJIT_NotifyEvent(int t, void *d) { return 0; }' | \
        gcc -shared -o /opt/conda/envs/deeprank-ab/lib/libjitprofiling.so -x c -)

# ---------------------------------------------------------------------------
# Install the shared service framework.
# ---------------------------------------------------------------------------
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/conda/envs/deeprank-ab/bin/pip install \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2 \
    && rm -rf /tmp/service-framework

# Build-time import smoke — catches missing symbols (e.g. iJIT_NotifyEvent
# from ittapi) at build, not at FC deploy time.
RUN /opt/conda/envs/deeprank-ab/bin/python -c "import torch; print('torch', torch.__version__)"

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env (~1+ GB).
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/deeprank-ab/bin/pip uninstall -y \
        jupyterlab jupyter_server jupyterlab_server jupyter_client \
        ipywidgets ipykernel notebook nbconvert nbformat || true \
    && find /opt/conda/envs/deeprank-ab/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/deeprank-ab/share/doc \
        /opt/conda/envs/deeprank-ab/share/man \
        /opt/conda/envs/deeprank-ab/share/locale \
    && find /opt/conda/envs/deeprank-ab/lib/python3.10/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/deeprank-ab -depth -type d -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/deeprank-ab -name '*.pyc' -delete


# ===========================================================================
# Stage 2: runtime — conda env + upstream source + weights + server code.
#
# Layer order (stable → volatile):
#   1. apt             — almost never changes
#   2. conda env       — changes only when environment-gpu.yml or framework changes
#   3. upstream source — changes when DeepRank-Ab upstream updates (rare)
#   4. ESM weights     — changes when new weights are fetched (~2.6 GB, very rare)
#   5. server code     — changes on every iteration (explicit file list)
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/deeprank-ab/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/conda /opt/conda

COPY opensource/DeepRank-Ab /opt/deeprank-ab/DeepRank-Ab
COPY services/deeprank-ab-server/weights/esm /opt/deeprank-ab/esm_weights

WORKDIR /opt/deeprank-ab

# Server code last — explicit file list so test/script/weight changes don't
# invalidate this layer.
COPY services/deeprank-ab-server/__init__.py \
     services/deeprank-ab-server/VERSION \
     services/deeprank-ab-server/app.py \
     services/deeprank-ab-server/adapter.py \
     services/deeprank-ab-server/models.py \
     services/deeprank-ab-server/settings.py \
     services/deeprank-ab-server/tools.py \
     services/deeprank-ab-server/uris.py \
     services/deeprank-ab-server/run_inference.py \
     /opt/deeprank-ab/server/

# Settings.
ENV DEEPRANK_AB_ROOT=/opt/deeprank-ab
ENV DEEPRANK_AB_JOBS_BASE_DIR=/data/deeprank_ab_jobs
ENV DEEPRANK_AB_PYTHON=/opt/conda/envs/deeprank-ab/bin/python
ENV DEEPRANK_AB_INFERENCE_SCRIPT=/opt/deeprank-ab/server/run_inference.py
ENV DEEPRANK_AB_NUM_WORKERS=4

# ESM-2 weight paths — read by run_inference.py's patched fetch_weights()
# and by load_model_and_alphabet_local() which derives the regression
# path from the model path.
ENV WEIGHT_PATH=/opt/deeprank-ab/esm_weights/esm2_t33_650M_UR50D.pt
ENV REG_WEIGHT_PATH=/opt/deeprank-ab/esm_weights/esm2_t33_650M_UR50D-contact-regression.pt

# PYTHONPATH so `from src.* import ...` resolves inside inference.py.
ENV PYTHONPATH=/opt/deeprank-ab/DeepRank-Ab

RUN mkdir -p /data/deeprank_ab_jobs

# Alibaba Cloud FC: port 9000, keep-alive ≥ 15 min.
ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/deeprank-ab/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
