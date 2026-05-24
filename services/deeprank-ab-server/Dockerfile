# Build from bioagent project root:
#   docker build --platform linux/amd64 -t deeprank-ab-server -f services/deeprank-ab-server/Dockerfile .
#
# Prereqs (both gitignored, populate before build):
#   ./services/deeprank-ab-server/scripts/fetch_esm_weights.sh   # downloads ESM-2 weights
# See engineering/decisions/2026-05-24-deeprank-ab-server-design.md.
#
# Multistage layout
# -----------------
#   builder = heavy: bzip2/curl/wget/git + micromamba + conda env create
#             (~8 GB env including pytorch 2.0.1+cu118, pyg, esm, etc.).
#   runtime = lean: only ca-certificates. The finished conda env + source +
#             weights + server code are copied over.
#
# Base: CUDA 11.8 -devel (matching pytorch-cuda=11.8 in the conda env).
# Using devel for both stages for safety — runtime would suffice since
# DeepRank-Ab has no JIT CUDA compilation, but devel avoids edge cases
# with torch extensions.
ARG BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04


# ===========================================================================
# Stage 1: builder — install the conda env + upstream source.
# ===========================================================================
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Build-only apt deps + micromamba bootstrap.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 curl git wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
       | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && /usr/local/bin/micromamba --version

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
  pyg: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  nvidia: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# ---------------------------------------------------------------------------
# Create the conda env from upstream's environment file.
# ---------------------------------------------------------------------------
COPY opensource/DeepRank-Ab/environment-gpu.yml /tmp/deeprank-ab-env/environment-gpu.yml

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

# ---------------------------------------------------------------------------
# Copy upstream source + ESM weights + framework.
# ---------------------------------------------------------------------------
COPY opensource/DeepRank-Ab /opt/deeprank-ab/DeepRank-Ab
COPY services/deeprank-ab-server/weights/esm /opt/deeprank-ab/esm_weights

# ---------------------------------------------------------------------------
# Patches applied to upstream inference.py:
#
# 1. fetch_weights() returns a bare filename ("esm2_t33_650M_UR50D.pt")
#    which fails when CWD != the directory containing the weights. Patch
#    to return the absolute path from WEIGHT_PATH env var.
#
# 2. NUM_WORKERS = 96 is excessive for FC GPU instances (typically 8 vCPU).
#    DataLoader with 96 workers on a constrained env can exhaust processes.
# ---------------------------------------------------------------------------
RUN sed -i \
        's|return f"{ESM_MODEL}.pt"|return os.getenv("WEIGHT_PATH") or f"{ESM_MODEL}.pt"|' \
        /opt/deeprank-ab/DeepRank-Ab/scripts/inference.py \
    && sed -i \
        's|NUM_WORKERS = 96|NUM_WORKERS = 8|' \
        /opt/deeprank-ab/DeepRank-Ab/scripts/inference.py

# ---------------------------------------------------------------------------
# Install the shared service framework.
# ---------------------------------------------------------------------------
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/conda/envs/deeprank-ab/bin/pip install \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2 \
    && rm -rf /tmp/service-framework

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
    && find /opt/conda/envs/deeprank-ab/lib/python3.9/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/deeprank-ab -depth -type d -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/deeprank-ab -name '*.pyc' -delete


# ===========================================================================
# Stage 2: runtime — drop build tools, keep the env + source.
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

# Bring over the conda env + source as separate COPYs for layer caching.
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /opt/deeprank-ab /opt/deeprank-ab

WORKDIR /opt/deeprank-ab

# Server code last so iterating on the FastAPI layer only rebuilds the tail.
COPY services/deeprank-ab-server/ /opt/deeprank-ab/server/

# Settings.
ENV DEEPRANK_AB_ROOT=/opt/deeprank-ab
ENV DEEPRANK_AB_JOBS_BASE_DIR=/data/deeprank_ab_jobs
ENV DEEPRANK_AB_PYTHON=/opt/conda/envs/deeprank-ab/bin/python
ENV DEEPRANK_AB_INFERENCE_SCRIPT=/opt/deeprank-ab/DeepRank-Ab/scripts/inference.py

# ESM-2 weight paths — read by the patched fetch_weights() and by
# load_model_and_alphabet_local() which derives the regression path
# from the model path.
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
