# ---------- ODesign Server ----------
# Cross-modality biomolecular interaction design (protein/DNA/RNA/ligand/ion).
#
# Pre-build:
#   Place model weights in services/odesign-server/weights/ckpt/ (see scripts/fetch_weights.sh)
#   Place CCD data in services/odesign-server/weights/data/      (see scripts/fetch_ccd_data.sh)
#
# Build (from bioagent project root):
#   docker build --platform linux/amd64 -t odesign-server \
#       -f services/odesign-server/Dockerfile .
#
# Multistage layout
# -----------------
#   builder = conda env only: micromamba create + pip install + framework +
#             cleanup. No ODesign source or weights — those go directly into
#             the runtime stage so builder cache survives source/weight edits.
#   runtime = apt + conda env from builder + CUTLASS + ODesign source +
#             weights + CCD data + server code.
#
# Layer ordering (stable → volatile):
#   apt/micromamba → mirrors → conda env → pip extras → framework → cleanup
#   ‖ runtime: apt → conda env → CUTLASS → source → weights → data → server .py

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ARG MICROMAMBA_IMAGE=mambaorg/micromamba:2.6.2

# Pinned ODesign upstream. Bump SHA to upgrade; build verifies the checkout.
# HF model checkpoints (services/odesign-server/weights/ckpt/) and CCD data
# (services/odesign-server/weights/data/) are NOT in the upstream repo and
# are COPYed from local pre-staged dirs.  grnade.h5 IS in upstream git, so
# it comes via the clone — no separate pre-stage needed.
ARG ODESIGN_REPO=https://github.com/OTeam-AI4S/ODesign.git
ARG ODESIGN_SHA=cc95c2f8f915af4d87b1db3f44c4db2df8566a41


# ===== Stage 1: builder — conda env only =====
FROM ${MICROMAMBA_IMAGE} AS micromamba
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror via uv (project-wide convention).
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake g++-12 git curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- micromamba + uv ---
COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
ENV MAMBA_EXE=/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/odesign/bin:$PATH

# ---------------------------------------------------------------------------
# Mirror configuration for pip + conda.
# pip.conf: Huawei Cloud (kept for any pip invocations inside conda env).
# .condarc: TUNA (Tsinghua) for standard channels.
# uv: controlled by UV_INDEX_URL env above (post-env pip installs).
# ---------------------------------------------------------------------------
RUN mkdir -p /root/.config/pip \
    && printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
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
# Create conda env with Python 3.10 (ODesign only has requirements.txt, no
# environment.yml — so we create a bare env then pip install).
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n odesign -c conda-forge python=3.10

# ---------------------------------------------------------------------------
# ODesign source — clone at pinned SHA in the builder stage so the runtime
# stage can COPY --from=builder later.  The clone is also the source of the
# requirements.txt consumed below.
# ---------------------------------------------------------------------------
ARG ODESIGN_REPO
ARG ODESIGN_SHA
RUN set -eux; \
    git clone --filter=blob:none --no-checkout "$ODESIGN_REPO" /opt/odesign-src; \
    cd /opt/odesign-src; \
    git checkout "$ODESIGN_SHA"; \
    test "$(git rev-parse HEAD)" = "$ODESIGN_SHA"; \
    rm -rf .git

# ---------------------------------------------------------------------------
# ODesign dependencies (uv pip — ~10× faster than pip, uses Huawei mirror).
# PyG wheels need the find-links URL for the torch-2.3.1+cu121 index.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/odesign/bin/python \
        -r /opt/odesign-src/requirements.txt \
        --find-links https://data.pyg.org/whl/torch-2.3.1+cu121.html

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env (~1+ GB).
# Multistage: COPY --from=builder copies the final filesystem state, so
# cleanup in a separate RUN reclaims bytes just as effectively as fusing.
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/odesign/bin/pip uninstall -y \
        jupyterlab jupyter_server jupyterlab_server jupyter_client \
        ipywidgets ipykernel notebook nbconvert nbformat || true \
    && find /opt/conda/envs/odesign/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/odesign/share/doc \
        /opt/conda/envs/odesign/share/man \
        /opt/conda/envs/odesign/share/locale \
    && find /opt/conda/envs/odesign/lib/python3.10/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/odesign -depth -type d -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/odesign -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Shared service framework — independent layer so framework changes don't
# invalidate the (slow) conda env + cleanup above.
# ---------------------------------------------------------------------------
RUN --mount=type=bind,source=services/_framework,target=/tmp/service-framework \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/odesign/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2


# ===== Stage 2: runtime =====
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/odesign/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git libgl1-mesa-glx libglib2.0-0 fontconfig \
        librdmacm1 libibverbs1 ibverbs-providers \
    && rm -rf /var/lib/apt/lists/*

# --- conda env from builder ---
COPY --from=builder /opt/conda /opt/conda

# --- CUTLASS (for DS4Sci_EvoformerAttention + fast_layernorm) ---
# Cloned in runtime stage: header-only library, rarely changes, avoids
# bloating the builder conda env cache layer.
RUN git clone --depth 1 -b v3.5.1 https://github.com/NVIDIA/cutlass.git /kernels/cutlass

# --- ODesign source — from the builder stage's clone ---
COPY --from=builder /opt/odesign-src /opt/odesign/ODesign

# --- Model weights ---
# HF-downloaded model checkpoints (pre-staged via
# services/odesign-server/scripts/fetch_weights.sh).
COPY services/odesign-server/weights/ckpt/ /opt/odesign/ckpt/
# grnade.h5 is tracked in the upstream ODesign repo, so it ships with the
# clone above — copy it into the runtime ckpt dir alongside the HF weights.
RUN cp /opt/odesign/ODesign/ckpt/grnade.h5 /opt/odesign/ckpt/grnade.h5

# --- CCD data (pre-placed in services/odesign-server/weights/data/ before build,
# downloaded manually from Google Drive) ---
COPY services/odesign-server/weights/data/ /opt/odesign/data/

# Server code last — explicit file list so test/doc changes don't bust cache.
COPY services/odesign-server/__init__.py \
     services/odesign-server/__main__.py \
     services/odesign-server/VERSION \
     services/odesign-server/app.py \
     services/odesign-server/adapter.py \
     services/odesign-server/models.py \
     services/odesign-server/settings.py \
     services/odesign-server/tools.py \
     services/odesign-server/uris.py \
     /opt/odesign/server/

# --- Environment ---
ENV PYTHONPATH=/opt/odesign
ENV CUTLASS_PATH=/kernels/cutlass
ENV LAYERNORM_TYPE=fast_layernorm

ENV ODESIGN_ROOT=/opt/odesign/ODesign
ENV ODESIGN_PYTHON=/opt/conda/envs/odesign/bin/python
ENV ODESIGN_INFERENCE_SCRIPT=/opt/odesign/ODesign/scripts/inference.py
ENV ODESIGN_JOBS_BASE_DIR=/data/odesign_jobs
ENV ODESIGN_CKPT_ROOT_DIR=/opt/odesign/ckpt
ENV ODESIGN_DATA_ROOT_DIR=/opt/odesign/data
RUN mkdir -p /data/odesign_jobs

WORKDIR /opt/odesign


# Keepalive pings this URL through FC's gateway so the instance stays alive
# during long GPU jobs. Override at deploy time if the URL changes.
# ENV ODESIGN_KEEPALIVE_URL=
# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance. Requires FC console config too.
ENV ODESIGN_SESSION_HEADER_NAME=bioagent-session-id

ENV ODESIGN_MAX_CONCURRENT_JOBS=1
ENV PORT=9000
EXPOSE 9000

CMD ["/opt/conda/envs/odesign/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "9000", "--timeout-keep-alive", "900"]
