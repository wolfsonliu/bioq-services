# ---------- ImmuneBuilder Server ----------
# Antibody / Nanobody / TCR structure prediction from amino acid sequences.
#
# Pre-build:
#   Place model weights in services/immunebuilder-server/trained_model/
#   (see scripts/fetch_weights.sh)
#
# Build (from bioagent project root):
#   docker build --platform linux/amd64 -t immunebuilder-server \
#       -f services/immunebuilder-server/Dockerfile .
#
# Multistage layout
# -----------------
#   builder = conda env only: micromamba create + conda install OpenMM/pdbfixer/
#             HMMER + pip install torch + ImmuneBuilder (editable) + ANARCI +
#             framework + cleanup. Source/weights go directly into runtime so
#             builder cache survives source/weight edits.
#   runtime = apt + conda env from builder + ImmuneBuilder source (with weights)
#             + server code.
#
# Layer ordering (stable → volatile):
#   apt/micromamba → mirrors → conda env → pip extras → framework → cleanup
#   ‖ runtime: apt → conda env → source+weights → server .py

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ARG MICROMAMBA_IMAGE=mambaorg/micromamba:2.6.2

# Pre-build (once; re-run to bump pinned upstream commit):
#   ./services/immunebuilder-server/scripts/vendor.sh         # vendor upstream/
#
# ImmuneBuilder source lives at services/immunebuilder-server/upstream/
# (vendored at pinned SHA via scripts/vendor.sh).  Trained model weights live
# on NAS at /data/models/immunebuilder/trained_model/ — the image only
# contains a symlink at the upstream package's expected path; weights are
# resolved at run time via the FC NAS mount (SIF / HPC users bind via
# `apptainer --bind`).
#
# Pre-stage / upload weights once:
#   ./services/immunebuilder-server/scripts/fetch_weights.sh
#   rsync -a services/immunebuilder-server/trained_model/ \
#       <NAS>:/data/models/immunebuilder/trained_model/
# (Or set WEIGHTS_DST in fetch_weights.sh to download directly to NAS.)


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
        build-essential git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- micromamba + uv ---
COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
ENV MAMBA_EXE=/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/immunebuilder/bin:$PATH

# ---------------------------------------------------------------------------
# Mirror configuration for pip + conda.
# pip.conf: Huawei Cloud (kept for any pip invocations inside conda env).
# .condarc: PKU for standard channels + bioconda for HMMER.
# uv: controlled by UV_INDEX_URL env above (post-env pip installs).
# ---------------------------------------------------------------------------
COPY deploy/conda/mirrors.condarc /tmp/mirrors.condarc
RUN mkdir -p /root/.config/pip \
    && printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
       > /root/.config/pip/pip.conf \
    && cat > /root/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
EOF
RUN cat /tmp/mirrors.condarc >> /root/.condarc

# ---------------------------------------------------------------------------
# Create conda env with Python 3.10 + OpenMM + pdbfixer + HMMER.
# OpenMM and pdbfixer are C++ native libraries only distributed via conda.
# HMMER is required by ANARCI for sequence numbering.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n immunebuilder -c conda-forge -c bioconda \
        python=3.10 openmm pdbfixer hmmer

# ---------------------------------------------------------------------------
# Install PyTorch (CUDA 12.6 wheels, compatible with 12.4 base runtime).
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/immunebuilder/bin/python \
        torch==2.7.1

# ---------------------------------------------------------------------------
# ImmuneBuilder source — from vendored upstream/ (see scripts/vendor.sh).
# Trained model weights live on NAS at /data/models/immunebuilder/trained_model/
# (FC mount; SIF / HPC bind via apptainer); we symlink the upstream-expected
# package path to that mount so the ImmuneBuilder Python package finds them
# transparently — no upstream code change required.
# Pre-stage / upload weights:
#   ./services/immunebuilder-server/scripts/fetch_weights.sh   # → ./trained_model/
#   rsync -a services/immunebuilder-server/trained_model/ <NAS>:/data/models/immunebuilder/trained_model/
# See engineering/decisions/2026-06-26-service-weights-externalization.md.
# ---------------------------------------------------------------------------
COPY services/immunebuilder-server/upstream \
     /opt/immunebuilder/ImmuneBuilder
RUN rm -rf /opt/immunebuilder/ImmuneBuilder/ImmuneBuilder/trained_model && \
    ln -s /data/models/immunebuilder/trained_model \
          /opt/immunebuilder/ImmuneBuilder/ImmuneBuilder/trained_model

# ---------------------------------------------------------------------------
# ImmuneBuilder editable install + ANARCI (CDR numbering).
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/immunebuilder/bin/python \
        -e /opt/immunebuilder/ImmuneBuilder \
        anarci

# Build-time import smoke test.
RUN /opt/conda/envs/immunebuilder/bin/python -c \
    "import torch; print('torch', torch.__version__); import openmm; print('openmm ok')"

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env (~1+ GB).
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/immunebuilder/bin/pip uninstall -y \
        jupyterlab jupyter_server jupyterlab_server jupyter_client \
        ipywidgets ipykernel notebook nbconvert nbformat || true \
    && find /opt/conda/envs/immunebuilder/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/immunebuilder/share/doc \
        /opt/conda/envs/immunebuilder/share/man \
        /opt/conda/envs/immunebuilder/share/locale \
    && find /opt/conda/envs/immunebuilder -depth -type d -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/immunebuilder -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Shared service framework — independent layer so framework changes don't
# invalidate the (slow) conda env + cleanup above.
# ---------------------------------------------------------------------------
COPY framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/immunebuilder/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2


# ===== Stage 2: runtime =====
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/immunebuilder/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- conda env from builder ---
COPY --from=builder /opt/conda /opt/conda

# --- ImmuneBuilder source + weights (editable install metadata in conda env
#     points to /opt/immunebuilder/ImmuneBuilder via .egg-link / .pth) ---
COPY --from=builder /opt/immunebuilder /opt/immunebuilder

WORKDIR /opt/immunebuilder

# Server code last — explicit file list so test/script changes don't bust cache.
COPY services/immunebuilder-server/__init__.py \
     services/immunebuilder-server/__main__.py \
     services/immunebuilder-server/VERSION \
     services/immunebuilder-server/app.py \
     services/immunebuilder-server/adapter.py \
     services/immunebuilder-server/models.py \
     services/immunebuilder-server/settings.py \
     services/immunebuilder-server/tools.py \
     /opt/immunebuilder/server/

# --- Environment ---
ENV PYTHONPATH=/opt/immunebuilder
ENV IMMUNEBUILDER_JOBS_BASE_DIR=/data/immunebuilder_jobs
ENV IMMUNEBUILDER_VENV_BIN=/opt/conda/envs/immunebuilder/bin
ENV IMMUNEBUILDER_WEIGHTS_DIR=/data/models/immunebuilder/trained_model
RUN mkdir -p /data/immunebuilder_jobs

# Output-sink: 当 gateway 传 X-Bioagent-Oss-Prefix 时，framework 把 job dir 镜像到此挂载点。
# 需在 FC 控制台把数据面 OSS bucket 挂到 /mnt/oss。缺挂载 => no-op（仅 NAS）。
ENV IMMUNEBUILDER_OSS_OUTPUT_MOUNT=/mnt/oss


# Keepalive pings this URL through FC's gateway so the instance stays alive
# during long GPU jobs. Override at deploy time if the URL changes.
# ENV IMMUNEBUILDER_KEEPALIVE_URL=https://fc-immuebuilder-mhxbldkfhc.cn-hangzhou-vpc.fcapp.run
# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance. Requires FC console config too.
ENV IMMUNEBUILDER_SESSION_HEADER_NAME=bioagent-session-id
ENV IMMUNEBUILDER_MAX_CONCURRENT_JOBS=1
ENV PORT=9000
EXPOSE 9000

CMD ["/opt/conda/envs/immunebuilder/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "9000", "--timeout-keep-alive", "900"]
