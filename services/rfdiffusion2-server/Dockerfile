# Build from bioagent project root:
#   docker build --platform linux/amd64 -t rfdiffusion2-server -f services/rfdiffusion2-server/Dockerfile .
#
# Prereqs (both gitignored, populate before build):
#   ./services/rfdiffusion2-server/scripts/vendor.sh         # syncs upstream/
#   ./services/rfdiffusion2-server/scripts/fetch_weights.sh  # syncs weights/
# See engineering/decisions/2026-05-19-rfdiffusion2-server-vendor.md.
#
# Multistage layout
# -----------------
#   builder = conda env only: micromamba env create + pip extras + framework +
#             cleanup. No vendored source or weights — those go directly into
#             the runtime stage so builder cache survives source edits.
#   runtime = build-essential (JIT compile) + conda env from builder + vendored
#             source + weights + server code.
#
# Layer ordering (stable → volatile):
#   apt/micromamba → mirrors → conda env → pip extras → framework → cleanup
#   ‖ runtime: apt → conda env → vendored source → weights → server .py
#
# Base: CUDA 12.4 -devel (REQUIRED, do not switch to -runtime). rf2aa imports
# ipd.dev.cuda.cudabuild which calls `torch.utils.cpp_extension.load(...)` —
# JIT-compiles a CUDA kernel on first model construction. That path needs
# nvcc + cuda_runtime_api.h, only present in -devel. We also keep
# build-essential in the runtime stage so the JIT compile succeeds.
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ARG MICROMAMBA_IMAGE=mambaorg/micromamba:2.6.2


# ===========================================================================
# Stage 1: builder — conda env only (no vendored source or weights).
# ===========================================================================
FROM ${MICROMAMBA_IMAGE} AS micromamba
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror via uv (project-wide convention).
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

# Pin torch's CUDA discovery to the system toolkit.
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
ENV MAMBA_EXE=/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/rfd2/bin:$PATH

# ---------------------------------------------------------------------------
# Mirror configuration for pip + conda.
#
# pip.conf: Huawei Cloud. Configure BEFORE micromamba env-create — env.yml's
# `pip:` section invokes pip during env create and would otherwise hit pypi.org.
# uv: controlled by UV_INDEX_URL env above (post-env pip installs).
#
# conda: PKU Open Source mirror for the standard channels, sourced from the
# shared deploy/conda/mirrors.condarc (single source of truth — see
# docs/specs/2026-08-18-conda-mirror-consolidation-design.md). Three upstream
# channels we leave alone (fall through to conda.anaconda.org):
#   - conda.rosettacommons.org    pyrosetta (proprietary, no Chinese mirror)
#   - nvidia/label/cuda-12.4.0    Chinese mirrors carry `cloud/nvidia` (main)
#                                 but NOT `label/*` — mapping it 404s the build.
#   - dglteam/label/th24_cu124    same limitation.
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
# Create the rfd2 conda env from the vendored env files.
# ---------------------------------------------------------------------------
RUN --mount=type=bind,source=services/rfdiffusion2-server/upstream/envs,target=/tmp/rfd2-env-ro \
    --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    cp -r /tmp/rfd2-env-ro /tmp/rfd2-env \
    && micromamba env create -y -n rfd2 -f /tmp/rfd2-env/cuda124_env.yml \
    && rm -rf /tmp/rfd2-env

# ---------------------------------------------------------------------------
# Pip extras from upstream's apptainer .spec.
#
# NOTE: `baker-laboratory/ipd` is NOT pip-installed here. Its github HEAD has
# moved past the RFdiffusion2 cut and now imports `evn` at module load, which
# isn't in the conda env. We instead vendor ipd/ from the pinned RFdiffusion2
# checkout into upstream/ipd/ (see scripts/vendor.sh) so the contemporary copy
# lives on PYTHONPATH alongside rf_diffusion/.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/rfd2/bin/python \
        git+https://github.com/RalphMao/PyTimer.git

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env (~1.2 GB).
# Multistage: COPY --from=builder copies the final filesystem state, so
# cleanup in a separate RUN reclaims bytes just as effectively as fusing.
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/rfd2/bin/pip uninstall -y \
        jupyterlab jupyter_server jupyterlab_server jupyter_client \
        ipywidgets ipykernel notebook nbconvert nbformat || true \
    && find /opt/conda/envs/rfd2/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/rfd2/share/doc \
        /opt/conda/envs/rfd2/share/man \
        /opt/conda/envs/rfd2/share/locale \
    && find /opt/conda/envs/rfd2/lib/python3.11/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/rfd2 -depth -type d -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/rfd2 -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Shared service framework — independent layer so framework changes don't
# invalidate the (slow) conda env + cleanup above.
#
# COPY (not bind-mount): BuildKit does not include bind-mount content in the
# layer cache key, so a bind would ship a stale framework even after the
# framework source is updated (would silently drop the output-sink fix).
# ---------------------------------------------------------------------------
COPY framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/rfd2/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2


# ===========================================================================
# Stage 2: runtime — conda env + vendored source + weights + server code.
#
# Layer order (stable → volatile):
#   1. apt             — almost never changes
#   2. conda env       — changes only when env.yml or framework changes
#   3. vendored source — changes when upstream updates or vendor.sh reruns
#   4. model weights   — changes when new checkpoints are fetched
#   5. server code     — changes on every iteration (explicit file list)
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/rfd2/bin:$PATH
# Checkpoint configs embed ${oc.env:USER} for wandb_dir; resolve needs this set.
ENV USER=root

# Runtime apt is minimal:
#   build-essential   — REQUIRED. ipd.dev.cuda.cudabuild does runtime JIT compile
#                       via torch.cpp_extension.load on first model construction.
#   ca-certificates   — httpx HTTPS for OSS / signed URL inputs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/conda /opt/conda

COPY services/rfdiffusion2-server/upstream /opt/rfdiffusion2-server/upstream
COPY services/rfdiffusion2-server/weights /opt/rfdiffusion2-server/upstream/rf_diffusion/model_weights

WORKDIR /opt/rfdiffusion2-server

# Server code last — explicit file list so test/script/upstream changes don't
# invalidate this layer.
COPY services/rfdiffusion2-server/__init__.py \
     services/rfdiffusion2-server/__main__.py \
     services/rfdiffusion2-server/VERSION \
     services/rfdiffusion2-server/app.py \
     services/rfdiffusion2-server/adapter.py \
     services/rfdiffusion2-server/models.py \
     services/rfdiffusion2-server/settings.py \
     services/rfdiffusion2-server/tools.py \
     /opt/rfdiffusion2-server/server/

# Settings (env-driven; see services/rfdiffusion2-server/settings.py).
ENV PYTHONPATH=/opt/rfdiffusion2-server
ENV RFDIFFUSION2_ROOT=/opt/rfdiffusion2-server
ENV RFDIFFUSION2_MODELS_DIR=/opt/rfdiffusion2-server/upstream/rf_diffusion/model_weights
ENV RFDIFFUSION2_INFERENCE_SCRIPT=/opt/rfdiffusion2-server/upstream/rf_diffusion/run_inference.py
ENV RFDIFFUSION2_PYTHON=/opt/conda/envs/rfd2/bin/python
ENV RFDIFFUSION2_PYTHONPATH=/opt/rfdiffusion2-server/upstream
ENV RFDIFFUSION2_JOBS_BASE_DIR=/data/rfdiffusion2_jobs
RUN mkdir -p /data/rfdiffusion2_jobs

# Output-sink: when the gateway sends X-Bioagent-Oss-Prefix, the framework
# mirrors the finished job dir to this mount. Requires an OSS mount at /mnt/oss
# in the FC console (storage config). Absent mount => no-op (NAS-only).
ENV RFDIFFUSION2_OSS_OUTPUT_MOUNT=/mnt/oss


# Keepalive pings this URL through FC's gateway so the instance stays alive
# during long GPU jobs. Override at deploy time if the URL changes.
# ENV RFDIFFUSION2_KEEPALIVE_URL=https://fc-rfdiffusion-xaxngakbwo.cn-hangzhou-vpc.fcapp.run
# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance. Requires FC console config too.
ENV RFDIFFUSION2_SESSION_HEADER_NAME=bioagent-session-id

# Alibaba Cloud FC: port 9000, keep-alive ≥ 15 min, /healthz ready < 120 s.
ENV RFDIFFUSION2_MAX_CONCURRENT_JOBS=1
ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/rfd2/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
