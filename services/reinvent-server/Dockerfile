# reinvent-server — REINVENT4 de novo generation / RL / TL / scoring.
#
# Build from bioagent project root:
#   docker build --platform linux/amd64 -t reinvent-server -f services/reinvent-server/Dockerfile .
#
# Prerequisites (one-time):
#   ./services/reinvent-server/scripts/vendor.sh         # vendor upstream/REINVENT4
#   ./services/reinvent-server/scripts/fetch_weights.sh  # priors -> NAS (not baked)
#
# REINVENT4 is a pure pip package (torch 2.12 cu126) — uv venv skeleton, not conda.
# Priors load from NAS at /data/models/reinvent (REINVENT_PRIOR_BASE), never baked.
#
# Base image: 12.4.1 is fine even though torch is a cu126 build — pip torch wheels
# bundle their own CUDA + cuDNN runtime (nvidia-*-cu12 deps) and never use the base
# image's CUDA libs; CUDA minor-version compat means cu126 code runs on any 12.x
# driver (>= r525). reinvent has no non-torch CUDA code, so the base CUDA version
# is irrelevant. -runtime (not -devel) suffices: torch/chemprop are prebuilt wheels,
# no nvcc needed. Override with --build-arg BASE_IMAGE=... if desired.
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ---- Stage 1: builder ----
FROM ${BASE_IMAGE} AS builder
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600
# NB: do NOT use ubuntu22.04's apt `python3.11` — Jammy ships it as 3.11.0rc1,
# which predates `sys.get_int_max_str_digits` and makes `import torch` (2.12)
# crash with AttributeError. Instead uv provisions a proper standalone CPython
# into a fixed dir we copy into the runtime stage (the venv symlinks into it).
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PYTHON_PREFERENCE=only-managed
# git required: iSIM is a git+ VCS specifier (uv 0.11+ shells out to git CLI).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
WORKDIR /opt/reinvent-server
RUN uv python install 3.11 && uv venv --python 3.11 .venv
ENV PATH=/opt/reinvent-server/.venv/bin:$PATH

# Heavy torch first (cu126 index) — cache-friendly layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        --index-strategy unsafe-best-match \
        torch==2.12.0 torchvision

# Upstream REINVENT4 (vendored) + chemprop2 + iSIM.
COPY services/reinvent-server/upstream/REINVENT4 /opt/reinvent-server/upstream/REINVENT4
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        --index-strategy unsafe-best-match \
        "/opt/reinvent-server/upstream/REINVENT4[chemprop2]" \
        "iSIM@git+https://github.com/mqcomplab/iSIM.git"

# Shared framework + toml writer (separate layer). COPY (not bind) so framework
# source changes bust this layer's cache — a bind mount's content is not part of
# the BuildKit layer cache key and would ship a stale framework (missing
# output-sink fixes).
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2 tomli-w

# ---- Stage 2: runtime ----
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# rdkit pip wheel's rdMolDraw2D (imported by REINVENT at startup via
# rdkit.Chem.Draw) links X11 + fontconfig libs the slim CUDA base lacks:
# libxrender1/libxext6/libsm6 (X11) + libexpat1/libfontconfig1 (font stack).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
        libxrender1 libxext6 libsm6 libexpat1 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*
# The venv's python symlinks into the uv-managed standalone CPython, so copy it
# too (no system python in the runtime stage). Path must match the builder's
# UV_PYTHON_INSTALL_DIR.
COPY --from=builder /opt/uv-python /opt/uv-python
COPY --from=builder /opt/reinvent-server/.venv /opt/reinvent-server/.venv
COPY --from=builder /opt/reinvent-server/upstream/REINVENT4 /opt/reinvent-server/upstream/REINVENT4
ENV PATH=/opt/reinvent-server/.venv/bin:$PATH

COPY services/reinvent-server/__init__.py \
     services/reinvent-server/__main__.py \
     services/reinvent-server/VERSION \
     services/reinvent-server/app.py \
     services/reinvent-server/adapter.py \
     services/reinvent-server/models.py \
     services/reinvent-server/settings.py \
     services/reinvent-server/tools.py \
     services/reinvent-server/config_builder.py \
     services/reinvent-server/reinvent_cli.py \
     /opt/reinvent-server/server/

WORKDIR /opt/reinvent-server
ENV PYTHONPATH=/opt/reinvent-server
ENV REINVENT_ROOT=/opt/reinvent-server
ENV REINVENT_UPSTREAM_DIR=/opt/reinvent-server/upstream/REINVENT4
ENV REINVENT_PYTHON=/opt/reinvent-server/.venv/bin/python
ENV REINVENT_REINVENT_BIN=/opt/reinvent-server/.venv/bin/reinvent
ENV REINVENT_PRIOR_BASE=/data/models/reinvent
ENV REINVENT_JOBS_BASE_DIR=/data/reinvent_jobs
RUN mkdir -p /data/reinvent_jobs

# Runtime smoke: run a trivial CPU scoring in the FINAL image (not the builder,
# which has extra libs) so runtime-only gaps — missing shared libs, broken
# torch/rdkit import — fail the BUILD with a real traceback instead of shipping
# an image that crashes rc=1 at runtime. QED needs no prior, so this runs with
# nothing mounted.
RUN set -e; \
    printf 'CCO\nc1ccccc1\n' > /tmp/smoke.smi; \
    printf 'run_type = "scoring"\ndevice = "cpu"\n[parameters]\nsmiles_file = "/tmp/smoke.smi"\noutput_csv = "/tmp/smoke_out.csv"\n[scoring]\ntype = "geometric_mean"\n[[scoring.component]]\n[scoring.component.QED]\n[[scoring.component.QED.endpoint]]\nname = "QED"\nweight = 1.0\n' > /tmp/smoke.toml; \
    /opt/reinvent-server/.venv/bin/reinvent -l /tmp/smoke.log /tmp/smoke.toml; \
    test -s /tmp/smoke_out.csv; \
    rm -f /tmp/smoke.smi /tmp/smoke.toml /tmp/smoke.log /tmp/smoke_out.csv; \
    echo "reinvent scoring smoke OK"

# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance.
ENV REINVENT_SESSION_HEADER_NAME=bioagent-session-id
ENV REINVENT_MAX_CONCURRENT_JOBS=1
# Output-sink: when gateway sends X-Bioagent-Oss-Prefix, framework mirrors the job dir
# to this mount. FC console must mount the data-plane OSS bucket to /mnt/oss (RW).
# Missing mount => no-op (NAS only).
ENV REINVENT_OSS_OUTPUT_MOUNT=/mnt/oss

ENV PORT=9000
EXPOSE 9000
CMD ["/opt/reinvent-server/.venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "9000", "--timeout-keep-alive", "900"]
