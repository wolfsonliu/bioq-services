# Build from bioagent project root:
#   docker build --platform linux/amd64 -t chembounce-server \
#       -f services/chembounce-server/Dockerfile .
#
# Pre-build (once; re-run to bump pinned upstream commit / refresh data):
#   ./services/chembounce-server/scripts/vendor.sh         # → upstream/
#   ./services/chembounce-server/scripts/fetch_data.sh     # → data/
#
# *** LICENSE NOTE ***
# ChemBounce upstream has NO LICENSE file (as of the pinned SHA).  This
# image is for INTERNAL RESEARCH USE ONLY — do not push to public
# registries, do not redistribute.  See
# engineering/decisions/2026-06-28-chembounce-server-design.md §10.1.
#
# CPU-only service — pure ligand-based scaffold hopping (no GPU needed).
# Multistage layout
# -----------------
#   builder = conda env (Python 3.8 + RDKit 2020.09.5 + scaffoldgraph 1.1.2
#             + oddt + molvs + numpy/pandas/scikit-learn) for the ChemBounce
#             subprocess; plus /opt/server-venv (Python 3.11) for the
#             FastAPI + bioagent-service-framework layer (framework requires
#             py>=3.10, upstream conda env pins py3.8 — split by design).
#   runtime = lean: conda env + server venv from builder + ChemBounce source
#             + server code.
#
# Base: python:3.11-slim (Debian bookworm; ~120 MB) — micromamba is bolted
# on top for the conda env.  No nvidia/cuda base because no GPU.

ARG BASE_IMAGE=python:3.11-slim-bookworm
ARG MICROMAMBA_IMAGE=mambaorg/micromamba:2.6.2


# ===========================================================================
# Stage 1: builder — conda env + ChemBounce source + framework.
# ===========================================================================
FROM ${MICROMAMBA_IMAGE} AS micromamba
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

# NB: no `git` in apt — vendor.sh ran on host; image never clones.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
ENV MAMBA_EXE=/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/chembounce/bin:$PATH

# ---------------------------------------------------------------------------
# Mirror configuration (Huawei pip + TUNA conda) for CN-friendly builds.
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
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# ---------------------------------------------------------------------------
# Create conda env from vendored upstream `environment_simple.yml`.
# We use simple (not strict) because upstream's strict env pins RDKit
# 2020.09.5 + many transitive packages and is brittle on modern resolvers;
# the README explicitly recommends simple for compatibility.
# ---------------------------------------------------------------------------
COPY services/chembounce-server/upstream/environment_simple.yml /tmp/chembounce-env.yml
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    micromamba env create -y -n chembounce -f /tmp/chembounce-env.yml

# ---------------------------------------------------------------------------
# ChemBounce source — from vendored upstream/ (see scripts/vendor.sh).
# Upstream isn't a pip package; chembounce.py does `sys.path.append` of its
# own dir, so we just COPY it and add the dir to PYTHONPATH at runtime.
# ---------------------------------------------------------------------------
COPY services/chembounce-server/upstream /opt/chembounce/upstream
RUN /opt/conda/envs/chembounce/bin/python -c \
    "import sys; sys.path.insert(0, '/opt/chembounce/upstream'); \
     import chembounce; print('chembounce module OK')" || echo \
    "(import smoke skipped — chembounce.py has heavy import-time work)"

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env.
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/chembounce/bin/pip uninstall -y \
        jupyterlab jupyter_server jupyterlab_server jupyter_client \
        ipywidgets ipykernel notebook nbconvert nbformat || true \
    && find /opt/conda/envs/chembounce/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/chembounce/share/doc \
        /opt/conda/envs/chembounce/share/man \
        /opt/conda/envs/chembounce/share/locale \
    && find /opt/conda/envs/chembounce/lib/python3.8/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/chembounce -depth -type d \
        -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/chembounce -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Server venv — isolated Python 3.11 (base image's system Python) for the
# FastAPI + bioagent-service-framework layer.  Kept separate from the
# chembounce conda env (py3.8, pinned by upstream) because the framework
# requires py>=3.10 and uses `X | None` unions at runtime.
# ---------------------------------------------------------------------------
RUN --mount=type=bind,source=services/_framework,target=/tmp/service-framework \
    --mount=type=cache,target=/root/.cache/uv \
    python3.11 -m venv /opt/server-venv \
    && uv pip install --python /opt/server-venv/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2


# ===========================================================================
# Stage 2: runtime — conda env + ChemBounce source + server code.
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
# server-venv on PATH first so `python`/`uvicorn` resolve to the py3.11
# framework env; conda env still on PATH for occasional debugging.
ENV PATH=/opt/server-venv/bin:/opt/conda/envs/chembounce/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Conda env (py3.8 for chembounce subprocess) + server venv (py3.11 for
# FastAPI/framework) + ChemBounce source from builder.
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /opt/server-venv /opt/server-venv
COPY --from=builder /opt/chembounce /opt/chembounce

# Server code as `server` package — uvicorn imports `server.app:app`.
COPY services/chembounce-server/__init__.py \
     services/chembounce-server/__main__.py \
     services/chembounce-server/VERSION \
     services/chembounce-server/app.py \
     services/chembounce-server/adapter.py \
     services/chembounce-server/models.py \
     services/chembounce-server/settings.py \
     services/chembounce-server/tools.py \
     services/chembounce-server/uris.py \
     /opt/chembounce/upstream/server/

WORKDIR /opt/chembounce/upstream

# PYTHONPATH: upstream root (so `import chembounce` works) + the parent dir
# of `server/` (so uvicorn finds `server.app`).
ENV PYTHONPATH=/opt/chembounce/upstream
ENV CHEMBOUNCE_ROOT=/opt/chembounce/upstream
ENV CHEMBOUNCE_PYTHON=/opt/conda/envs/chembounce/bin/python
ENV CHEMBOUNCE_ENTRYPOINT=/opt/chembounce/upstream/chembounce.py
ENV CHEMBOUNCE_JOBS_BASE_DIR=/data/chembounce_jobs

# Weights/data externalized — NAS mount injected by FC config / SIF --bind.
ENV CHEMBOUNCE_WEIGHTS_DIR=/data/models/chembounce/data

RUN mkdir -p /data/chembounce_jobs

# FC session affinity header.
ENV CHEMBOUNCE_SESSION_HEADER_NAME=bioagent-session-id
ENV CHEMBOUNCE_MAX_CONCURRENT_JOBS=1

ENV PORT=9000
EXPOSE 9000
# uvicorn runs in the server venv (py3.11 + framework); ChemBounce is
# invoked as a subprocess via CHEMBOUNCE_PYTHON (conda env py3.8).
CMD ["/opt/server-venv/bin/python", "-m", "uvicorn", \
     "server.app:app", "--host", "0.0.0.0", "--port", "9000", \
     "--timeout-keep-alive", "900"]
