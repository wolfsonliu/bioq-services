# ---------- OpenADMET Server ----------
# General-purpose ADMET ML toolbox HTTP wrapper — OpenADMET Models (MIT).
#
# Build (from bioagent project root):
#   docker build --platform linux/amd64 -t openadmet-server \
#       -f services/openadmet-server/Dockerfile .
#
# Prerequisites (one-time; re-run to bump pinned upstream commit / refresh weights):
#   ./services/openadmet-server/scripts/vendor.sh          # upstream + pip_deps (4 git)
#   ./services/openadmet-server/scripts/fetch_weights.sh   # 6 HF models + CheMeleon → NAS
#
# Weights are NOT baked into the image — they live on NAS at
# /data/models/openadmet/{models/,foundations/}.  FC auto-mounts;
# SIF / HPC use `apptainer run --bind /scratch/models/openadmet:/data/models/openadmet`.
# See engineering/decisions/2026-07-05-openadmet-server-design.md §6.7.
#
# HOME override: upstream chemprop.py caches CheMeleon foundation at
# `Path.home() / ".chemprop"`.  We set HOME=/data/models/openadmet/foundations
# so it hits the pre-staged NAS cache without triggering Zenodo download.
#
# Multistage layout
# -----------------
#   builder = conda env (py3.12 + pytorch-gpu + chemprop + molfeat + rdkit
#             + all conda deps from openadmet-models-gpu.yaml) + pip_deps +
#             openadmet source + framework + cleanup.
#   runtime = lean: conda env + upstream + server code.
#
# Base: nvidia/cuda 12.4 -runtime (matches conda-forge pytorch-gpu default;
# openadmet-models is py3.12 which requires the newer Ubuntu 22.04 base).

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ARG MICROMAMBA_IMAGE=mambaorg/micromamba:2.6.2


# ===========================================================================
# Stage 1: builder
# ===========================================================================
FROM ${MICROMAMBA_IMAGE} AS micromamba
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

# BuildKit builders don't expose the NVIDIA driver ABI, so conda's `__cuda`
# virtual package is missing at solve time.  Modern conda-forge pytorch-gpu
# metapackage requires `__cuda` and will refuse to install without an
# override.  This env var fakes the presence of CUDA 12.4 for the solver;
# actual GPU support comes from the base image (nvidia/cuda:12.4.1-runtime)
# plus the NVIDIA container runtime at deploy time.
# See: https://conda-forge.org/docs/user/tipsandtricks/#installing-cuda-enabled-packages
ENV CONDA_OVERRIDE_CUDA=12.4

# No `git` in apt — vendor.sh ran on host; image never clones.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
ENV MAMBA_EXE=/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/openadmet-models/bin:$PATH

# ---------------------------------------------------------------------------
# CN mirrors — Huawei for pip, PKU for conda-forge.  Note: upstream
# openadmet-models-gpu.yaml uses ONLY conda-forge (no pytorch/nvidia/defaults
# channels).  If we drop the yaml's channel constraint by adding `-c pytorch`
# etc., the solver — under `channel_priority: strict` — picks pytorch-gpu
# 1.9.1 from the `pytorch` channel (old py3.6 build), which conflicts with
# every modern package.  So: mirror upstream exactly — conda-forge only.
# ---------------------------------------------------------------------------
COPY deploy/conda/mirrors.condarc /tmp/mirrors.condarc
RUN mkdir -p /root/.config/pip \
    && printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
       > /root/.config/pip/pip.conf \
    && cat > /root/.condarc <<'EOF'
channels:
  - conda-forge
show_channel_urls: true
# `flexible` (not strict) — lets the solver upgrade transitive deps across
# builds within conda-forge itself (e.g. old libtiff pinning old pillow
# pinning old datamol).  strict would freeze at the first-found build and
# create false conflicts.
channel_priority: flexible
EOF
RUN cat /tmp/mirrors.condarc >> /root/.condarc

# ---------------------------------------------------------------------------
# Conda env — mirrors upstream devtools/conda-envs/openadmet-models-gpu.yaml
# (conda-forge only channel).  Key pins:
#   * python=3.12                — upstream requires >=3.12,<3.14
#   * pytorch-gpu>=2.4           — old (1.9) metapackage would trap solver
#   * pytorch-lightning<=2.6.1   — upstream pin
#   * chemprop>=2.1.1, zarr>=3.0 — upstream pins
# All pip-git deps in the upstream yaml (molfeat/TabPFN/etc.) are installed
# from local pip_deps/ (vendor.sh output), NOT github — see design §6.8.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    CONDA_OVERRIDE_CUDA=12.4 \
    micromamba create -y -n openadmet-models \
        -c conda-forge \
        python=3.12 \
        "pytorch-gpu>=2.4" \
        "pytorch-lightning<=2.6.1" \
        rdkit \
        ase \
        boto3 \
        catboost \
        "chemprop>=2.1.1" \
        datamol \
        h5py \
        hyperopt \
        intake \
        intake-parquet \
        joblib \
        lightgbm \
        loguru \
        matplotlib \
        mdanalysis \
        mordredcommunity \
        numpy \
        pandas \
        pyyaml \
        rich \
        scikit-learn \
        seaborn \
        splito \
        statsmodels \
        tabulate \
        umap-learn \
        uncertainty-toolbox \
        xgboost \
        "zarr>=3.0.0"

# ---------------------------------------------------------------------------
# Pip deps — reportlab (for `openadmet compare --report`), email-validator
# + phx-class-registry (referenced by upstream registries), plus the 4 vendored
# git pip deps.  All installed OFFLINE from services/openadmet-server/pip_deps/.
#
# `--no-build-isolation` for editable installs so they can see the conda-env's
# already-installed torch / rdkit at build time (upstream may reference them
# in setup.py).
# ---------------------------------------------------------------------------
COPY services/openadmet-server/pip_deps /tmp/pip_deps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/openadmet-models/bin/python \
        reportlab email-validator phx-class-registry \
        "setuptools<82.0.0" "wandb<0.26.0" tabpfn-extensions \
        /tmp/pip_deps/molfeat \
        /tmp/pip_deps/TabPFN \
        /tmp/pip_deps/neural-pairwise-regression \
        /tmp/pip_deps/useful_rdkit_utils \
    && rm -rf /tmp/pip_deps

# ---------------------------------------------------------------------------
# Sync boto3 <-> botocore.  The pip layer above may pull transitive deps
# that upgrade botocore beyond conda's boto3 compat window (or vice versa),
# breaking `import boto3` with:
#     ImportError: cannot import name 'DocumentModifiedShape' from botocore.docs.utils
# because boto3 1.35+ requires the newer symbol while an older botocore
# (or a botocore held back by aiobotocore/s3fs pins) doesn't have it.
# Reinstall as a matched pair — uv resolves them together.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/openadmet-models/bin/python \
        --upgrade boto3 botocore

# ---------------------------------------------------------------------------
# OpenADMET source — from vendored upstream/ (see scripts/vendor.sh).
# Editable install so `openadmet` CLI command and `from openadmet.models import ...`
# both work.  Upstream's pyproject.toml declares `openadmet = "openadmet.models.cli.cli:cli"`.
# ---------------------------------------------------------------------------
COPY services/openadmet-server/upstream /opt/openadmet/upstream
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/openadmet-models/bin/python \
        --no-deps -e /opt/openadmet/upstream

# Build-time smoke: verify critical imports + CLI entry-point script.
# The `openadmet` script is created by setuptools from the pyproject.toml
# [project.scripts] table (`openadmet = openadmet.models.cli.cli:cli`).
# We rely on it — NOT `python -m openadmet.models.cli.cli` — because the
# module lacks a `__main__` guard and would silently no-op.
RUN /opt/conda/envs/openadmet-models/bin/python -c \
    "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available()); \
     from rdkit import Chem; print('rdkit ok'); \
     import chemprop; print('chemprop', chemprop.__version__); \
     import molfeat; print('molfeat', getattr(molfeat, '__version__', '?')); \
     import openadmet.models; print('openadmet.models ok'); \
     from openadmet.models.cli.cli import cli; print('openadmet CLI ok')" \
    && test -x /opt/conda/envs/openadmet-models/bin/openadmet \
    && /opt/conda/envs/openadmet-models/bin/openadmet --help > /dev/null \
    && echo "openadmet entry-point OK"

# ---------------------------------------------------------------------------
# Cleanup dead weight — same recipe as drughive-server.
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/openadmet-models/bin/pip uninstall -y \
        jupyterlab jupyter_server jupyterlab_server jupyter_client \
        ipywidgets ipykernel notebook nbconvert nbformat || true \
    && find /opt/conda/envs/openadmet-models/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/openadmet-models/share/doc \
        /opt/conda/envs/openadmet-models/share/man \
        /opt/conda/envs/openadmet-models/share/locale \
    && find /opt/conda/envs/openadmet-models/lib/python3.12/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/openadmet-models -depth -type d \
        -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/openadmet-models -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# bioagent service framework — separate layer for cache locality.
# ---------------------------------------------------------------------------
COPY framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/openadmet-models/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2


# ===========================================================================
# Stage 2: runtime
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/openadmet-models/bin:$PATH

# ca-certificates for OSS / HTTPS URI fetches; libgomp1 for scikit-learn OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Conda env + upstream from builder.
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /opt/openadmet/upstream /opt/openadmet/upstream

# Server code as `server` package.
COPY services/openadmet-server/__init__.py \
     services/openadmet-server/__main__.py \
     services/openadmet-server/VERSION \
     services/openadmet-server/app.py \
     services/openadmet-server/adapter.py \
     services/openadmet-server/models.py \
     services/openadmet-server/settings.py \
     services/openadmet-server/tools.py \
     /opt/openadmet/server/

WORKDIR /opt/openadmet

# PYTHONPATH so `import server` works from any CWD (Apptainer ignores WORKDIR).
ENV PYTHONPATH=/opt/openadmet
ENV OPENADMET_ROOT=/opt/openadmet/upstream
ENV OPENADMET_PYTHON=/opt/conda/envs/openadmet-models/bin/python
ENV OPENADMET_JOBS_BASE_DIR=/data/openadmet_jobs
ENV OPENADMET_WEIGHTS_DIR=/data/models/openadmet
# Output-sink: when gateway sends X-Bioagent-Oss-Prefix, framework mirrors the
# job dir here. Requires the data-plane OSS bucket mounted at /mnt/oss RW in the
# FC console. Absent mount => no-op (NAS only). Default already /mnt/oss;
# explicit for self-documentation.
ENV OPENADMET_OSS_OUTPUT_MOUNT=/mnt/oss

# HOME override — upstream chemprop.py caches CheMeleon foundation under
# `Path.home() / ".chemprop"`.  Point HOME at the NAS-mounted foundations
# subdir so the cache hit skips Zenodo (see §6.7).  FC does not require an
# interactive shell so overriding HOME here is safe.
ENV HOME=/data/models/openadmet/foundations

# Disable wandb (only used by anvil training; predict/compare have no
# telemetry surface but chemprop / lightning may lazy-import it).
ENV WANDB_DISABLED=true

# Disable loguru ANSI color so subprocess log parsing stays clean.
ENV LOGURU_COLORIZE=false

RUN mkdir -p /data/openadmet_jobs

# FC session affinity + concurrency knobs.
ENV OPENADMET_SESSION_HEADER_NAME=bioagent-session-id
ENV OPENADMET_MAX_CONCURRENT_JOBS=1

ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/openadmet-models/bin/python", "-m", "uvicorn", \
     "server.app:app", "--host", "0.0.0.0", "--port", "9000", \
     "--timeout-keep-alive", "900"]
