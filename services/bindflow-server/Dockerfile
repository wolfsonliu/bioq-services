# Build from bioagent project root:
#   docker build --platform linux/amd64 -t bindflow-server \
#       -f services/bindflow-server/Dockerfile .
#
# Pre-build (once; re-run to bump the pinned upstream SHA):
#   ./services/bindflow-server/scripts/vendor.sh   # → upstream/
#
# Upstream (ale94mleon/BindFlow) is GPL-3.0.  Downstream distribution
# must comply with GPL-3 (source availability etc.).
#
# HPC-primary service — Docker image is mainly consumed as an
# apptainer/singularity SIF and invoked via `python -m server {fep|mmpbsa}`.
# HTTP mode (uvicorn) is kept for local dev / smoke testing; not deployed to FC.
#
# Multistage layout
# -----------------
#   builder = conda env (Python 3.10 + snakemake + gromacs>=2023,<2026 (bioconda / conda-forge)
#             + ambertools + openmm + openff-toolkit + parmed + mpi4py
#             + pdbfixer + bindflow + gmx_MMPBSA pinned commit
#             + bioq-service-framework)
#   runtime = lean: conda env + BindFlow upstream + server code.
#
# Base: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 — reserved for future
# GPU mdrun; gromacs>=2023,<2026 (bioconda / conda-forge) in v0.0.1 is CPU-only.  Runtime image
# does NOT include the CUDA toolkit (headers/nvcc); only the runtime libs
# needed by future GPU builds.

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ARG MICROMAMBA_IMAGE=mambaorg/micromamba:2.6.2


# ===========================================================================
# Stage 1: builder — conda env + BindFlow source + framework.
# ===========================================================================
FROM ${MICROMAMBA_IMAGE} AS micromamba
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# NB: no `git` in apt — vendor.sh ran on host; image never clones.
# gmx_MMPBSA is installed from a git URL via pip below, which does need git
# at builder time only.  Runtime stage does not include git.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates wget git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=micromamba /bin/micromamba /bin/micromamba
ENV MAMBA_EXE=/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/bindflow/bin:$PATH

# ---------------------------------------------------------------------------
# Mirror configuration for CN-friendly builds.
# ---------------------------------------------------------------------------
RUN mkdir -p /root/.config/pip \
    && printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
       > /root/.config/pip/pip.conf \
    && cat > /root/.condarc <<'EOF'
channels:
  - conda-forge
  - bioconda
show_channel_urls: true
channel_priority: flexible
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# ---------------------------------------------------------------------------
# Create conda env — SPLIT into 4 install layers.
#
# Rationale: a single `micromamba env create` on the full 20+ package pinned
# stack peaks at ~4-6 GB solver memory (ambertools + openmm + openff-toolkit +
# mpi4py + bioconda gromacs cross-channel resolution).  On memory-constrained
# hosts (< 16 GB total) the OOM killer takes out the build (exit 137).
#
# Splitting into layers lets each solve work on a smaller graph:
#   Layer 1: Python + core numerics + PATH essentials
#   Layer 2: bioconda gromacs + snakemake (small; separate channel resolution)
#   Layer 3: ambertools alone (biggest single package)
#   Layer 4: chemistry stack (openmm, openff, pdbfixer, parmed, mpi4py)
#
# Each layer also gives Docker its own layer cache so incremental
# `environment.yml` bumps only rebuild from the changed layer down.
#
# Pins mostly follow docs/source/env/latest/environment-linux.yml from
# upstream (BindFlow), MINUS espaloma (design doc §11.3).
# ---------------------------------------------------------------------------

# Layer 1: base python + numerics.
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n bindflow \
        python=3.10.14 pip \
        numpy=1.22.4 pandas=2.0.3 pyyaml=6.0

# Layer 2: gromacs + snakemake.
#
# GROMACS pin: `gromacs>=2023,<2026` — a range rather than exact patch pin
# because bioconda / conda-forge don't ship every point release; only a
# subset (e.g. 2024.1, 2024.4) is packaged.  BindFlow's supported range is
# [2022, 2026); we start at 2023 for pymbar / mdanalysis compatibility.
# Solver picks the newest available across both channels (see /root/.condarc).
# Snakemake pinned to 7.32.2 per BindFlow's tested env.
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba install -y -n bindflow \
        'bioconda::snakemake=7.32.2' \
        'gromacs>=2023,<2026'

# Layer 3: ambertools (largest single package, ~2 GB installed).
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba install -y -n bindflow ambertools=22.5

# Layer 4: chemistry / MD stack.
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba install -y -n bindflow \
        openmm=8.0.0 openmmforcefields=0.11.2 \
        pdbfixer=1.9 openff-toolkit=0.14.2 \
        parmed=4.1.0 mpi4py=3.1.5

# Layer 5: pip-only chemistry deps (small; batched into one solve).
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/conda/envs/bindflow/bin/pip install \
        alchemlyb==2.0.0 mdanalysis==2.5.0 mdrestraintsgenerator==0.2.1 \
        pymbar==4.0.1 pytraj==2.0.6 rdkit==2023.3.2 toff==0.2.0

# `environment.yml` retained as authoritative source of pins; not consumed
# by the build directly (see layer split above).
COPY services/bindflow-server/environment.yml /opt/bindflow/environment.yml

# ---------------------------------------------------------------------------
# BindFlow itself — installed editable from vendored upstream/.
# --no-deps because the conda env already provides the deps in tested pins.
# ---------------------------------------------------------------------------
COPY services/bindflow-server/upstream /opt/bindflow/upstream
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/conda/envs/bindflow/bin/pip install --no-deps -e /opt/bindflow/upstream

# ---------------------------------------------------------------------------
# gmx_MMPBSA — required for /api/calculate/mmpbsa.  Upstream's install
# guide pins this specific commit for tested Python 3.10 compatibility.
# ---------------------------------------------------------------------------
ARG GMX_MMPBSA_COMMIT=27929e02067bc2321286809818d778a77a872010
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/conda/envs/bindflow/bin/pip install --no-deps -U \
    "git+https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA.git@${GMX_MMPBSA_COMMIT}"

# ---------------------------------------------------------------------------
# Framework — into the same conda env (Python 3.10 satisfies requires-python).
# ---------------------------------------------------------------------------
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/conda/envs/bindflow/bin/pip install \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env.
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/bindflow/bin/pip uninstall -y \
        jupyterlab jupyter_server jupyterlab_server jupyter_client \
        ipywidgets ipykernel notebook nbconvert nbformat || true \
    && find /opt/conda/envs/bindflow/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/bindflow/share/doc \
        /opt/conda/envs/bindflow/share/man \
        /opt/conda/envs/bindflow/share/locale \
    && find /opt/conda/envs/bindflow/lib/python3.10/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/bindflow -depth -type d \
        -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/bindflow -name '*.pyc' -delete


# ===========================================================================
# Stage 2: runtime — conda env + BindFlow source + server code.
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/bindflow/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 libxml2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /opt/bindflow /opt/bindflow

# Server code as `server` package — uvicorn imports `server.app:app`,
# `python -m server` triggers server/__main__.py.
COPY services/bindflow-server/__init__.py \
     services/bindflow-server/__main__.py \
     services/bindflow-server/VERSION \
     services/bindflow-server/app.py \
     services/bindflow-server/adapter.py \
     services/bindflow-server/inference.py \
     services/bindflow-server/models.py \
     services/bindflow-server/settings.py \
     services/bindflow-server/tools.py \
     services/bindflow-server/uris.py \
     /opt/bindflow/server/

WORKDIR /opt/bindflow

# PYTHONPATH lets `server.app` / `server.__main__` be importable regardless
# of the container runner's WORKDIR (Apptainer does not honor WORKDIR — see
# engineering/guides/apptainer-compatibility.md).
ENV PYTHONPATH=/opt/bindflow
ENV BINDFLOW_ROOT=/opt/bindflow
ENV BINDFLOW_PYTHON=/opt/conda/envs/bindflow/bin/python
ENV BINDFLOW_INFERENCE_SCRIPT=/opt/bindflow/server/inference.py
ENV BINDFLOW_JOBS_BASE_DIR=/data/bindflow_jobs

# Weights placeholder — BindFlow has no NN weights; NAS mount optional.
ENV BINDFLOW_WEIGHTS_DIR=/data/models/bindflow
ENV BINDFLOW_MAX_CONCURRENT_JOBS=1
# HPC-primary — task endpoints disabled by default (see design doc §3.3).
ENV BINDFLOW_TASK_ENDPOINTS_ENABLED=false

RUN mkdir -p /data/bindflow_jobs

ENV PORT=9000
EXPOSE 9000
# HTTP mode is the container default.  For CLI batch mode (HPC-primary):
#   apptainer exec image.sif python -m server {fep|mmpbsa} ...
CMD ["/opt/conda/envs/bindflow/bin/python", "-m", "uvicorn", \
     "server.app:app", "--host", "0.0.0.0", "--port", "9000", \
     "--timeout-keep-alive", "900"]
