# ---------- DrugHIVE Server ----------
# Hierarchical VAE for structure-based drug design (Weller & Rohs, JCIM 2024).
#
# LICENSE: Upstream DrugHIVE code is USC-RL v2.0 (non-commercial academic
# research only).  Commercial use requires separate authorization from USC
# Stevens Center for Innovation (info@stevens.usc.edu).  See LICENSE in the
# vendored upstream tree and engineering/decisions/2026-07-02-drughive-server-design.md.
#
# Build (from bioagent project root):
#   docker build --platform linux/amd64 -t drughive-server \
#       -f services/drughive-server/Dockerfile .
#
# Prerequisites (one-time, re-run to bump pinned upstream commit / refresh weights):
#   ./services/drughive-server/scripts/vendor.sh          # → upstream/
#   ./services/drughive-server/scripts/vendor_qvina.sh    # → qvina/qvina2.1
#   ./services/drughive-server/scripts/fetch_weights.sh   # → weights/  (Zenodo)
#
# DrugHIVE source lives at services/drughive-server/upstream/ (vendored at
# pinned SHA via scripts/vendor.sh).  Pretrained checkpoint
# `drughive_model_ch9.ckpt` lives on NAS at
# /data/models/drughive/checkpoints/ — NOT baked into the image
# (FC NAS mount; SIF / HPC `apptainer run --bind`).
# See engineering/decisions/2026-06-26-service-weights-externalization.md.
#
# Multistage layout
# -----------------
#   builder = conda env (Py 3.10 + PyTorch 1.12 + cu117 + rdkit + openbabel)
#             + vendored qvina2.1 static binary + framework + cleanup.
#   runtime = lean: conda env from builder + DrugHIVE source + server code.
#
# Base: nvidia/cuda 11.8 -runtime (forward-compatible with conda's
# pytorch-cuda=11.7; DrugHIVE pins pytorch 1.12.1 which has no >11.8 wheels).

ARG BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
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
ENV PATH=/opt/conda/envs/drughive/bin:$PATH

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
# Strict priority: resolver stops at first match, dramatically faster on
# big channels like conda-forge / bioconda.  Without this, the pytorch 1.12
# + cu117 + rdkit + openbabel + qvina resolve can take 1-2 hours.
channel_priority: strict
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
# custom_channels: only map channels that TUNA actually hosts.
# IMPORTANT: do NOT map `nvidia` here — TUNA returns 404 for that channel.
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# ---------------------------------------------------------------------------
# Conda env — based on upstream's requirements.txt (Python 3.9 + pytorch
# 1.12.1 + cu117 + rdkit + openbabel + pytorch-lightning 2.0) with:
#   - Python bumped 3.9 → 3.10 (better wheel coverage; 1.12 has cu117 wheels
#     for both).
#   - rdkit bumped 2021.09.5 → 2022.09.3 (py3.10-friendly; API compatible).
#   - Kept plotly (upstream `drughive/atomgridding.py` imports it at module
#     scope — not just for viz).  Verified 2026-07-05 by FC runtime crash.
#   - Added matplotlib (imported unconditionally by drughive/gridutils,
#     lightning, trainutils, molfitting, visutils — not in upstream
#     requirements.txt but they were relying on a transitive pull).
#
# QVina2 is NOT installed via conda — bioconda's `qvina` recipe has no
# actual package files (verified 2026-07-02).  We drop the pre-built,
# statically-linked qvina2.1 binary from the QVina github repo (Apache-2.0)
# below via the COPY step; see scripts/vendor_qvina.sh.
#
# iJIT_NotifyEvent fix: pytorch 1.12 / 1.13's libtorch_cpu.so weak-references
# `iJIT_NotifyEvent` (Intel VTune ITT).  Without a provider in the env's
# lib/, `import torch` blows up with `undefined symbol: iJIT_NotifyEvent`.
# Historically `mkl` bundled the symbol, but mkl 2024.1+ / intel-openmp
# 2024.1+ dropped it — hence the failure on fresh builds.
# Pin `mkl<2024.1` / `intel-openmp<2024.1` (which still ship the ITT
# collector) + add `ittapi` as a second source.  Same fix as
# diffusion-hopping-server; see engineering/guides/pytorch-conda-ittapi.md
# and pytorch/pytorch#123097.
#
# setuptools<80: pytorch-lightning 2.0's bundled `lightning_fabric` calls
# `pkg_resources.declare_namespace(__name__)` at import time; setuptools
# 80+ removed the `pkg_resources` module (April 2025).  Without the pin,
# `python generate_molecules.py` dies with
# `ModuleNotFoundError: No module named 'pkg_resources'`.  Verified on FC
# 2026-07-05 with a fresh solve that pulled setuptools 81.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n drughive \
        -c pytorch -c nvidia -c conda-forge -c bioconda \
        python=3.10 \
        pytorch::pytorch=1.12.1 \
        pytorch-cuda=11.7 \
        pytorch-lightning=2.0.0 \
        rdkit=2022.09.3 \
        openbabel=3.1.1 \
        h5py=3.8.0 \
        numpy=1.24.1 \
        pandas=1.5.2 \
        scipy=1.10.0 \
        scikit-image=0.20.0 \
        scikit-learn=1.3.0 \
        tqdm=4.65.0 \
        pyyaml=6.0 \
        plotly=5.9.0 \
        matplotlib \
        conda-forge::ittapi \
        "mkl<2024.1" \
        "intel-openmp<2024.1" \
        "setuptools<80"

# ---------------------------------------------------------------------------
# QVina2 static binary — vendored via scripts/vendor_qvina.sh (Apache-2.0,
# from https://github.com/QVina/qvina).  Placed inside the conda env's bin
# so it lands on PATH alongside python / rdkit tools.  Both `qvina2.1` (the
# upstream DrugHIVE default docking_cmd) and `qvina2` (short-name symlink)
# are exposed for flexibility.
# ---------------------------------------------------------------------------
COPY services/drughive-server/qvina/qvina2.1 \
     /opt/conda/envs/drughive/bin/qvina2.1
RUN chmod +x /opt/conda/envs/drughive/bin/qvina2.1 && \
    ln -sf qvina2.1 /opt/conda/envs/drughive/bin/qvina2

# ---------------------------------------------------------------------------
# DrugHIVE source — from vendored upstream/ (see scripts/vendor.sh).
# Editable install via `setup.py` (upstream ships one) so
# `from drughive import ...` resolves without PYTHONPATH gymnastics.
# ---------------------------------------------------------------------------
COPY services/drughive-server/upstream /opt/drughive
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/drughive/bin/python \
        -e /opt/drughive

# ---------------------------------------------------------------------------
# SA_Score is an RDKit **Contrib** module — not on the Python path by
# default.  Upstream `drughive/molecules.py` does `from SA_Score import
# sascorer` expecting the caller to have added rdkit's Contrib/SA_Score
# to sys.path (or, more commonly, dropped a copy under cwd).
# Conda's rdkit ships Contrib at ``share/RDKit/Contrib/`` — symlink
# SA_Score onto /opt/drughive so the `from SA_Score import` works from
# any cwd.  Verified 2026-07-05.
# ---------------------------------------------------------------------------
RUN CONTRIB_DIR="/opt/conda/envs/drughive/share/RDKit/Contrib" && \
    if [ -d "$CONTRIB_DIR/SA_Score" ]; then \
        ln -sf "$CONTRIB_DIR/SA_Score" /opt/drughive/SA_Score; \
        echo "SA_Score linked: /opt/drughive/SA_Score -> $CONTRIB_DIR/SA_Score"; \
    else \
        echo "ERROR: SA_Score missing from rdkit Contrib at $CONTRIB_DIR" >&2; \
        ls "$CONTRIB_DIR" >&2 || true; \
        exit 1; \
    fi

# Build-time smoke: verify critical inference imports + qvina2.1 binary.
# Note: `import drughive` alone was too weak (empty __init__.py) — must
# import a real submodule to exercise the transitive chain (plotly /
# matplotlib / SA_Score / pytorch-lightning / etc).
RUN cd /opt/drughive && /opt/conda/envs/drughive/bin/python -c \
    "import torch; print('torch', torch.__version__); \
     import openbabel; print('openbabel ok'); \
     from rdkit import Chem; print('rdkit', Chem.__name__); \
     import plotly, matplotlib; print('plotly/matplotlib ok'); \
     from SA_Score import sascorer; print('sascorer ok'); \
     from drughive.generating import MolGenerator, MolGeneratorSpatial; \
     print('drughive.generating ok'); \
     import shutil; \
     assert shutil.which('qvina2.1'), 'qvina2.1 binary missing!'; \
     print('qvina2.1 at', shutil.which('qvina2.1'))"

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env.
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/drughive/bin/pip uninstall -y \
        jupyterlab jupyter_server jupyterlab_server jupyter_client \
        ipywidgets ipykernel notebook nbconvert nbformat || true \
    && find /opt/conda/envs/drughive/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/drughive/share/doc \
        /opt/conda/envs/drughive/share/man \
        /opt/conda/envs/drughive/share/locale \
    && find /opt/conda/envs/drughive/lib/python3.10/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/drughive -depth -type d \
        -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/drughive -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Shared service framework — independent layer so framework bumps don't
# invalidate the conda env layer above.
#
# COPY (not bind-mount): BuildKit does not include bind-mount content in the
# layer cache key, so a bind would ship a stale framework even after the
# framework source is updated (would silently drop the output-sink fix).
# ---------------------------------------------------------------------------
COPY framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/drughive/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2


# ===========================================================================
# Stage 2: runtime — conda env + DrugHIVE source + server code.
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/drughive/bin:$PATH

# libgomp1 for openbabel; ca-certificates for OSS / HTTP URI fetches.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Conda env from builder.
COPY --from=builder /opt/conda /opt/conda

# DrugHIVE source from builder (vendored upstream; editable install refs it).
COPY --from=builder /opt/drughive /opt/drughive

# Server code as `server` package — uvicorn imports `server.app:app`.
COPY services/drughive-server/__init__.py \
     services/drughive-server/__main__.py \
     services/drughive-server/VERSION \
     services/drughive-server/app.py \
     services/drughive-server/adapter.py \
     services/drughive-server/configs.py \
     services/drughive-server/models.py \
     services/drughive-server/settings.py \
     services/drughive-server/tools.py \
     /opt/drughive/server/

WORKDIR /opt/drughive

# PYTHONPATH — root (for `server` package). DrugHIVE is editable-installed
# into the conda env, so `import drughive` resolves without PYTHONPATH.
# Apptainer/SIF ignores WORKDIR, so PYTHONPATH must contain /opt/drughive
# explicitly for the `server` package.
ENV PYTHONPATH=/opt/drughive
ENV DRUGHIVE_ROOT=/opt/drughive
ENV DRUGHIVE_PYTHON=/opt/conda/envs/drughive/bin/python
ENV DRUGHIVE_GENERATE_SCRIPT=/opt/drughive/generate_molecules.py
ENV DRUGHIVE_OPTIMIZE_SCRIPT=/opt/drughive/generate_optimize.py
ENV DRUGHIVE_JOBS_BASE_DIR=/data/drughive_jobs

# Output-sink: when the gateway sends X-Bioagent-Oss-Prefix, the framework
# mirrors the finished job dir to this mount. Requires an OSS mount at /mnt/oss
# in the FC console (storage config). Absent mount => no-op (NAS-only).
ENV DRUGHIVE_OSS_OUTPUT_MOUNT=/mnt/oss

# Weights externalized — NAS mount injected by FC config / SIF --bind.
ENV DRUGHIVE_WEIGHTS_DIR=/data/models/drughive/checkpoints

RUN mkdir -p /data/drughive_jobs

# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance.
ENV DRUGHIVE_SESSION_HEADER_NAME=bioagent-session-id
ENV DRUGHIVE_MAX_CONCURRENT_JOBS=1

ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/drughive/bin/python", "-m", "uvicorn", \
     "server.app:app", "--host", "0.0.0.0", "--port", "9000", \
     "--timeout-keep-alive", "900"]
