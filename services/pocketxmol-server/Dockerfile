# ---------- PocketXMol Server ----------
# Pocket-interacting foundation model (Peng et al., *Cell* 2026).
#
# LICENSE: Upstream PocketXMol is MIT-licensed.  See LICENSE inside the
# vendored upstream tree and engineering/decisions/
# 2026-07-06-pocketxmol-server-design.md.
#
# Build (from bioagent project root):
#   docker build --platform linux/amd64 -t pocketxmol-server \
#       -f services/pocketxmol-server/Dockerfile .
#
# Prerequisites (one-time, re-run to bump pinned upstream commit / refresh weights):
#   ./services/pocketxmol-server/scripts/vendor.sh          # → upstream/
#   ./services/pocketxmol-server/scripts/fetch_weights.sh   # → weights/  (Zenodo, ~500 MB)
#
# PocketXMol source lives at services/pocketxmol-server/upstream/ (vendored
# at pinned SHA via scripts/vendor.sh).  Model weights (3 ckpts:
# pocketxmol.ckpt + tuned_ranker.ckpt + flex_cfd.ckpt) live on NAS at
# /data/models/pocketxmol/ — NOT baked into the image (FC NAS mount; SIF /
# HPC `apptainer run --bind`).  See engineering/decisions/
# 2026-06-26-service-weights-externalization.md.
#
# Multistage layout
# -----------------
#   builder = conda env (Py 3.10 + rdkit + openbabel + biopython + lmdb …)
#             + pip torch 2.7 (cu128) + PyG 2.7 wheels + framework + cleanup.
#   runtime = lean: conda env from builder + PocketXMol source + server code.
#
# Base: nvidia/cuda 12.4 -runtime — driver 12.4 is forward-compatible with
# pytorch cu128 binaries (CUDA minor-version compat).  Choosing 12.4
# over 12.8 base image because 12.4 has broader distro coverage today and
# still runs cu128 wheels.

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ARG MICROMAMBA_IMAGE=mambaorg/micromamba:2.6.2


# ===========================================================================
# Stage 1: builder — conda env + pip torch/PyG + framework.
# ===========================================================================
FROM ${MICROMAMBA_IMAGE} AS micromamba
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

# NB: no `git` in apt — vendor.sh ran on the host; the image never clones.
# wget is needed only by the (optional) in-image download flows, not by
# core build.  build-essential brings gcc for occasional pip-source deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
ENV MAMBA_EXE=/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/pocketxmol/bin:$PATH

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
channel_priority: strict
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# ---------------------------------------------------------------------------
# Conda env (chem/bio stack only) — from upstream environment_cu128_base.yml
# minus tensorboard (~800 MB TF stack we do not need at inference).  Torch +
# PyG go through pip (see next RUN) since the CUDA 12.8 wheels are on
# download.pytorch.org / data.pyg.org, not conda-forge.
#
# WHY these pins:
#   - python=3.10 : supported by upstream cu128 config; sample_use.py uses
#     3.10-friendly syntax.
#   - rdkit=2023.9.3 : upstream default; API stable across chem utilities.
#   - openbabel=3.1.1 : peptide .sdf → .pdb conversion in sample_use.py.
#   - python-lmdb>=1.4 : utils/dataset UseDataset touches lmdb.  Upstream's
#     cu128 yml pins 1.2.1, but that build only exists for py<=3.9 — pinning
#     it against python=3.10 makes mamba thrash on an UNSAT spec for hours
#     before erroring.  1.4+ has py310 builds; the lmdb read API used here
#     is unchanged since 1.2.  Range (not exact) guards against mirror gaps.
#   - biopython=1.83 : PDB parsing / peptide handling.
#   - networkx=2.8 : molecule graph utilities.
#   - setuptools<70 : upstream requires this pin (pkg_resources reliance).
#   - scipy 1.10.1 : conservative pin from upstream cu128 base.
#   - easydict=1.9 / pyyaml=6.0.2 : config parsing (sample_use.py).
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n pocketxmol \
        -c conda-forge -c bioconda \
        python=3.10 \
        pip=24.3.1 \
        easydict=1.9 \
        numpy=1.24 \
        openbabel=3.1.1 \
        pandas=1.5.2 \
        "python-lmdb>=1.4,<1.7" \
        pyyaml=6.0.2 \
        tqdm=4.64.0 \
        rdkit=2023.9.3 \
        biopython=1.83 \
        networkx=2.8 \
        "setuptools<70" \
        scikit-learn=1.1.0 \
        scipy=1.10.1

# ---------------------------------------------------------------------------
# pytorch 2.7 + cu128 + lightning + PyG 2.x + native ext wheels
# ---------------------------------------------------------------------------
# The upstream cu128 setup guide runs the following three pip commands.
# We reproduce them verbatim (via uv) into the pocketxmol env:
#   pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
#   pip install lightning torch_geometric
#   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
#     -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
#
# uv respects --index-url / --find-links flags identically to pip.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/pocketxmol/bin/python \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.0

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/pocketxmol/bin/python \
        lightning torch_geometric peptidebuilder==1.1.0

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/pocketxmol/bin/python \
        --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html \
        pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv

# ---------------------------------------------------------------------------
# PocketXMol source — from vendored upstream/ (scripts/vendor.sh).
# The upstream tree isn't a Python package (no setup.py / pyproject.toml);
# scripts/sample_use.py does `sys.path.append('.')` at import time and
# relies on WORKDIR = the tree root.  We honor that via WORKDIR + PYTHONPATH
# in the runtime stage.
# ---------------------------------------------------------------------------
COPY services/pocketxmol-server/upstream /opt/pocketxmol

# Build-time smoke: verify critical inference imports.  We *don't* import
# `from scripts.train_pl import DataModule` here because it exhaustively
# instantiates optimizers; a shallow import of the low-level bits is
# enough to catch the ITT / mkl / pytorch-lightning bootstrap issues at
# build time rather than at first request.
RUN cd /opt/pocketxmol && /opt/conda/envs/pocketxmol/bin/python -c "\
import torch; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
assert torch.version.cuda is not None, \
    'pytorch built without CUDA — pip wheel may have fallen back to CPU'; \
import torch_geometric; print('pyg', torch_geometric.__version__); \
import torch_scatter; print('torch_scatter ok'); \
import torch_sparse; print('torch_sparse ok'); \
import torch_cluster; print('torch_cluster ok'); \
import pytorch_lightning as pl; print('pytorch-lightning', pl.__version__); \
from rdkit import Chem; print('rdkit ok'); \
import openbabel; print('openbabel ok'); \
import easydict, yaml, lmdb, Bio; print('easydict/yaml/lmdb/Bio ok'); \
import PeptideBuilder; print('PeptideBuilder ok')"

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env.
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/pocketxmol/bin/pip uninstall -y \
        jupyterlab jupyter_server jupyterlab_server jupyter_client \
        ipywidgets ipykernel notebook nbconvert nbformat || true \
    && find /opt/conda/envs/pocketxmol/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/pocketxmol/share/doc \
        /opt/conda/envs/pocketxmol/share/man \
        /opt/conda/envs/pocketxmol/share/locale \
    && find /opt/conda/envs/pocketxmol/lib/python3.10/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/pocketxmol -depth -type d \
        -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/pocketxmol -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Shared service framework — independent layer so framework bumps don't
# invalidate the conda env layer above.
# ---------------------------------------------------------------------------
RUN --mount=type=bind,source=services/_framework,target=/tmp/service-framework \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/pocketxmol/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2


# ===========================================================================
# Stage 2: runtime — conda env + PocketXMol source + server code.
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/pocketxmol/bin:$PATH

# libgomp1 for openbabel; ca-certificates for OSS / HTTP URI fetches.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Conda env from builder.
COPY --from=builder /opt/conda /opt/conda

# PocketXMol source from builder (vendored upstream).
COPY --from=builder /opt/pocketxmol /opt/pocketxmol

# Server code as `server` package — uvicorn imports `server.app:app`.
COPY services/pocketxmol-server/__init__.py \
     services/pocketxmol-server/__main__.py \
     services/pocketxmol-server/VERSION \
     services/pocketxmol-server/app.py \
     services/pocketxmol-server/adapter.py \
     services/pocketxmol-server/configs.py \
     services/pocketxmol-server/models.py \
     services/pocketxmol-server/settings.py \
     services/pocketxmol-server/tools.py \
     services/pocketxmol-server/uris.py \
     /opt/pocketxmol/server/

WORKDIR /opt/pocketxmol

# PYTHONPATH — root (for `server` package + upstream `sys.path.append('.')`
# equivalent).  Apptainer/SIF ignores WORKDIR, so PYTHONPATH must contain
# /opt/pocketxmol explicitly.
ENV PYTHONPATH=/opt/pocketxmol
ENV POCKETXMOL_ROOT=/opt/pocketxmol
ENV POCKETXMOL_PYTHON=/opt/conda/envs/pocketxmol/bin/python
ENV POCKETXMOL_SAMPLE_SCRIPT=/opt/pocketxmol/scripts/sample_use.py
ENV POCKETXMOL_CONFIDENCE_SCRIPT=/opt/pocketxmol/scripts/believe_use_pdb.py
ENV POCKETXMOL_JOBS_BASE_DIR=/data/pocketxmol_jobs

# Weights externalized — NAS mount injected by FC config / SIF --bind.
ENV POCKETXMOL_WEIGHTS_DIR=/data/models/pocketxmol
ENV POCKETXMOL_PXM_CHECKPOINT=/data/models/pocketxmol/pxm/checkpoints/pocketxmol.ckpt
ENV POCKETXMOL_TUNED_CFD_CKPT=/data/models/pocketxmol/tuned_ranker/checkpoints/tuned_ranker.ckpt
ENV POCKETXMOL_FLEX_CFD_CKPT=/data/models/pocketxmol/flex_cfd/checkpoints/flex_cfd.ckpt
ENV POCKETXMOL_CCD_DIR=/data/models/pocketxmol/ccd

RUN mkdir -p /data/pocketxmol_jobs

# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance.
ENV POCKETXMOL_SESSION_HEADER_NAME=bioagent-session-id
ENV POCKETXMOL_MAX_CONCURRENT_JOBS=1

ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/pocketxmol/bin/python", "-m", "uvicorn", \
     "server.app:app", "--host", "0.0.0.0", "--port", "9000", \
     "--timeout-keep-alive", "900"]
