# Build from bioagent project root:
#   docker build --platform linux/amd64 -t iggm-server \
#       -f services/iggm-server/Dockerfile .
#
# Pre-build (once; re-run to bump pinned upstream commit / refresh weights):
#   ./services/iggm-server/scripts/vendor.sh          # → upstream/
#   ./services/iggm-server/scripts/fetch_weights.sh   # → weights/ then NAS
#
# IgGM source lives at services/iggm-server/upstream/ (vendored at pinned SHA).
# Checkpoints (5 .pth) on NAS at /data/models/iggm/ (FC mount; SIF / HPC
# `apptainer run --bind`).  NOT baked into the image.  A symlink
# /opt/iggm/checkpoints -> /data/models/iggm makes upstream's hardcoded
# ./checkpoints/<name>.pth resolve there (no runtime torch.hub download).
# See engineering/decisions/2026-06-26-service-weights-externalization.md.
#
# Multistage layout
# -----------------
#   builder = conda env (Python 3.10 + pytorch 2.0.1 + cu118 + openmm +
#             pdbfixer + numpy/scipy/biopython) + framework + cleanup.
#   runtime = lean: conda env from builder + IgGM source + server code.
#
# The inference path uses only torch/numpy/scipy/Bio/ml_collections/openmm/
# pdbfixer — NO PyG / torch_scatter / rdkit / dgl / deepspeed (those are
# environment.yaml training/notebook deps the design path never imports).

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
ENV PATH=/opt/conda/envs/iggm/bin:$PATH

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
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# ---------------------------------------------------------------------------
# Conda env — pytorch + the numpy-linked scientific stack, all from a single
# effective numpy source (conda-forge) so there is ONE numpy ABI.
#
# WHY all-conda for numpy here (vs semlaflow's all-pip): IgGM needs openmm +
# pdbfixer (module-level import in base_designer via PdbFixer) which are
# conda-only and pull a conda numpy anyway.  There is NO rdkit in the inference
# path, so the pip-rdkit-vs-conda-numpy segfault (see project memory
# `feedback_follow_upstream_pip_stack_numpy_abi`) does not apply; keeping
# numpy/scipy/biopython/openmm/pdbfixer all on conda-forge avoids a mixed ABI.
# pytorch (pytorch channel) is numpy-ABI forward-compatible with 1.x.
#
# cu118 (upstream pins cu117): the model is pure PyTorch (self-implemented
# Semla/ESM2/IPA, no compiled PyG extensions), so CUDA is decoupled from the
# code.  cu118 widens FC GPU coverage to Ada/L20 (sm_89), which cu117 drops.
#
# Explicit pytorch build string forces the CUDA variant (without it the solver
# can silently pick the CPU build → libc10_cuda.so missing at runtime).
#
# iJIT_NotifyEvent (Intel VTune ITT): pytorch weak-refs it; install ittapi
# (conda-forge) + pin mkl/intel-openmp <2024.1 (these live on `defaults`, do
# NOT prefix conda-forge).  See engineering/guides/pytorch-conda-ittapi.md.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n iggm \
        -c pytorch -c nvidia -c conda-forge \
        python=3.10 \
        pip \
        "setuptools<80" \
        "pytorch::pytorch=2.0.1=py3.10_cuda11.8_cudnn8.7.0_0" \
        pytorch-cuda=11.8 \
        pytorch::pytorch-mutex=1.0=cuda \
        conda-forge::openmm=8.1.1 \
        conda-forge::pdbfixer=1.9 \
        "conda-forge::numpy=1.23.5" \
        conda-forge::scipy \
        conda-forge::biopython=1.79 \
        conda-forge::ittapi \
        "mkl<2024.1" \
        "intel-openmp<2024.1"

# Pip deps — pure-python bits with no numpy ABI entanglement.
# ml-collections: config; dm-tree provides `import tree`; termcolor for the
# upstream colored logger; requests for the (unused-at-runtime) torch.hub path.
RUN /opt/conda/envs/iggm/bin/pip install --no-cache-dir \
        "ml-collections==0.1.1" \
        "dm-tree" \
        "termcolor" \
        "tqdm" \
        "requests" \
        "pyyaml"

# ---------------------------------------------------------------------------
# Build-time smoke #1: filesystem sanity — vendor.sh must have populated the
# source tree with all files the wrappers depend on.
# ---------------------------------------------------------------------------
COPY services/iggm-server/upstream /opt/iggm
RUN set -e; \
    for f in /opt/iggm/design.py \
             /opt/iggm/IgGM/__init__.py \
             /opt/iggm/IgGM/deploy/ab_design.py \
             /opt/iggm/IgGM/deploy/base_designer.py \
             /opt/iggm/IgGM/model/pretrain.py \
             /opt/iggm/IgGM/protein/__init__.py; do \
        [ -f "$f" ] || { echo "ERROR: $f missing; run scripts/vendor.sh"; exit 1; }; \
    done

# ---------------------------------------------------------------------------
# Build-time smoke #2: full upstream module chain — catches
#   1. iJIT_NotifyEvent undefined symbol (torch import fails)
#   2. resolver picking CPU-only pytorch (torch.version.cuda is None)
#   3. openmm / pdbfixer / ml_collections / dm-tree import failures
#   4. IgGM package import chain (AbDesigner + cal_ppi — the wrapper targets)
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/iggm/bin/python -c "\
import sys; sys.path.insert(0, '/opt/iggm'); \
import torch; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
assert torch.version.cuda is not None, \
    'pytorch built without CUDA — resolver picked CPU variant (check mkl/intel-openmp pins vs pytorch-cuda)'; \
from openmm.app import PDBFile; print('openmm OK'); \
from pdbfixer import PDBFixer; print('pdbfixer OK'); \
import ml_collections, tree, Bio; print('ml_collections/tree/Bio OK'); \
from IgGM.deploy import AbDesigner; \
from IgGM.protein import cal_ppi; \
from IgGM.protein.parser import parse_fasta; \
print('IgGM module chain OK')"

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env.
# ---------------------------------------------------------------------------
RUN find /opt/conda/envs/iggm/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/iggm/share/doc \
        /opt/conda/envs/iggm/share/man \
        /opt/conda/envs/iggm/share/locale \
    && find /opt/conda/envs/iggm/lib/python3.10/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/iggm -depth -type d \
        -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/iggm -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Shared service framework — independent layer.
#
# COPY (not bind-mount): BuildKit does not include bind-mount content in the
# layer cache key, so a bind would ship a stale framework even after the
# framework source is updated (would silently drop the output-sink fix).
# ---------------------------------------------------------------------------
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/iggm/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2


# ===========================================================================
# Stage 2: runtime — conda env + IgGM source + server code.
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/iggm/bin:$PATH

# LANG=C.UTF-8: FC containers default LANG unset → ascii locale crashes UTF-8
# file reads.  See engineering/guides/adding-a-new-service.md.
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# libgomp for openmp-linked libs; ca-certificates for OSS/HTTP.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Conda env from builder.
COPY --from=builder /opt/conda /opt/conda

# IgGM source from builder (vendored upstream).
COPY --from=builder /opt/iggm /opt/iggm

# Server code as `server` package — uvicorn imports `server.app:app`.  The two
# upstream wrappers (run_design.py + epitope.py) live here too and are invoked
# by tools.py as subprocesses.
COPY services/iggm-server/__init__.py \
     services/iggm-server/__main__.py \
     services/iggm-server/VERSION \
     services/iggm-server/app.py \
     services/iggm-server/adapter.py \
     services/iggm-server/models.py \
     services/iggm-server/settings.py \
     services/iggm-server/tools.py \
     services/iggm-server/run_design.py \
     services/iggm-server/epitope.py \
     /opt/iggm/server/

# checkpoints symlink -> NAS weights_dir.  Dangling at build (NAS not mounted);
# resolves once FC mounts /data.  Upstream reads os.getcwd()/checkpoints and
# cwd is /opt/iggm (subprocess_cwd + WORKDIR), so this is the path it hits.
RUN ln -s /data/models/iggm /opt/iggm/checkpoints

WORKDIR /opt/iggm

# PYTHONPATH — upstream root (so `import IgGM` / `import design`) + the `server`
# package.  Apptainer/SIF ignores WORKDIR, so set it explicitly.
ENV PYTHONPATH=/opt/iggm
ENV IGGM_ROOT=/opt/iggm
ENV IGGM_PYTHON=/opt/conda/envs/iggm/bin/python
ENV IGGM_DESIGN_SCRIPT=/opt/iggm/server/run_design.py
ENV IGGM_EPITOPE_SCRIPT=/opt/iggm/server/epitope.py
ENV IGGM_JOBS_BASE_DIR=/data/iggm_jobs

# Output-sink: when the gateway sends X-Bioagent-Oss-Prefix, the framework
# mirrors the finished job dir to this mount. Requires an OSS mount at /mnt/oss
# in the FC console (storage config). Absent mount => no-op (NAS-only).
ENV IGGM_OSS_OUTPUT_MOUNT=/mnt/oss

# Weights externalized — NAS mount injected by FC config / SIF --bind.
ENV IGGM_WEIGHTS_DIR=/data/models/iggm

RUN mkdir -p /data/iggm_jobs

# FC session affinity + single-GPU serial execution.
ENV IGGM_SESSION_HEADER_NAME=bioagent-session-id
ENV IGGM_MAX_CONCURRENT_JOBS=1

ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/iggm/bin/python", "-m", "uvicorn", \
     "server.app:app", "--host", "0.0.0.0", "--port", "9000", \
     "--timeout-keep-alive", "900"]
