# Build from bioagent project root:
#   docker build --platform linux/amd64 -t diffusion-hopping-server \
#       -f services/diffusion-hopping-server/Dockerfile .
#
# Pre-build (once; re-run to bump pinned upstream commit / refresh weights):
#   ./services/diffusion-hopping-server/scripts/vendor.sh         # → upstream/
#   ./services/diffusion-hopping-server/scripts/fetch_weights.sh  # → weights/
#
# DiffHopp source lives at services/diffusion-hopping-server/upstream/
# (vendored at pinned SHA via scripts/vendor.sh).  4 .ckpt weights (~189 MB)
# live on NAS at /data/models/diffusion-hopping/checkpoints/ — NOT baked into
# the image (FC NAS mount; SIF / HPC `apptainer run --bind`).
# See engineering/decisions/2026-06-26-service-weights-externalization.md.
#
# Multistage layout
# -----------------
#   builder = conda env (Python 3.10 + pytorch 1.13 + cu117 + PyG 2.2 +
#             rdkit + openbabel + reduce) + framework + cleanup.
#   runtime = lean: conda env from builder + DiffHopp source + server code.
#
# Base: nvidia/cuda 11.8 -runtime (forward-compatible with conda's
# pytorch-cuda=11.7; DiffHopp pins pytorch 1.13.1 which has no >11.8 wheels).

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
ENV PATH=/opt/conda/envs/diffusion_hopping/bin:$PATH

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
# big channels like conda-forge.  Without this, the old pytorch 1.13.1 +
# cu117 + pyg 2.2 + Py3.10 resolve takes 2+ hours and risks timing out.
channel_priority: strict
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
# custom_channels: only map channels that TUNA actually hosts.
# IMPORTANT: do NOT map `pyg` or `nvidia` here — TUNA returns 404 for those
# channels (verified 2026-06-28).  They fall through to the upstream
# https://conda.anaconda.org/{pyg,nvidia}/ which works fine.
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# ---------------------------------------------------------------------------
# Conda env — based on upstream's environment.yml (Python 3.10 + pytorch
# 1.13.1 + cu117 + PyG 2.2 + scatter + rdkit + openbabel) PLUS `reduce`
# (Duke biochemistry tool; bioconda). Removed wandb (not needed at inference).
#
# iJIT_NotifyEvent fix (belt-and-braces): pytorch 1.13's libtorch_cpu.so
# weak-references `iJIT_NotifyEvent` (Intel VTune ITT).  Without a provider
# in the env's lib/, `import torch` blows up with `undefined symbol:
# iJIT_NotifyEvent`.  Historically `mkl` bundled the symbol, but mkl 2024.1+
# removed it — hence the failure on fresh builds.
#
# For pytorch 2.0.1 (deeprank-ab), `ittapi` alone fixes it; for pytorch 1.13
# `ittapi` is insufficient (verified 2026-07-03 on FC) — the loader still
# reports the symbol undefined.  Pin `mkl<2024.1` / `intel-openmp<2024.1`
# so the mkl-provided collector still resolves the symbol.  Keep `ittapi`
# too — it's harmless and gives a second-source for pytorch 2.x rebuilds.
#
# Build-string pin (`=py3.10_cuda11.7_cudnn8.5.0_0`) forces the CUDA variant
# of pytorch.  Without it, the resolver silently picks the CPU-only build
# (`=py3.10_cpu_0`) under the mkl<2024.1 + intel-openmp<2024.1 constraints
# — leading to `libc10_cuda.so: cannot open shared object file` at
# torch_sparse import (verified 2026-07-04 on FC).
#
# `setuptools<80`: setuptools 80.0 (2025-04) removed the `pkg_resources`
# module; pytorch-lightning 1.8.1's __init__ still uses `pkg_resources.
# declare_namespace` at import time.  Pin below 80 to keep it.  (Not just
# a hypothetical — hit this on 2026-07-05.)
#
# See engineering/guides/pytorch-conda-ittapi.md and pytorch/pytorch#123097.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n diffusion_hopping \
        -c pyg -c nvidia -c pytorch -c conda-forge -c bioconda \
        python=3.10 \
        pip \
        "setuptools<80" \
        "pytorch::pytorch=1.13.1=py3.10_cuda11.7_cudnn8.5.0_0" \
        pytorch-cuda=11.7 \
        torchvision=0.14.1 \
        pyg::pyg=2.2.0 \
        pyg::pytorch-scatter=2.1.0 \
        pytorch-lightning=1.8.1 \
        python-lmdb=1.4.0 \
        numpy=1.24.1 \
        pandas=1.5.2 \
        biopandas=0.4.1 \
        biopython=1.80 \
        conda-forge::rdkit=2022.09.3 \
        openbabel=3.1.1 \
        tqdm=4.64.1 \
        bioconda::reduce \
        conda-forge::ittapi \
        "mkl<2024.1" \
        "intel-openmp<2024.1"

# ---------------------------------------------------------------------------
# Build-time smoke test — catches:
#   1. iJIT_NotifyEvent undefined symbol (torch import fails)
#   2. resolver silently picking CPU-only pytorch (torch.version.cuda is None
#      → later torch_sparse import fails with `libc10_cuda.so: cannot open
#      shared object file` on FC; observed 2026-07-04)
#   3. torch_geometric / torch_sparse ABI mismatch
# Fail fast at build so we don't discover it at FC deploy time.
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/diffusion_hopping/bin/python -c "\
import torch; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
assert torch.version.cuda is not None, \
    'pytorch built without CUDA — resolver picked CPU variant (check mkl/intel-openmp pins vs pytorch-cuda)'; \
import torch_geometric; print('pyg', torch_geometric.__version__); \
import torch_sparse; print('torch_sparse OK'); \
import torch_scatter; print('torch_scatter OK'); \
import pytorch_lightning as pl; print('pytorch-lightning', pl.__version__); \
import pkg_resources; print('pkg_resources OK')"

# ---------------------------------------------------------------------------
# DiffHopp source — from vendored upstream/ (see scripts/vendor.sh).
# We install it editable so `from diffusion_hopping import ...` resolves.
# Upstream has no setup.py / pyproject — we use pip's PEP 660 "src as
# package" via a tiny shim by adding the dir to PYTHONPATH at runtime.
# ---------------------------------------------------------------------------
COPY services/diffusion-hopping-server/upstream /opt/diffusion-hopping
RUN /opt/conda/envs/diffusion_hopping/bin/python -c \
    "import sys; sys.path.insert(0, '/opt/diffusion-hopping'); \
     import diffusion_hopping; print('diffusion_hopping OK')"

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env.
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/diffusion_hopping/bin/pip uninstall -y \
        jupyterlab jupyter_server jupyterlab_server jupyter_client \
        ipywidgets ipykernel notebook nbconvert nbformat || true \
    && find /opt/conda/envs/diffusion_hopping/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/diffusion_hopping/share/doc \
        /opt/conda/envs/diffusion_hopping/share/man \
        /opt/conda/envs/diffusion_hopping/share/locale \
    && find /opt/conda/envs/diffusion_hopping/lib/python3.10/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/diffusion_hopping -depth -type d \
        -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/diffusion_hopping -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Shared service framework — independent layer so framework bumps don't
# invalidate the conda env layer above.
# ---------------------------------------------------------------------------
COPY framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/diffusion_hopping/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2


# ===========================================================================
# Stage 2: runtime — conda env + DiffHopp source + server code.
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/diffusion_hopping/bin:$PATH

# `reduce` (copied from conda env via the builder COPY below) writes its
# own temp files; openbabel needs libgomp; ca-certificates for OSS / HTTP.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Conda env from builder.
COPY --from=builder /opt/conda /opt/conda

# DiffHopp source from builder (vendored upstream).
COPY --from=builder /opt/diffusion-hopping /opt/diffusion-hopping

# Server code as `server` package — uvicorn imports `server.app:app`.
COPY services/diffusion-hopping-server/__init__.py \
     services/diffusion-hopping-server/__main__.py \
     services/diffusion-hopping-server/VERSION \
     services/diffusion-hopping-server/app.py \
     services/diffusion-hopping-server/adapter.py \
     services/diffusion-hopping-server/models.py \
     services/diffusion-hopping-server/settings.py \
     services/diffusion-hopping-server/tools.py \
     services/diffusion-hopping-server/inference.py \
     /opt/diffusion-hopping/server/

WORKDIR /opt/diffusion-hopping

# PYTHONPATH — root (`server` package) + upstream root (so `diffusion_hopping`
# imports resolve without pip install).  Apptainer/SIF ignores WORKDIR, so
# PYTHONPATH must contain /opt/diffusion-hopping explicitly.
ENV PYTHONPATH=/opt/diffusion-hopping
ENV DIFFUSION_HOPPING_ROOT=/opt/diffusion-hopping
ENV DIFFUSION_HOPPING_PYTHON=/opt/conda/envs/diffusion_hopping/bin/python
ENV DIFFUSION_HOPPING_INFERENCE_SCRIPT=/opt/diffusion-hopping/server/inference.py
ENV DIFFUSION_HOPPING_JOBS_BASE_DIR=/data/diffusion_hopping_jobs
# Output-sink: when gateway sends X-Bioagent-Oss-Prefix, framework mirrors the
# job dir here. Requires the data-plane OSS bucket mounted at /mnt/oss RW in the
# FC console. Absent mount => no-op (NAS only). Default already /mnt/oss;
# explicit for self-documentation.
ENV DIFFUSION_HOPPING_OSS_OUTPUT_MOUNT=/mnt/oss

# Weights externalized — NAS mount injected by FC config / SIF --bind.
ENV DIFFUSION_HOPPING_WEIGHTS_DIR=/data/models/diffusion-hopping/checkpoints

RUN mkdir -p /data/diffusion_hopping_jobs

# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance.
ENV DIFFUSION_HOPPING_SESSION_HEADER_NAME=bioagent-session-id
ENV DIFFUSION_HOPPING_MAX_CONCURRENT_JOBS=1

ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/diffusion_hopping/bin/python", "-m", "uvicorn", \
     "server.app:app", "--host", "0.0.0.0", "--port", "9000", \
     "--timeout-keep-alive", "900"]
