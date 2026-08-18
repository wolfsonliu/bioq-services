# ---------- Megalodon Server ----------
# NVIDIA Megalodon — unconditional 3D de novo small-molecule generation
# (Reidenbach et al., arXiv:2505.18392, 2025).
#
# LICENSE: source Apache-2.0; model weights under the NVIDIA Open Model
# License.  See the vendored upstream tree + engineering/decisions/
# 2026-07-09-megalodon-server-design.md.
#
# Build (from bioagent project root):
#   docker build --platform linux/amd64 -t megalodon-server \
#       -f services/megalodon-server/Dockerfile .
#
# Prerequisites (one-time, re-run to bump pinned upstream commit / refresh weights):
#   ./services/megalodon-server/scripts/vendor.sh          # → upstream/
#   ./services/megalodon-server/scripts/fetch_weights.sh   # → weights/ then NAS
#
# Megalodon source lives at services/megalodon-server/upstream/ (vendored at
# pinned SHA via scripts/vendor.sh).  Checkpoints (6 variants, HuggingFace)
# + per-dataset statistics bundles live on NAS at /data/models/megalodon/ —
# NOT baked into the image (FC NAS mount; SIF / HPC `apptainer run --bind`).
# See engineering/decisions/2026-06-26-service-weights-externalization.md.
#
# Multistage layout
# -----------------
#   builder = conda env (Py 3.10 + pip) + pip torch 2.7 (cu126) + PyG 2.6.1
#             wheels (torch_scatter/sparse/cluster/spline_conv, NO pyg-lib)
#             + rdkit + lightning + hydra + framework + cleanup.
#   runtime = lean: conda env from builder + Megalodon source + server code.
#
# Base: nvidia/cuda 12.4.1 -runtime — the pip torch 2.7 cu126 wheels bundle
# their own CUDA userspace libs, so the base image CUDA version only needs a
# forward-compatible host driver (CUDA minor-version compat).  12.4.1 is
# already pulled across the project's other GPU services (e.g. pocketxmol),
# so reusing it unifies the base layer and avoids an extra registry pull.
# FC GPU T4 / A10 / L20 / Ada compatible.  Blackwell (sm_120, needs CUDA
# >= 12.8) would require cu128 wheels — out of scope for v0.0.1.

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
# libxrender1 / libxext6: the pip rdkit wheel links libXrender.so.1 +
# libXext.so.6 (needed even for headless Chem imports).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential wget ca-certificates libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
ENV MAMBA_EXE=/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/megalodon/bin:$PATH

# ---------------------------------------------------------------------------
# Mirror configuration (Huawei pip + PKU conda) for CN-friendly builds.
# ---------------------------------------------------------------------------
COPY deploy/conda/mirrors.condarc /tmp/mirrors.condarc
RUN mkdir -p /root/.config/pip \
    && printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
       > /root/.config/pip/pip.conf \
    && cat > /root/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
channel_priority: strict
EOF
RUN cat /tmp/mirrors.condarc >> /root/.condarc

# ---------------------------------------------------------------------------
# Conda env — ONLY python + pip.  Everything else (torch / PyG / rdkit /
# lightning / hydra / einops / pandas) goes through pip, exactly as upstream
# requirements.txt does.
#
# WHY all-pip: mixing conda numpy (one channel) with a pip rdkit/torch wheel
# (built against pip numpy) risks a cross-channel numpy ABI mismatch that
# segfaults (rc=-11) at torch.from_numpy(<rdkit array>).  Following upstream's
# all-pip stack keeps a single numpy ABI.  See project memory
# feedback_follow_upstream_pip_stack_numpy_abi.  Bonus: the conda solve is
# tiny (strict priority parses in seconds).
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n megalodon \
        -c conda-forge \
        python=3.10 \
        pip \
        "setuptools<80"

# ---------------------------------------------------------------------------
# pytorch 2.7 + cu126 + core deps + PyG 2.6.1 + native ext wheels
# ---------------------------------------------------------------------------
# Upstream requirements.txt pins torch==2.7.0 + PyG stack from
#   https://data.pyg.org/whl/torch-2.7.0+cu126.html
# with torch_geometric==2.6.1 and (unused here) pyg-lib==0.4.0.
#
# **pyg-lib is intentionally NOT installed.**  Megalodon's code only uses
# torch_scatter (scatter / scatter_mean); it never calls knn_graph /
# radius_graph / torch.ops.pyg (verified 2026-07-09).  torch_geometric==2.6.1
# routes any pool ops through torch_cluster anyway, and the pyg-lib
# torch-2.7+cu126 wheel is the least reliably published of the set — skipping
# it avoids an sdist build / ABI mismatch for zero benefit.  See
# engineering/guides/pyg-knn-graph-requires-pyg-lib.md.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/megalodon/bin/python \
        --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.7.0

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/megalodon/bin/python \
        "torch_geometric==2.6.1" \
        hydra-core==1.3.2 \
        "rdkit==2025.3.2" \
        "lightning==2.5.1.post0" \
        einops==0.8.1 \
        pandas==2.2.3 \
        omegaconf

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/megalodon/bin/python \
        --find-links https://data.pyg.org/whl/torch-2.7.0+cu126.html \
        torch_scatter==2.1.2 torch_sparse==0.6.18 \
        torch_cluster==1.6.3 torch_spline_conv==1.2.2

# ---------------------------------------------------------------------------
# Megalodon source — from vendored upstream/ (scripts/vendor.sh).  src/ layout
# (pyproject `where=["src"]`); we put src on PYTHONPATH instead of pip install.
# ---------------------------------------------------------------------------
COPY services/megalodon-server/upstream /opt/megalodon

# Build-time smoke: torch CUDA + PyG backend + the exact inference import
# chain (module + metrics) + a real get_molecules-adjacent import.  Fails at
# build so we don't discover missing deps at FC deploy time.
RUN cd /opt/megalodon && PYTHONPATH=/opt/megalodon/src \
    /opt/conda/envs/megalodon/bin/python -c "\
import torch; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
assert torch.version.cuda is not None, \
    'pytorch built without CUDA — pip wheel may have fallen back to CPU'; \
from torch_scatter import scatter_mean; print('torch_scatter ok'); \
import torch_geometric; assert torch_geometric.__version__.startswith('2.6'), torch_geometric.__version__; \
print('pyg', torch_geometric.__version__); \
from rdkit import Chem; print('rdkit ok'); \
import lightning, hydra, omegaconf, einops, pandas; print('lightning/hydra/omegaconf/einops/pandas ok'); \
from megalodon.models.module import Graph3DInterpolantModel; \
from megalodon.metrics.molecule import get_molecules; \
from megalodon.data.statistics import Statistics; \
from megalodon.metrics.molecule_evaluation_callback import MoleculeEvaluationCallback; \
import numpy.testing; \
print('megalodon module chain OK, cuda=' + torch.version.cuda)"

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env.
#
# NB: do NOT blanket-delete every `tests`/`test` dir under site-packages —
# numpy 2.x ships `numpy/_core/tests/_natype.py`, imported at runtime by
# `numpy.testing._private.utils` (pulled in via lightning → torchmetrics →
# scipy.signal). Removing it raises `ModuleNotFoundError: numpy._core.tests`
# on first real import.  The tests dirs save little; skip that deletion.
# ---------------------------------------------------------------------------
RUN find /opt/conda/envs/megalodon/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/megalodon/share/doc \
        /opt/conda/envs/megalodon/share/man \
        /opt/conda/envs/megalodon/share/locale \
    && find /opt/conda/envs/megalodon -depth -type d \
        -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/megalodon -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Shared service framework — independent layer so framework bumps don't
# invalidate the conda env layer above.
# ---------------------------------------------------------------------------
COPY framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/megalodon/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2


# ===========================================================================
# Stage 2: runtime — conda env + Megalodon source + server code.
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/megalodon/bin:$PATH

# LANG=C.UTF-8: FC containers default LANG unset → ascii locale crashes
# UTF-8 file reads.  See engineering/guides/adding-a-new-service.md.
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# libgomp1 for openmp-linked libs; ca-certificates for OSS/HTTP;
# libxrender1 / libxext6 for the pip rdkit wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Conda env from builder.
COPY --from=builder /opt/conda /opt/conda

# Megalodon source from builder (vendored upstream).
COPY --from=builder /opt/megalodon /opt/megalodon

# Server code as `server` package — uvicorn imports `server.app:app`.
COPY services/megalodon-server/__init__.py \
     services/megalodon-server/__main__.py \
     services/megalodon-server/VERSION \
     services/megalodon-server/app.py \
     services/megalodon-server/adapter.py \
     services/megalodon-server/configs.py \
     services/megalodon-server/inference.py \
     services/megalodon-server/models.py \
     services/megalodon-server/settings.py \
     services/megalodon-server/tools.py \
     /opt/megalodon/server/

WORKDIR /opt/megalodon

# PYTHONPATH — repo root (for the `server` package) + src/ (for `megalodon`).
# Apptainer/SIF ignores WORKDIR, so both must be explicit.
ENV PYTHONPATH=/opt/megalodon:/opt/megalodon/src
ENV MEGALODON_ROOT=/opt/megalodon
ENV MEGALODON_PYTHON=/opt/conda/envs/megalodon/bin/python
ENV MEGALODON_INFERENCE_SCRIPT=/opt/megalodon/server/inference.py
ENV MEGALODON_CONF_DIR=/opt/megalodon/scripts/conf
ENV MEGALODON_JOBS_BASE_DIR=/data/megalodon_jobs

# Weights + statistics externalized — NAS mount injected by FC config / SIF --bind.
ENV MEGALODON_WEIGHTS_DIR=/data/models/megalodon

RUN mkdir -p /data/megalodon_jobs

# Output-sink: 当 gateway 传 X-Bioagent-Oss-Prefix 时，framework 把 job dir 镜像到此挂载点。
# 需在 FC 控制台把数据面 OSS bucket 挂到 /mnt/oss。缺挂载 => no-op（仅 NAS）。
ENV MEGALODON_OSS_OUTPUT_MOUNT=/mnt/oss

# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance.
ENV MEGALODON_SESSION_HEADER_NAME=bioagent-session-id
ENV MEGALODON_MAX_CONCURRENT_JOBS=1

ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/megalodon/bin/python", "-m", "uvicorn", \
     "server.app:app", "--host", "0.0.0.0", "--port", "9000", \
     "--timeout-keep-alive", "900"]
