# Build from bioagent project root:
#   docker build --platform linux/amd64 -t flowmol-server \
#       -f services/flowmol-server/Dockerfile .
#
# Pre-build (once; re-run to bump pinned upstream commit / refresh weights):
#   ./services/flowmol-server/scripts/vendor.sh          # → upstream/
#   ./services/flowmol-server/scripts/fetch_weights.sh   # → weights/ then NAS
#
# FlowMol3 source lives at services/flowmol-server/upstream/ (vendored at
# pinned SHA via scripts/vendor.sh).  22 pretrained checkpoints on NAS at
# /data/models/flowmol/trained_models/<variant>/ (FC mount; SIF / HPC
# `apptainer run --bind`).  NOT baked into the image.
# See engineering/decisions/2026-06-26-service-weights-externalization.md.
#
# Multistage layout
# -----------------
#   builder = conda env (Python 3.10 + pytorch 2.2 + cu121 + dgl 2.0 +
#             PyG scatter/cluster + rdkit) + framework + cleanup.
#   runtime = lean: conda env from builder + FlowMol source + server code.
#
# Base: nvidia/cuda 12.1.0 -runtime — matches pytorch-cuda=12.1 + dgl cu121.
# (12.1.0 is what's cached on the build host; pytorch cu121 wheels are ABI-
# compatible with any 12.1.x runtime.)

ARG BASE_IMAGE=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
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
ENV PATH=/opt/conda/envs/flowmol/bin:$PATH

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
# custom_channels: only map channels that TUNA hosts.  Do NOT map `pyg`,
# `nvidia`, or `dglteam` — TUNA returns 404 for those; they fall through
# to the upstream https://conda.anaconda.org/{pyg,nvidia,dglteam}/.
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# ---------------------------------------------------------------------------
# Conda env — mirror upstream `environment.yml` (Python 3.10 + pytorch 2.2 +
# cu121 + dgl 2.0.0.cu121 + pyg scatter/cluster + rdkit + lightning).  Removed
# from env vs upstream: wandb/sentry/gitpython (training-only); posebusters/xtb
# (metrics-only, v0.0.1 out of scope).
#
# `pytorch=2.2.0=py3.10_cuda12.1_cudnn8.9.2_0` — explicit CUDA build string.
# Without it, the resolver can silently pick the CPU variant
# (`=py3.10_cpu_0`) under the mkl<2024.1 constraint below — leading to
# `libc10_cuda.so: cannot open shared object file` at torch_sparse import
# (verified 2026-07-04 on diffdock-pp-server FC build).
#
# iJIT_NotifyEvent (Intel VTune ITT) — pytorch's libtorch weak-references it.
# Missing symbol → `import torch` fails.  `mkl` historically bundled it but
# mkl 2024.1+ removed the collector.  Belt-and-braces: install both `ittapi`
# (source of truth for pytorch 2.x) and pin `mkl<2024.1` / `intel-openmp
# <2024.1` (fallback + keeps pyg's ABI stable).  See
# engineering/guides/pytorch-conda-ittapi.md and pytorch/pytorch#123097.
#
# `dgl=2.0.0.cu121` (from dglteam/label/cu121) is the ONE build that matches
# pytorch 2.2 + cu121.  Do NOT bump pytorch or CUDA without verifying dgl has
# a corresponding build (dgl cadence is 3-6 months behind pytorch).
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n flowmol \
        -c pyg -c nvidia -c pytorch -c "dglteam/label/cu121" -c conda-forge \
        python=3.10 \
        pip \
        "setuptools<80" \
        "pytorch::pytorch=2.2.0=py3.10_cuda12.1_cudnn8.9.2_0" \
        pytorch-cuda=12.1 \
        pytorch::torchvision=0.17.0 \
        pytorch::torchaudio=2.2.0 \
        pytorch::pytorch-mutex=1.0=cuda \
        pyg::pytorch-scatter=2.1.2 \
        pyg::pytorch-cluster=1.6.3 \
        "dglteam/label/cu121::dgl=2.0.0.cu121" \
        pytorch-lightning=2.1.3 \
        torchmetrics=1.2.1 \
        numpy=1.26.3 \
        scipy=1.12.0 \
        pandas=2.2.0 \
        matplotlib-base=3.8.2 \
        einops=0.7.0 \
        conda-forge::rdkit=2023.09.5 \
        conda-forge::ittapi \
        "mkl<2024.1" \
        "intel-openmp<2024.1" \
        tqdm=4.66.1 \
        pyyaml=6.0.1

# Pip-only deps that don't have clean conda packages.
# `click` — undeclared dep of useful-rdkit-utils (ring_systems.py imports it).
RUN /opt/conda/envs/flowmol/bin/pip install --no-cache-dir \
        useful-rdkit-utils==0.3.12 \
        torch-ema==0.3 \
        appdirs==1.4.4 \
        click

# ---------------------------------------------------------------------------
# Build-time smoke #1: filesystem sanity — vendor.sh must have populated
# the source tree with all files the wrapper depends on.  Fails in <1 s if
# vendor.sh was skipped / partial / had a stale exclude rule.
# ---------------------------------------------------------------------------
COPY services/flowmol-server/upstream /opt/flowmol
RUN set -e; \
    for f in /opt/flowmol/flowmol/__init__.py \
             /opt/flowmol/flowmol/models/flowmol.py \
             /opt/flowmol/flowmol/analysis/molecule_builder.py \
             /opt/flowmol/flowmol/model_utils/load.py \
             /opt/flowmol/test.py \
             /opt/flowmol/data/geom_full_kekulized/train_data_marginal_dists.pt \
             /opt/flowmol/data/geom_full_kekulized/train_data_n_atoms_histogram.pt; do \
        [ -f "$f" ] || { echo "ERROR: $f missing; run scripts/vendor.sh"; exit 1; }; \
    done

# ---------------------------------------------------------------------------
# Build-time smoke #2: full upstream module chain — catches
#   1. iJIT_NotifyEvent undefined symbol (torch import fails)
#   2. resolver picking CPU-only pytorch (torch.version.cuda is None → later
#      torch_sparse / dgl import fails with libc10_cuda.so missing)
#   3. torch_geometric / dgl ABI mismatch vs pytorch 2.2 cu121
#   4. `wandb` + `posebusters` dead-imports in flowmol.analysis.metrics
#      (must be stubbed before we touch flowmol.*; the runtime path mirrors
#      this in server/inference.py)
# Fail fast at build so we don't discover it at FC deploy time.
# See engineering/guides/adding-a-new-service.md
# §"包装 conda-based upstream 的常见陷阱".
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/flowmol/bin/python -c "\
import sys, types; \
sys.modules.setdefault('wandb', types.ModuleType('wandb')); \
sys.modules.setdefault('posebusters', types.ModuleType('posebusters')); \
sys.path.insert(0, '/opt/flowmol'); \
import torch; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
assert torch.version.cuda is not None, \
    'pytorch built without CUDA — resolver picked CPU variant (check mkl/intel-openmp pins vs pytorch-cuda)'; \
import torch_scatter; print('torch_scatter OK'); \
import torch_cluster; print('torch_cluster OK'); \
import dgl; print('dgl', dgl.__version__); \
import pytorch_lightning as pl; print('pytorch-lightning', pl.__version__); \
from flowmol.models.flowmol import FlowMol; \
from flowmol.analysis.molecule_builder import SampledMolecule; \
from flowmol.model_utils.load import read_config_file; \
print('flowmol module chain OK')"

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env.
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/flowmol/bin/pip uninstall -y \
        jupyterlab jupyter_server jupyterlab_server jupyter_client \
        ipywidgets ipykernel notebook nbconvert nbformat || true \
    && find /opt/conda/envs/flowmol/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/flowmol/share/doc \
        /opt/conda/envs/flowmol/share/man \
        /opt/conda/envs/flowmol/share/locale \
    && find /opt/conda/envs/flowmol/lib/python3.10/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/flowmol -depth -type d \
        -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/flowmol -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Shared service framework — independent layer so framework bumps don't
# invalidate the conda env layer above.
# ---------------------------------------------------------------------------
RUN --mount=type=bind,source=services/_framework,target=/tmp/service-framework \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/flowmol/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2


# ===========================================================================
# Stage 2: runtime — conda env + FlowMol source + server code.
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/flowmol/bin:$PATH

# LANG=C.UTF-8: FC containers default LANG unset → Python locale.
# getpreferredencoding()='ascii', crashes any `open()`-of-UTF-8 config file
# in upstream code.  This bit diffdock-pp v0.0.6 (yaml comment with em dash).
# See engineering/guides/adding-a-new-service.md §"包装 conda-based upstream".
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# libgomp for openmp-linked libs; ca-certificates for OSS/HTTP.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Conda env from builder.
COPY --from=builder /opt/conda /opt/conda

# FlowMol source from builder (vendored upstream).
COPY --from=builder /opt/flowmol /opt/flowmol

# Server code as `server` package — uvicorn imports `server.app:app`.
COPY services/flowmol-server/__init__.py \
     services/flowmol-server/__main__.py \
     services/flowmol-server/VERSION \
     services/flowmol-server/app.py \
     services/flowmol-server/adapter.py \
     services/flowmol-server/models.py \
     services/flowmol-server/settings.py \
     services/flowmol-server/tools.py \
     services/flowmol-server/inference.py \
     /opt/flowmol/server/

WORKDIR /opt/flowmol

# PYTHONPATH — root (`server` package) + upstream root (so `flowmol` imports
# resolve without pip install).  Apptainer/SIF ignores WORKDIR, so PYTHONPATH
# must contain /opt/flowmol explicitly.
ENV PYTHONPATH=/opt/flowmol
ENV FLOWMOL_ROOT=/opt/flowmol
ENV FLOWMOL_PYTHON=/opt/conda/envs/flowmol/bin/python
ENV FLOWMOL_INFERENCE_SCRIPT=/opt/flowmol/server/inference.py
ENV FLOWMOL_JOBS_BASE_DIR=/data/flowmol_jobs

# Weights externalized — NAS mount injected by FC config / SIF --bind.
ENV FLOWMOL_WEIGHTS_DIR=/data/models/flowmol

RUN mkdir -p /data/flowmol_jobs

# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance.
ENV FLOWMOL_SESSION_HEADER_NAME=bioagent-session-id
ENV FLOWMOL_MAX_CONCURRENT_JOBS=1

ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/flowmol/bin/python", "-m", "uvicorn", \
     "server.app:app", "--host", "0.0.0.0", "--port", "9000", \
     "--timeout-keep-alive", "900"]
