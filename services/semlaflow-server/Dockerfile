# Build from bioagent project root:
#   docker build --platform linux/amd64 -t semlaflow-server \
#       -f services/semlaflow-server/Dockerfile .
#
# Pre-build (once; re-run to bump pinned upstream commit / refresh weights):
#   ./services/semlaflow-server/scripts/vendor.sh          # → upstream/
#   ./services/semlaflow-server/scripts/fetch_weights.sh   # → weights/ then NAS
#
# SemlaFlow source lives at services/semlaflow-server/upstream/ (vendored at
# pinned SHA via scripts/vendor.sh).  Checkpoints + reference .smol datasets
# on NAS at /data/models/semlaflow/<model>/ (FC mount; SIF / HPC
# `apptainer run --bind`).  NOT baked into the image.
# See engineering/decisions/2026-06-26-service-weights-externalization.md.
#
# Multistage layout
# -----------------
#   builder = conda env (Python 3.11 + pytorch 2.2 + cu121 + rdkit +
#             lightning + openbabel-wheel) + framework + cleanup.
#   runtime = lean: conda env from builder + SemlaFlow source + server code.
#
# Unlike flowmol/diffdock, SemlaFlow uses a self-contained equivariant
# attention architecture — NO dgl / PyG.  The env is correspondingly simpler.

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
# libxrender1 / libxext6: the pip rdkit wheel links libXrender.so.1 +
# libXext.so.6 (needed even for headless Chem imports).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential wget ca-certificates libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
ENV MAMBA_EXE=/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/semlaflow/bin:$PATH

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
# Conda env — ONLY the pytorch/CUDA stack + the ITT fix.  Everything else
# (numpy / scipy / pandas / rdkit / lightning / torchmetrics / openbabel)
# goes through pip, exactly as upstream `environment.yaml` does.
#
# WHY not conda for the sci stack: mixing conda numpy (one channel) with a
# pip rdkit wheel (built against pip numpy) risks a cross-channel numpy ABI
# mismatch that segfaults (rc=-11) at `torch.from_numpy(<rdkit array>)`.
# Following upstream's all-pip stack keeps a single numpy ABI.  See project
# memory `feedback_follow_upstream_pip_stack_numpy_abi`.  This also keeps the
# conda solve tiny (strict priority parses in seconds, not the 73 min a
# flexible multi-channel numpy solve took).
#
# `pytorch=2.2.0=py3.11_cuda12.1_cudnn8.9.2_0` — explicit CUDA build string.
# Without it the resolver can silently pick the CPU variant under the
# mkl<2024.1 constraint → `libc10_cuda.so` missing at runtime.
#
# iJIT_NotifyEvent (Intel VTune ITT): pytorch weak-references it; missing
# symbol → `import torch` fails.  Install `ittapi` (conda-forge) + pin mkl /
# intel-openmp <2024.1 (both live on `defaults`, NOT conda-forge — do not
# prefix them).  See engineering/guides/pytorch-conda-ittapi.md.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n semlaflow \
        -c nvidia -c pytorch -c conda-forge \
        python=3.11 \
        pip \
        "setuptools<80" \
        "pytorch::pytorch=2.2.0=py3.11_cuda12.1_cudnn8.9.2_0" \
        pytorch-cuda=12.1 \
        pytorch::pytorch-mutex=1.0=cuda \
        conda-forge::ittapi \
        "mkl<2024.1" \
        "intel-openmp<2024.1"

# Pip deps — mirror upstream `environment.yaml` pip block (single numpy ABI).
# openbabel-wheel provides `from openbabel import pybel` (scriptutil top-level
# import) without conda openbabel build pain.  lightning pinned to the 2.2 era
# matching the pytorch 2.2 stack and the checkpoint's expected pickle format
# (see design §12.4).  pyyaml is for settings.py manifest.yaml parsing.
RUN /opt/conda/envs/semlaflow/bin/pip install --no-cache-dir \
        "numpy==1.26.2" \
        "scipy==1.11.4" \
        "pandas==2.2.2" \
        "rdkit==2023.9.6" \
        "lightning==2.2.5" \
        "torchmetrics==1.2.1" \
        "openbabel-wheel==3.1.1.21" \
        "typing_extensions>=4.7" \
        "pyyaml==6.0.1" \
        "tqdm==4.66.1"

# ---------------------------------------------------------------------------
# Build-time smoke #1: filesystem sanity — vendor.sh must have populated the
# source tree with all files the wrapper depends on.  Fails in <1 s if
# vendor.sh was skipped / partial / had a stale exclude rule.
# ---------------------------------------------------------------------------
COPY services/semlaflow-server/upstream /opt/semlaflow
RUN set -e; \
    for f in /opt/semlaflow/semlaflow/__init__.py \
             /opt/semlaflow/semlaflow/predict.py \
             /opt/semlaflow/semlaflow/scriptutil.py \
             /opt/semlaflow/semlaflow/models/semla.py \
             /opt/semlaflow/semlaflow/models/fm.py \
             /opt/semlaflow/semlaflow/data/datamodules.py; do \
        [ -f "$f" ] || { echo "ERROR: $f missing; run scripts/vendor.sh"; exit 1; }; \
    done

# ---------------------------------------------------------------------------
# Build-time smoke #2: full upstream module chain — catches
#   1. iJIT_NotifyEvent undefined symbol (torch import fails)
#   2. resolver picking CPU-only pytorch (torch.version.cuda is None)
#   3. openbabel-wheel / rdkit / torchmetrics import failures
#   4. semlaflow package import chain (predict.py reuse targets)
# Fail fast at build so we don't discover it at FC deploy time.
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/semlaflow/bin/python -c "\
import sys, types; \
sys.modules.setdefault('wandb', types.ModuleType('wandb')); \
sys.path.insert(0, '/opt/semlaflow'); \
import torch; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
assert torch.version.cuda is not None, \
    'pytorch built without CUDA — resolver picked CPU variant (check mkl/intel-openmp pins vs pytorch-cuda)'; \
from openbabel import pybel; print('openbabel OK'); \
import lightning as L; print('lightning', L.__version__); \
import torchmetrics; print('torchmetrics', torchmetrics.__version__); \
import semlaflow.scriptutil as u; \
from semlaflow.predict import load_model, dm_from_ckpt, save_rdkit_sdf, save_raw_smol; \
print('semlaflow module chain OK')"

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env.
# ---------------------------------------------------------------------------
RUN find /opt/conda/envs/semlaflow/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/semlaflow/share/doc \
        /opt/conda/envs/semlaflow/share/man \
        /opt/conda/envs/semlaflow/share/locale \
    && find /opt/conda/envs/semlaflow/lib/python3.11/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/semlaflow -depth -type d \
        -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/semlaflow -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Shared service framework — independent layer so framework bumps don't
# invalidate the conda env layer above.
# ---------------------------------------------------------------------------
COPY framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/semlaflow/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2


# ===========================================================================
# Stage 2: runtime — conda env + SemlaFlow source + server code.
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/semlaflow/bin:$PATH

# LANG=C.UTF-8: FC containers default LANG unset → ascii locale crashes
# UTF-8 file reads.  See engineering/guides/adding-a-new-service.md.
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# libgomp for openmp-linked libs; ca-certificates for OSS/HTTP;
# libxrender1 / libxext6 for the pip rdkit wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Conda env from builder.
COPY --from=builder /opt/conda /opt/conda

# SemlaFlow source from builder (vendored upstream).
COPY --from=builder /opt/semlaflow /opt/semlaflow

# Server code as `server` package — uvicorn imports `server.app:app`.
COPY services/semlaflow-server/__init__.py \
     services/semlaflow-server/__main__.py \
     services/semlaflow-server/VERSION \
     services/semlaflow-server/app.py \
     services/semlaflow-server/adapter.py \
     services/semlaflow-server/models.py \
     services/semlaflow-server/settings.py \
     services/semlaflow-server/tools.py \
     services/semlaflow-server/inference.py \
     /opt/semlaflow/server/

WORKDIR /opt/semlaflow

# PYTHONPATH — service root (`server` package) + upstream root (so
# `import semlaflow` resolves without pip install).  Apptainer/SIF ignores
# WORKDIR, so PYTHONPATH must contain /opt/semlaflow explicitly.
ENV PYTHONPATH=/opt/semlaflow
ENV SEMLAFLOW_ROOT=/opt/semlaflow
ENV SEMLAFLOW_PYTHON=/opt/conda/envs/semlaflow/bin/python
ENV SEMLAFLOW_INFERENCE_SCRIPT=/opt/semlaflow/server/inference.py
# Output-sink: when gateway sends X-Bioagent-Oss-Prefix, framework mirrors the
# job dir (generated molecules) here. Requires the data-plane OSS bucket mounted
# at /mnt/oss RW in the FC console. Absent mount => no-op (NAS only). Default is
# already /mnt/oss; explicit for self-documentation. (No input direct-read: the
# request carries only scalar generation params, so there are no oss:// input
# URIs to rewrite and no `oss_mount: true` / uris.py is needed for this service.)
ENV SEMLAFLOW_OSS_OUTPUT_MOUNT=/mnt/oss
ENV SEMLAFLOW_JOBS_BASE_DIR=/data/semlaflow_jobs

# Weights + reference datasets externalized — NAS mount injected by FC config
# / SIF --bind.
ENV SEMLAFLOW_WEIGHTS_DIR=/data/models/semlaflow

RUN mkdir -p /data/semlaflow_jobs

# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance.
ENV SEMLAFLOW_SESSION_HEADER_NAME=bioagent-session-id
ENV SEMLAFLOW_MAX_CONCURRENT_JOBS=1

ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/semlaflow/bin/python", "-m", "uvicorn", \
     "server.app:app", "--host", "0.0.0.0", "--port", "9000", \
     "--timeout-keep-alive", "900"]
