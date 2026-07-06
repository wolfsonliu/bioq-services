# ---------- DiffDock Server ----------
# Small-molecule blind docking with DiffDock-L (Corso et al., ICLR 2024).
# LICENSE: MIT (upstream https://github.com/gcorso/DiffDock).
#
# Build (from bioagent project root):
#   docker build --platform linux/amd64 -t diffdock-server \
#       -f services/diffdock-server/Dockerfile .
#
# Prerequisites (one-time, re-run to bump pinned upstream commit / refresh weights):
#   ./services/diffdock-server/scripts/vendor.sh          # → upstream/
#   ./services/diffdock-server/scripts/fetch_weights.sh   # → weights/ NAS
#
# DiffDock source lives at services/diffdock-server/upstream/ (vendored at
# pinned SHA via scripts/vendor.sh).  Weights (~1.2 GB score + confidence +
# ~2.5 GB ESM-2 + ~5 GB ESMFold) live on NAS at /data/models/diffdock/ —
# NOT baked into the image (FC NAS mount; SIF / HPC `apptainer run --bind`).
# See engineering/decisions/2026-06-26-service-weights-externalization.md
# and engineering/decisions/2026-07-06-diffdock-server-design.md.
#
# Multistage layout
# -----------------
#   builder = cuda-devel + conda env (Py 3.9 + pytorch 1.13.1 + cu117 + PyG 2.2
#             + e3nn + fair-esm[esmfold] + openfold git build) + LUT precompute
#             + framework + cleanup.
#   runtime = lean: cuda-runtime + conda env from builder + DiffDock source
#             + server code + precomputed `.so3_*.npy` / `.torus_*.npy` LUTs.
#
# Base: nvidia/cuda 11.8.0 (both devel builder and runtime).  pytorch
# 1.13.1+cu117 wheels have cu117 as CUDA runtime, but CUDA 11.8 is
# fully backward-compatible (11.7 wheels run on 11.8 driver + libs).
# Aligns with diffdock-pp-server / diffusion-hopping-server / drughive-server —
# all four pytorch-1.13 conda services share the same 11.8 base.
#
# openfold pip build: nvcc 11.8 backward-emits CUDA 11.7 ABI-compatible
# object code (verified 2026-07-06 in diffdock-pp-server smoke).
# 12.x devel bases are NOT safe here: pytorch 1.13.1+cu117 statically
# links cudart 11.7 + cudnn 8.5, but base 12.x cudnn 9+ would ABI-conflict
# with pytorch's bundled shared libs at runtime.

ARG BASE_DEVEL=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG BASE_RUNTIME=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ARG MICROMAMBA_IMAGE=mambaorg/micromamba:2.6.2


# ===========================================================================
# Stage 1: builder — conda env + openfold build + LUT precompute + framework.
# ===========================================================================
FROM ${MICROMAMBA_IMAGE} AS micromamba
FROM ${BASE_DEVEL} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

# nvcc is in cuda-devel; add gcc (openfold C++/CUDA ext build), wget, ca-certs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/
ENV MAMBA_EXE=/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/diffdock/bin:$PATH

# ---------------------------------------------------------------------------
# Mirror configuration (Huawei pip + TUNA conda) for CN-friendly builds.
# Strict channel priority — essential for old pytorch/pyg resolves.
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
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# ---------------------------------------------------------------------------
# Conda env — matches upstream DiffDock environment.yml.
#
# CRITICAL pins:
#   - python=3.9.18                : upstream pin (openfold + setuptools work)
#   - pytorch=1.13.1 + cu117       : upstream pin (also matches diffdock-pp)
#   - CUDA build string explicit   : otherwise resolver silently picks CPU
#                                    build under mkl<2024.1 (verified on
#                                    diffdock-pp-server 2026-07-05)
#   - setuptools=69.5.1            : openfold setup.py hits ImportError with
#                                    setuptools 70+ (which removed several
#                                    legacy compat modules)
#   - mkl<2024.1, intel-openmp<2024.1, ittapi : iJIT_NotifyEvent weak symbol
#                                    (pytorch 1.13 libtorch_cpu.so).  See
#                                    engineering/guides/pytorch-conda-ittapi.md.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n diffdock \
        -c pyg -c nvidia -c pytorch -c conda-forge \
        python=3.9.18 \
        "pytorch::pytorch=1.13.1=py3.9_cuda11.7_cudnn8.5.0_0" \
        pytorch-cuda=11.7 \
        torchvision=0.14.1 \
        pyg::pyg=2.2.0 \
        pyg::pytorch-scatter=2.1.0 \
        conda-forge::e3nn=0.5.1 \
        conda-forge::prody=2.2.0 \
        conda-forge::biopython \
        conda-forge::rdkit=2022.03.3 \
        conda-forge::numpy=1.24 \
        conda-forge::pandas=1.5 \
        conda-forge::scikit-learn \
        conda-forge::scipy=1.12.0 \
        conda-forge::pyyaml \
        conda-forge::networkx=2.8.4 \
        conda-forge::tqdm \
        conda-forge::pybind11=2.11.1 \
        conda-forge::ittapi \
        "mkl<2024.1" \
        "intel-openmp<2024.1" \
        "setuptools=69.5.1"

# fair-esm[esmfold] pulls openfold + ml-collections + einops. openfold is a
# git-installed package that compiles CUDA extensions — needs nvcc + gcc
# (present in cuda-devel base) + setuptools=69.5.1 (already pinned above).
# Compile time ~5 min; extension size ~500 MB.
#
# pytorch-lightning=1.9.5, torchmetrics=0.11.0 come from upstream env yaml.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/diffdock/bin/python \
        "pytorch-lightning==1.9.5" \
        "torchmetrics==0.11.0" \
        "fair-esm[esmfold]==2.0.0" \
        "openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307" \
        "dllogger @ git+https://github.com/NVIDIA/dllogger.git" \
        requests

# ---------------------------------------------------------------------------
# Build-time smoke #1: CUDA torch + PyG native ops OK.  Catches the CPU-only
# fallback + PyG ABI mismatch failures that only surface at first inference.
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/diffdock/bin/python -c "\
import torch; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
assert torch.version.cuda is not None, 'pytorch built without CUDA (resolver picked CPU variant)'; \
import torch_geometric, torch_sparse, torch_scatter, torch_cluster; \
print('pyg / sparse / scatter / cluster OK')"

# Build-time smoke #2: fair-esm + esmfold import path (openfold underneath).
RUN /opt/conda/envs/diffdock/bin/python -c "\
from esm import pretrained; \
import esm.esmfold.v1.esmfold as esmfold; \
print('esm + esmfold import OK')"

# ---------------------------------------------------------------------------
# DiffDock source — from vendored upstream/ (see scripts/vendor.sh).
# Upstream has no setup.py; PYTHONPATH at runtime instead of `pip install -e`.
# ---------------------------------------------------------------------------
COPY services/diffdock-server/upstream /opt/diffdock

# Fast smoke: required upstream files landed.
RUN set -e; \
    for f in /opt/diffdock/inference.py \
             /opt/diffdock/utils/precompute_series.py \
             /opt/diffdock/utils/so3.py \
             /opt/diffdock/utils/torus.py \
             /opt/diffdock/utils/inference_utils.py \
             /opt/diffdock/default_inference_args.yaml; do \
        [ -f "$f" ] || { \
            echo "ERROR: required upstream file missing: $f"; \
            echo "Did you run ./services/diffdock-server/scripts/vendor.sh?"; \
            exit 1; \
        }; \
    done; \
    echo 'vendor filesystem check OK'

# ---------------------------------------------------------------------------
# Pre-compute SO(3) / torus LUT files.  Upstream utils/so3.py + utils/torus.py
# call ``np.load('.so3_omegas_array4.npy')`` — CWD-relative, no path prefix.
# Running the script here bakes the `.npy` files into /opt/diffdock/; runtime
# subprocess CWD points there.  ~2 min build cost saved on every request.
# ---------------------------------------------------------------------------
WORKDIR /opt/diffdock
RUN /opt/conda/envs/diffdock/bin/python utils/precompute_series.py \
    && ls -la /opt/diffdock/.so3_*.npy /opt/diffdock/.torus_*.npy 2>/dev/null \
    && echo 'SO(3) / torus LUT precompute OK'

# ---------------------------------------------------------------------------
# Build-time smoke #3: exercise the full upstream inference module chain
# (torch + rdkit + prody + fair-esm + upstream utils).  Catches missing
# transitives / import errors before FC deploy.
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/diffdock/bin/python -c "\
import sys; sys.path.insert(0, '/opt/diffdock'); \
from inference import get_parser; \
from utils.inference_utils import InferenceDataset; \
from utils.sampling import sampling; \
from utils.utils import get_model; \
from datasets.process_mols import write_mol_with_coords; \
print('diffdock upstream module chain OK')"

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env.
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/diffdock/bin/pip uninstall -y \
        jupyterlab jupyter_server jupyterlab_server jupyter_client \
        ipywidgets ipykernel notebook nbconvert nbformat || true \
    && find /opt/conda/envs/diffdock/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/diffdock/share/doc \
        /opt/conda/envs/diffdock/share/man \
        /opt/conda/envs/diffdock/share/locale \
    && find /opt/conda/envs/diffdock/lib/python3.9/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/diffdock -depth -type d \
        -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/diffdock -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Shared service framework — independent layer so framework bumps don't
# invalidate the conda env layer above.
# ---------------------------------------------------------------------------
RUN --mount=type=bind,source=services/_framework,target=/tmp/service-framework \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/diffdock/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2


# ===========================================================================
# Stage 2: runtime — cuda-runtime + conda env + DiffDock source + server code.
# ===========================================================================
FROM ${BASE_RUNTIME}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/diffdock/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Conda env from builder.
COPY --from=builder /opt/conda /opt/conda

# DiffDock source from builder (vendored upstream + pre-computed LUT `.npy`).
COPY --from=builder /opt/diffdock /opt/diffdock

# Server code as `server` package — uvicorn imports `server.app:app`.
# Bundled default_inference_args.yaml lands next to run_inference.py so
# DIFFDOCK_CONFIG_YAML default resolves without extra COPY commands.
COPY services/diffdock-server/__init__.py \
     services/diffdock-server/__main__.py \
     services/diffdock-server/VERSION \
     services/diffdock-server/app.py \
     services/diffdock-server/adapter.py \
     services/diffdock-server/models.py \
     services/diffdock-server/settings.py \
     services/diffdock-server/tools.py \
     services/diffdock-server/uris.py \
     services/diffdock-server/run_inference.py \
     services/diffdock-server/default_inference_args.yaml \
     /opt/diffdock/server/

WORKDIR /opt/diffdock

# PYTHONPATH — upstream root (for `server` package + upstream flat imports
# like `from inference import ...`, `from utils.xxx import ...`).
ENV PYTHONPATH=/opt/diffdock
ENV DIFFDOCK_ROOT=/opt/diffdock
ENV DIFFDOCK_PYTHON=/opt/conda/envs/diffdock/bin/python
ENV DIFFDOCK_INFERENCE_SCRIPT=/opt/diffdock/server/run_inference.py
ENV DIFFDOCK_CONFIG_YAML=/opt/diffdock/server/default_inference_args.yaml
ENV DIFFDOCK_JOBS_BASE_DIR=/data/diffdock_jobs

# Weights externalized — NAS mount injected by FC config / SIF --bind.
ENV DIFFDOCK_WEIGHTS_DIR=/data/models/diffdock

# torch.hub read of ESM-2 / ESMFold weights.  Also set by run_inference.py
# at runtime as belt-and-suspenders; declared here so a raw `python -m server
# dock` from sbatch inherits it.
ENV TORCH_HOME=/data/models/diffdock/esm_cache
# Disable upstream's release-zip fallback download (offline FC environment).
ENV REPOSITORY_URL=file:///dev/null

RUN mkdir -p /data/diffdock_jobs

# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance.
ENV DIFFDOCK_SESSION_HEADER_NAME=bioagent-session-id
ENV DIFFDOCK_MAX_CONCURRENT_JOBS=1

ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/diffdock/bin/python", "-m", "uvicorn", \
     "server.app:app", "--host", "0.0.0.0", "--port", "9000", \
     "--timeout-keep-alive", "900"]
