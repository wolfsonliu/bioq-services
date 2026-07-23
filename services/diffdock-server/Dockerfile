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
#   builder = cuda-devel + conda env (Py 3.10 + pytorch 1.13.1 + cu117 + PyG 2.2
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

# nvcc is in cuda-devel; add gcc (openfold C++/CUDA ext build), wget, ca-certs,
# and **git** — uv needs the git CLI on PATH to resolve `openfold @ git+...`
# and `dllogger @ git+...` VCS specifiers (uv 0.11 no longer bundles libgit2).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential wget ca-certificates git \
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
# Conda env — MINIMAL, mirroring upstream DiffDock environment.yml structure.
#
# CRITICAL LESSON (2026-07-09): the entire torch / PyG / rdkit / numpy stack
# MUST be installed via PIP wheels (as upstream does), NOT conda.  An earlier
# revision put pytorch (pytorch channel) + pyg (pyg channel) + rdkit +
# numpy=1.24 (conda-forge) all in conda across 3 channels.  That produced a
# numpy C-API ABI mismatch: `torch.from_numpy(mol.GetConformer().GetPositions())`
# (datasets/process_mols.py:298) SEGFAULTED (rc=-11) because the pytorch-channel
# torch and conda-forge rdkit were built against different numpy ABIs than the
# forced numpy=1.24.  Upstream avoids this entirely by using the official pip
# torch wheel + PyG pip wheels + pip rdkit — all ABI-coherent against one numpy.
#
# Bonus: the official pip torch wheel is self-contained (bundles its own MKL),
# so the iJIT_NotifyEvent weak-symbol issue (which forced mkl<2024.1 + ittapi
# for CONDA pytorch 1.13) DOES NOT apply — those pins are removed.
#
# Conda holds only what upstream puts in conda: python, setuptools (openfold
# needs <70), prody + scipy (upstream comment: "ProDy needs conda").
#   - python=3.10 : bumped from upstream's 3.9 for framework Python>=3.10.
#   - setuptools=69.5.1 : openfold setup.py breaks on setuptools 70+.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n diffdock \
        -c conda-forge \
        python=3.10 \
        pip \
        "setuptools=69.5.1" \
        prody=2.2.0 \
        scipy=1.12.0

# ---------------------------------------------------------------------------
# Pip stack (Step A: wheels).  Mirrors upstream environment.yml's pip section
# exactly (torch cu117 wheel + PyG wheels + rdkit + pandas + sklearn + ...),
# on py3.10, with the esmfold deps handled deepspeed-free (see below).
#
# ABI coherence: torch, PyG native ext (cluster/scatter/sparse/spline-conv),
# rdkit, pandas, numpy all come from pip → built against one numpy 1.x ABI.
# `numpy<2` guards against numpy 2.0 (none of these old pkgs support it).
#
# ESMFold deps: we do NOT use the `fair-esm[esmfold]` extra — it hard-pins
# deepspeed==0.5.9 → triton==1.0.0 which has NO cp310 wheel (only cp36-39).
# Since framework[mcp] needs py>=3.10 we can't drop to 3.9, so install base
# fair-esm + the extra's real runtime deps manually (omegaconf / ml-collections
# / einops), skipping deepspeed.  ESMFold inference doesn't need deepspeed —
# openfold guards its deepspeed activation-checkpointing.  dm-tree (imports as
# `tree`) is an openfold runtime dep declared only in openfold's environment.yml
# (not setup.py), so pip won't pull it transitively — add explicitly.
# biopython is used by upstream (Bio.PDB / Bio.pairwise2) — add explicitly.
# ninja + wheel speed up the openfold CUDA-extension build in Step B.
# ---------------------------------------------------------------------------
# --index-strategy unsafe-best-match: torch/pyg/rdkit etc live on
# download.pytorch.org, but torchmetrics/pandas/sklearn/etc live on the
# Huawei PyPI mirror.  uv's default "first index that has the package wins"
# rejects e.g. torchmetrics==0.11.0 (pytorch index has torchmetrics but not
# that version → uv won't look further).  All our indexes are trusted
# (Huawei mirror of PyPI + official pytorch + pyg), so allow best-match
# across all of them.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/diffdock/bin/python \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu117 \
        --find-links https://data.pyg.org/whl/torch-1.13.1+cu117.html \
        "numpy<2" \
        "torch==1.13.1+cu117" \
        "torch-cluster==1.6.0+pt113cu117" \
        "torch-scatter==2.1.0+pt113cu117" \
        "torch-sparse==0.6.16+pt113cu117" \
        "torch-spline-conv==1.2.1+pt113cu117" \
        "torch-geometric==2.2.0" \
        "e3nn==0.5.1" \
        "fair-esm==2.0.0" \
        "networkx==2.8.4" \
        "pandas==1.5.1" \
        "pybind11==2.11.1" \
        "pytorch-lightning==1.9.5" \
        "rdkit==2022.03.3" \
        "scikit-learn==1.1.0" \
        "torchmetrics==0.11.0" \
        "biopython" \
        omegaconf \
        ml-collections \
        einops \
        dm-tree \
        pyyaml \
        requests \
        ninja \
        wheel

# ---------------------------------------------------------------------------
# Pip stack (Step B: git-built CUDA extensions).  openfold compiles CUDA
# kernels — needs nvcc + gcc (cuda-devel base) + setuptools<70 + torch present.
# `--no-build-isolation` reuses the Step-A torch (openfold's setup.py does
# `import torch` at top level; an isolated PEP 517 env has no torch →
# ModuleNotFoundError).  Both use git+ VCS specifiers → need the `git` CLI
# (apt-installed above).
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/diffdock/bin/python \
        --no-build-isolation \
        "openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307" \
        "dllogger @ git+https://github.com/NVIDIA/dllogger.git"

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
# Pre-compute SO(3) / torus LUT files.  Upstream utils/so3.py writes
# ``.so3_*.npy`` and utils/torus.py writes ``.p.npy`` + ``.score.npy`` —
# all CWD-relative, no path prefix.  precompute_series.py just imports both
# modules to trigger the else-branch that generates + saves them.
# Baking here into /opt/diffdock/ (runtime subprocess CWD points there)
# saves ~2 min on the first request.
# ---------------------------------------------------------------------------
WORKDIR /opt/diffdock
RUN /opt/conda/envs/diffdock/bin/python utils/precompute_series.py \
    && ls -la /opt/diffdock/.so3_omegas_array4.npy \
              /opt/diffdock/.so3_cdf_vals4.npy \
              /opt/diffdock/.p.npy \
              /opt/diffdock/.score.npy \
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
    && find /opt/conda/envs/diffdock/lib/python3.10/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/diffdock -depth -type d \
        -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/diffdock -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Shared service framework — independent layer so framework bumps don't
# invalidate the conda env layer above.
# ---------------------------------------------------------------------------
# COPY (not bind) so framework source changes bust this layer's cache — a bind mount's
# content is not part of the BuildKit layer cache key and would ship a stale framework
# (missing output-sink fixes).
COPY framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
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

# ---------------------------------------------------------------------------
# Runtime smoke: reproduce the script-dir sys.path scenario.  run_inference.py
# lives in /opt/diffdock/server/, which Python puts at sys.path[0] when the
# wrapper runs as a script.  server/ also has our own models.py, which would
# shadow upstream's `models/` package (`from models.aa_model import AAModel`
# in utils/utils.py) unless the wrapper front-loads /opt/diffdock.  This smoke
# simulates that exact path ordering and asserts `models` resolves to the
# upstream PACKAGE — catching the shadow at build time instead of on FC.
RUN /opt/conda/envs/diffdock/bin/python -c "\
import sys, os; \
sys.path.insert(0, '/opt/diffdock/server'); \
root='/opt/diffdock'; \
[sys.path.remove(root) for _ in range(sys.path.count(root))]; \
sys.path.insert(0, root); \
os.chdir(root); \
import models; \
assert hasattr(models, '__path__'), 'models shadowed by non-package: '+getattr(models,'__file__','?'); \
from models.aa_model import AAModel; \
from inference import get_parser; \
print('script-dir models/ shadow handled OK')"

# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance.
ENV DIFFDOCK_SESSION_HEADER_NAME=bioagent-session-id
ENV DIFFDOCK_MAX_CONCURRENT_JOBS=1
# Output-sink: when gateway sends X-Bioagent-Oss-Prefix, framework mirrors the job dir
# to this mount. FC console must mount the data-plane OSS bucket to /mnt/oss (RW).
# Missing mount => no-op (NAS only).
ENV DIFFDOCK_OSS_OUTPUT_MOUNT=/mnt/oss

ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/diffdock/bin/python", "-m", "uvicorn", \
     "server.app:app", "--host", "0.0.0.0", "--port", "9000", \
     "--timeout-keep-alive", "900"]
