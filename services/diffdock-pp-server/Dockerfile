# Build from bioagent project root:
#   docker build --platform linux/amd64 -t diffdock-pp-server \
#       -f services/diffdock-pp-server/Dockerfile .
#
# Pre-build (once; re-run to bump pinned upstream commit / refresh weights):
#   ./services/diffdock-pp-server/scripts/vendor.sh         # → upstream/
#   ./services/diffdock-pp-server/scripts/fetch_weights.sh  # → weights/ (score + confidence + ESM-2 + esm src)
#
# DiffDock-PP source lives at services/diffdock-pp-server/upstream/
# (vendored at pinned SHA via scripts/vendor.sh).  Weights (~22 MB checkpoints
# + ~2.5 GB ESM-2 + esm src) live on NAS at /data/models/diffdock-pp/ —
# NOT baked into the image (FC NAS mount; SIF / HPC `apptainer run --bind`).
# See engineering/decisions/2026-06-26-service-weights-externalization.md
# and engineering/decisions/2026-07-03-diffdock-pp-server-design.md.
#
# Multistage layout
# -----------------
#   builder = conda env (Python 3.10 + pytorch 1.13.0 + cu117 + PyG 2.2 +
#             e3nn + biopython + fair-esm + tensorboardX) + framework + cleanup.
#   runtime = lean: conda env from builder + DiffDock-PP source + server code.
#
# Base: nvidia/cuda 11.8 -runtime (forward-compatible with conda's
# pytorch-cuda=11.7; pytorch 1.13.0 has no >11.7 wheels).
# CUDA 11.8 driver back-compatible with cu117 binaries — verified on the
# diffusion-hopping-server build (same conda pin).

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
ENV PATH=/opt/conda/envs/diffdock_pp/bin:$PATH

# ---------------------------------------------------------------------------
# Mirror configuration (Huawei pip + TUNA conda) for CN-friendly builds.
# Same rationale as diffusion-hopping-server: strict channel_priority is
# essential for old pytorch/pyg resolves.
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
# custom_channels: only map channels that TUNA actually hosts.
# Do NOT map `pyg` or `nvidia` — TUNA returns 404 for those; they fall
# through to the upstream conda.anaconda.org URLs which work fine.
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# ---------------------------------------------------------------------------
# Conda env — matches upstream DiffDock-PP README dependencies (Python
# 3.10 + pytorch 1.13.0 + cu117 + PyG 2.2 + torch-scatter/sparse/cluster/
# spline-conv + e3nn + biopython + biopandas). Skip wandb (training-only)
# and matplotlib (visualization plots).
#
# iJIT_NotifyEvent fix (belt-and-braces): pytorch 1.13's libtorch_cpu.so
# weak-references `iJIT_NotifyEvent` (Intel VTune ITT).  Without a provider
# in the env's lib/, `import torch` blows up with `undefined symbol:
# iJIT_NotifyEvent`.  Historically `mkl` bundled the symbol, but mkl 2024.1+
# removed it — hence the failure on fresh builds.
#
# For pytorch 1.13 `ittapi` alone is INSUFFICIENT (verified 2026-07-03 on
# both diffusion-hopping-server and diffdock-pp-server builds) — the
# dynamic loader still reports the symbol undefined.  Pin `mkl<2024.1` /
# `intel-openmp<2024.1` so the mkl-provided collector still resolves the
# symbol.  Keep `ittapi` too — it's harmless and gives a second-source for
# pytorch 2.x rebuilds.
#
# See engineering/guides/pytorch-conda-ittapi.md and pytorch/pytorch#123097.
# ---------------------------------------------------------------------------
# pytorch pin note (belt-and-braces, verified 2026-07-05):
#
# - Upstream DiffDock-PP README uses `pytorch=1.13.0` + `pytorch-cuda=11.6`,
#   but conda `pyg` channel has thinner matrix coverage — pin `pytorch=1.13.1`
#   + `pytorch-cuda=11.7` (same as diffusion-hopping-server; 1.13.0→1.13.1 has
#   no API changes affecting DiffDock-PP source).
#
# - **Explicit CUDA build string** on the pytorch pin (`=py3.10_cuda11.7_cudnn8.5.0_0`)
#   is essential: with `mkl<2024.1` + `intel-openmp<2024.1` added for the ITT
#   fix, conda resolver otherwise **silently picks the CPU-only pytorch build**
#   (`=py3.10_cpu_0`) without warning. Symptom: `import torch` works but
#   `torch.version.cuda is None`, and `import torch_geometric` crashes with
#   `libc10_cuda.so: cannot open shared object file` because `pytorch-scatter`
#   / `pytorch-sparse` were built against the CUDA ABI.
#
# - Let `pyg::pyg=2.2.0` transitively pull compatible `pytorch-sparse` /
#   `pytorch-cluster` / `pytorch-spline-conv` versions — only pin
#   `pytorch-scatter=2.1.0` explicitly (matches upstream README's target).
#
# See engineering/guides/pytorch-conda-ittapi.md for the full trap chain
# (ITT symbol → mkl pin → CPU variant fallback) and the recommended
# build-time smoke.
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    micromamba create -y -n diffdock_pp \
        -c pyg -c nvidia -c pytorch -c conda-forge \
        python=3.10 \
        "pytorch::pytorch=1.13.1=py3.10_cuda11.7_cudnn8.5.0_0" \
        pytorch-cuda=11.7 \
        torchvision=0.14.1 \
        pyg::pyg=2.2.0 \
        pyg::pytorch-scatter=2.1.0 \
        conda-forge::e3nn=0.5.1 \
        conda-forge::biopython=1.80 \
        conda-forge::biopandas=0.4.1 \
        conda-forge::numpy=1.24 \
        conda-forge::pandas=1.5 \
        conda-forge::scikit-learn \
        conda-forge::scipy \
        conda-forge::pyyaml \
        conda-forge::dill \
        conda-forge::tqdm \
        conda-forge::tensorboardx \
        conda-forge::tensorboard \
        conda-forge::ittapi \
        "mkl<2024.1" \
        "intel-openmp<2024.1"

# fair-esm from PyPI (not on conda-forge). Deferred so the base env cache
# isn't invalidated when we bump only this.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/diffdock_pp/bin/python \
        fair-esm

# ---------------------------------------------------------------------------
# Build-time smoke: catch all three known conda-pytorch failure modes before
# the image ships (see engineering/guides/pytorch-conda-ittapi.md):
#   1. iJIT_NotifyEvent undefined symbol (mkl 2024.1+ dropped ITT collector)
#   2. CPU-only pytorch silently picked by resolver (missing libc10_cuda.so)
#   3. PyG native ops (scatter / sparse / cluster) ABI mismatch with pytorch
# All three would otherwise show up only at first inference request.
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/diffdock_pp/bin/python -c "\
import torch; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
assert torch.version.cuda is not None, 'pytorch built without CUDA (resolver picked CPU variant)'; \
import torch_geometric, torch_sparse, torch_scatter, torch_cluster; \
print('pyg / sparse / scatter / cluster OK')"

# ---------------------------------------------------------------------------
# DiffDock-PP source — from vendored upstream/ (see scripts/vendor.sh).
# Upstream has no setup.py; we import via PYTHONPATH at runtime instead of
# `pip install -e`.
#
# **Before this step you MUST run scripts/vendor.sh** — the COPY below
# does not fetch from github, it just copies from the host tree.  If the
# host `services/diffdock-pp-server/upstream/` is stale (e.g. missing
# `src/notebooks/utils_notebooks.py` which factory.py imports), the image
# ships broken.  vendor.sh has been updated 2026-07-05 to include
# notebooks/*.py — if you saw a v0.0.2/v0.0.3 build succeed, that only
# means the layer cache was hiding the problem.
# ---------------------------------------------------------------------------
COPY services/diffdock-pp-server/upstream /opt/diffdock-pp

# ---------------------------------------------------------------------------
# Fast smoke #1 (< 1s): verify the required source files landed.  Fails
# immediately if scripts/vendor.sh wasn't re-run with the notebooks-*.py
# fix.  Cheaper than waiting for the Python import smoke below to fail
# after 3 minutes of torch startup.
# ---------------------------------------------------------------------------
RUN set -e; \
    for f in /opt/diffdock-pp/src/main_inf.py \
             /opt/diffdock-pp/src/sample.py \
             /opt/diffdock-pp/src/model/factory.py \
             /opt/diffdock-pp/src/model/__init__.py \
             /opt/diffdock-pp/src/data/data.py \
             /opt/diffdock-pp/src/filtering/dataset.py \
             /opt/diffdock-pp/src/evaluation/compute_rmsd.py \
             /opt/diffdock-pp/src/geom_utils/transform.py \
             /opt/diffdock-pp/src/notebooks/utils_notebooks.py; do \
        [ -f "$f" ] || { \
            echo "ERROR: required upstream file missing: $f"; \
            echo "Did you run ./services/diffdock-pp-server/scripts/vendor.sh?"; \
            exit 1; \
        }; \
    done; \
    echo 'vendor filesystem check OK'

# ---------------------------------------------------------------------------
# Smoke #2: exercise the FULL upstream module chain that inference.py
# will eventually load.  We inject `wandb` + `matplotlib` stubs the same
# way inference.py does at runtime — the smoke must exactly mirror the
# runtime import environment.
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/diffdock_pp/bin/python -c "\
import sys, types; \
sys.path.insert(0, '/opt/diffdock-pp/src'); \
sys.modules.setdefault('wandb', types.ModuleType('wandb')); \
mpl = sys.modules.setdefault('matplotlib', types.ModuleType('matplotlib')); \
plt = sys.modules.setdefault('matplotlib.pyplot', types.ModuleType('matplotlib.pyplot')); \
mpl.pyplot = plt; \
from geom_utils import set_time; \
from model import load_model, to_cuda; \
from data import load_data, get_data, BindingDataset; \
from filtering.dataset import get_confidence_loader; \
from evaluation.compute_rmsd import evaluate_all_rmsds; \
from sample import sample; \
import main_inf; \
print('diffdock_pp upstream module chain OK')"

# ---------------------------------------------------------------------------
# Cleanup dead weight from the conda env (mirrors diffusion-hopping cleanup).
# ---------------------------------------------------------------------------
RUN /opt/conda/envs/diffdock_pp/bin/pip uninstall -y \
        jupyterlab jupyter_server jupyterlab_server jupyter_client \
        ipywidgets ipykernel notebook nbconvert nbformat || true \
    && find /opt/conda/envs/diffdock_pp/lib -name '*.a' -delete \
    && rm -rf \
        /opt/conda/envs/diffdock_pp/share/doc \
        /opt/conda/envs/diffdock_pp/share/man \
        /opt/conda/envs/diffdock_pp/share/locale \
    && find /opt/conda/envs/diffdock_pp/lib/python3.10/site-packages \
        -depth -type d \( -name 'tests' -o -name 'test' \) \
        -exec rm -rf {} + \
    && find /opt/conda/envs/diffdock_pp -depth -type d \
        -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/diffdock_pp -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Shared service framework — independent layer so framework bumps don't
# invalidate the conda env layer above.
#
# COPY (not bind-mount): BuildKit does not include bind-mount content in the
# layer cache key, so a bind would ship a stale framework even after the
# framework source is updated (would silently drop the output-sink fix).
# ---------------------------------------------------------------------------
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/diffdock_pp/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2


# ===========================================================================
# Stage 2: runtime — conda env + DiffDock-PP source + server code.
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/diffdock_pp/bin:$PATH

# FC containers ship with LANG unset → Python's locale.getpreferredencoding()
# returns 'ascii', which makes `open(path)` (no encoding kwarg) fail on any
# UTF-8 byte.  Upstream `src/args.py:process_args` does `open(args.config_file)`
# without encoding, so if the bundled yaml has any UTF-8 char, the second
# yaml read (from our _load_visualization_values postprocess step) crashes
# with `UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2 ...`.
# Setting LANG closes this class of bug for every `open()` in the container.
# Verified 2026-07-06 on v0.0.6 dock run.
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Conda env from builder.
COPY --from=builder /opt/conda /opt/conda

# DiffDock-PP source from builder (vendored upstream).
COPY --from=builder /opt/diffdock-pp /opt/diffdock-pp

# Server code as `server` package — uvicorn imports `server.app:app`.
# Bundled config yaml lands next to inference.py so DIFFDOCK_PP_CONFIG_YAML
# default resolves without extra COPY commands.
COPY services/diffdock-pp-server/__init__.py \
     services/diffdock-pp-server/__main__.py \
     services/diffdock-pp-server/VERSION \
     services/diffdock-pp-server/app.py \
     services/diffdock-pp-server/adapter.py \
     services/diffdock-pp-server/models.py \
     services/diffdock-pp-server/settings.py \
     services/diffdock-pp-server/tools.py \
     services/diffdock-pp-server/inference.py \
     services/diffdock-pp-server/single_pair_inference.yaml \
     /opt/diffdock-pp/server/

WORKDIR /opt/diffdock-pp

# PYTHONPATH — upstream repo root (for `server` package) AND src/ (upstream
# uses flat imports like `from args import parse_args` — src/ must be on
# sys.path).  Apptainer/SIF ignores WORKDIR, so PYTHONPATH must contain both
# explicitly.
ENV PYTHONPATH=/opt/diffdock-pp:/opt/diffdock-pp/src
ENV DIFFDOCK_PP_ROOT=/opt/diffdock-pp
ENV DIFFDOCK_PP_PYTHON=/opt/conda/envs/diffdock_pp/bin/python
ENV DIFFDOCK_PP_INFERENCE_SCRIPT=/opt/diffdock-pp/server/inference.py
ENV DIFFDOCK_PP_CONFIG_YAML=/opt/diffdock-pp/server/single_pair_inference.yaml
ENV DIFFDOCK_PP_JOBS_BASE_DIR=/data/diffdock_pp_jobs

# Output-sink: when the gateway sends X-Bioagent-Oss-Prefix, the framework
# mirrors the finished job dir to this mount. Requires an OSS mount at /mnt/oss
# in the FC console (storage config). Absent mount => no-op (NAS-only).
ENV DIFFDOCK_PP_OSS_OUTPUT_MOUNT=/mnt/oss

# Weights externalized — NAS mount injected by FC config / SIF --bind.
ENV DIFFDOCK_PP_WEIGHTS_DIR=/data/models/diffdock-pp

# Point torch.hub + HuggingFace caches at the NAS-mounted esm_cache dir so
# `torch.hub.load("facebookresearch/esm:main", ...)` runs offline.  These
# echo the settings.torchhub_dir default; also set by inference.py at
# runtime as belt-and-suspenders.
ENV TORCH_HOME=/data/models/diffdock-pp/esm_cache
ENV HF_HOME=/data/models/diffdock-pp/esm_cache/hf_cache

RUN mkdir -p /data/diffdock_pp_jobs

# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance.
ENV DIFFDOCK_PP_SESSION_HEADER_NAME=bioagent-session-id
ENV DIFFDOCK_PP_MAX_CONCURRENT_JOBS=1

ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/diffdock_pp/bin/python", "-m", "uvicorn", \
     "server.app:app", "--host", "0.0.0.0", "--port", "9000", \
     "--timeout-keep-alive", "900"]
