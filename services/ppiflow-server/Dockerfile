# Build from bioagent project root:
#   docker build --platform linux/amd64 -t ppiflow-server -f services/ppiflow-server/Dockerfile .
#
# PPIFlow source lives at services/ppiflow-server/upstream/ (vendored at
# pinned SHA via scripts/vendor.sh).  Weights are pre-staged at
# services/ppiflow-server/checkpoint/ — not in the upstream repo — and
# baked into the image.
# This image only supports structure generation (sample_*.py). Sequence design,
# scoring, packing, and Rosetta steps are NOT included — they belong in
# separate bioagent services.
#
# Base: cudnn-DEVEL variant of nvidia/cuda 12.4.1. The 2026-05-14 attempt to use
# `-runtime` failed at runtime with:
#
#     fatal error: cuda_runtime_api.h: No such file or directory
#     ninja: build stopped: subcommand failed.
#     RuntimeError: Error building extension 'fastfold_layer_norm_cuda'
#
# `PPIFlow/tool/PPIFlow/models/layer_norm/layer_norm.py` JIT-compiles a custom
# CUDA layer-norm kernel via `torch.utils.cpp_extension.load` on first import
# (triggered by every `sample_*.py` script when the model is constructed). The
# JIT path requires `nvcc` + `cuda_runtime_api.h` from the CUDA Toolkit, which
# only ship in `-devel`. The 2026-05-14 commit comment "PPIFlow's inference
# path doesn't trigger JIT CUDA compilation" was based on inspecting the
# env.yml install chain only and missed the at-import-time compile.
#
# We pre-compile the .so during image build (see the RUN below COPY of PPIFlow
# source) so runtime cold starts skip the ~30s JIT step.
#
# Single-stage build (no multistage). PPIFlow needs devel for the pre-compile
# step AND at runtime for any future JIT fallback. Splitting stages would just
# copy back the same devel base — no size win.
#
# Layer ordering (stable → volatile):
#   apt/micromamba/uv → mirrors → conda env → source + checkpoints →
#   pre-compile → framework → server .py
#
# Size cost: devel = 6.7 GB vs runtime = 3.1 GB (~3.6 GB extra). Total image
# stays well under the FC 15 GB limit.
# Pre-build (once; re-run to bump pinned upstream commit):
#   ./services/ppiflow-server/scripts/vendor.sh

FROM mambaorg/micromamba:2.6.2 AS micromamba
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip mirror (project-wide convention). UV_INDEX_URL controls the
# uv pip layer below; pip.conf (further down) controls the pip invocations
# micromamba makes while resolving env.yml's pip: section.
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600

# ---------------------------------------------------------------------------
# Pin torch's CUDA discovery to the system toolkit, NOT conda's.
#
# PPIFlow's env.yml installs `cuda-nvcc=12.1.66` from the nvidia conda channel.
# That puts nvcc at /opt/conda/envs/ppiflow/bin/nvcc, and since PATH prepends
# the conda env (see ENV PATH below), torch's `_find_cuda_home()` does
# `which nvcc` → finds conda's nvcc → sets CUDA_HOME=/opt/conda/envs/ppiflow.
# But that prefix has nvcc only; the CUDA headers (cuda_runtime_api.h etc.)
# live in /usr/local/cuda/include/ from the -devel base image.
#
# Result without this env: JIT compile crashes with
#   fatal error: cuda_runtime_api.h: No such file or directory
# even though we're on the -devel base.
#
# Torch checks CUDA_HOME / CUDA_PATH BEFORE walking PATH, so setting these to
# /usr/local/cuda short-circuits the conda lookup and gives torch the matching
# nvcc + headers from the same 12.4 toolkit.
# ---------------------------------------------------------------------------
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda

# ---------------------------------------------------------------------------
# micromamba (from pinned image) + uv + apt build tools. The conda env stays
# conda-managed because PPIFlow's env.yml leans on bioconda-only packages
# (pymol-open-source, pyqt5, cuda-nvcc) that uv/PyPI can't supply.
# uv is brought in for the final framework-install layer only.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

ENV MAMBA_EXE=/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/ppiflow/bin:$PATH

# ---------------------------------------------------------------------------
# Configure pip to use the Huawei Cloud mirror **before** micromamba runs
# `env create` (which itself invokes pip for the pip: section of env.yml).
# Without this, CN build hosts hit flaky pypi.org and get "(from versions: none)"
# errors on random packages — we observed this on aiohappyeyeballs==2.5.0 even
# though absl-py from the same install batch succeeded.
#
# Huawei Cloud is the project-wide pip mirror (matches genie3-server and
# rfantibody-server). The pytorch wheel index (download.pytorch.org) stays as
# the explicit --extra-index-url inside env.yml's pip: section — we don't
# override that.
#
# ~/.pydistutils.cfg: some pip: packages (tmtools==0.2.0) use the deprecated
# setuptools `setup_requires` → `fetch_build_eggs` path, which bypasses
# pip.conf and downloads from upstream pypi.org directly. The distutils config
# redirects easy_install (used by fetch_build_eggs) to the Huawei mirror.
# ---------------------------------------------------------------------------
RUN mkdir -p /root/.config/pip && \
    printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
    > /root/.config/pip/pip.conf && \
    printf '[easy_install]\nindex_url = https://repo.huaweicloud.com/repository/pypi/simple/\n' \
    > /root/.pydistutils.cfg

# ---------------------------------------------------------------------------
# Redirect conda channels to the shared mirror mapping via `.condarc`
# (deploy/conda/mirrors.condarc — single source of truth). The upstream
# channels nvidia / schrodinger / ostrokach / anaconda are intentionally
# NOT remapped: they fall through to conda.anaconda.org (slow, but works).
# ---------------------------------------------------------------------------
COPY deploy/conda/mirrors.condarc /tmp/mirrors.condarc
RUN cat > /root/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
EOF
RUN cat /tmp/mirrors.condarc >> /root/.condarc

# ---------------------------------------------------------------------------
# Create the `ppiflow` conda env from upstream's environment.yml.
#
# BuildKit cache mounts (the big optimization):
#   - /opt/conda/pkgs        — conda package cache (extracted tarballs).
#                              Saves ~1–2 GB re-download on env.yml-changing
#                              rebuilds.
#   - /root/.cache/pip       — pip wheel cache for the env.yml `pip:` section.
#                              The ~2 GB torch wheels from download.pytorch.org
#                              (no China CDN — ~10 min download) get cached
#                              across rebuilds. First build is unavoidably
#                              slow; rebuilds drop from ~25 min to ~3 min.
#
# We DELIBERATELY DROP `micromamba clean --all`. The cache-mounted directories
# are excluded from the image layer by BuildKit, so cleaning inside the RUN
# would only purge the persistent cache — defeating the optimization. Image
# size is unaffected: only `/opt/conda/envs/ppiflow/` (the realized env, ~10 GB)
# ends up in the layer; `pkgs/` and `~/.cache/pip` do not.
#
# Cache-friendly layer ordering: env.yml change invalidates this layer, but
# the cache mount still hits → most package downloads are re-used from cache,
# only newly-added packages get fetched.
# ---------------------------------------------------------------------------
WORKDIR /opt/ppiflow

# ---------------------------------------------------------------------------
# PPIFlow source — from vendored upstream/ (see scripts/vendor.sh).  env.yml
# and the tool/PPIFlow tree are both consumed from this tree.
# ---------------------------------------------------------------------------
COPY services/ppiflow-server/upstream /opt/ppiflow-src

# ---------------------------------------------------------------------------
# Patch upstream's environment.yml before `env create`.
#
# The upstream `channels:` list hardcodes Tsinghua (TUNA) mirror URLs:
#
#     - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
#     - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
#     - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
#
# These sit inside `-f environment.yml`, whose `channels:` take precedence over
# `/root/.condarc` — so the PKU mapping from deploy/conda/mirrors.condarc is
# bypassed entirely. TUNA has removed the legacy `pkgs/free` channel (returns
# 404 today), and libmamba 2.x treats a 404 repodata.json as a fatal error,
# aborting `env create` ("Multiple errors occurred").
#
# Strip every TUNA line; the remaining channels (defaults / conda-forge /
# pytorch / bioconda) then resolve through mirrors.condarc → PKU, while
# nvidia / schrodinger / ostrokach / anaconda fall through to conda.anaconda.org
# as documented above.
# ---------------------------------------------------------------------------
RUN sed -i '/mirrors\.tuna\.tsinghua\.edu\.cn/d' \
    /opt/ppiflow-src/tool/PPIFlow/environment.yml

RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    micromamba env create -y -n ppiflow \
        -f /opt/ppiflow-src/tool/PPIFlow/environment.yml

# ---------------------------------------------------------------------------
# Move the PPIFlow tool source into place + COPY pretrained checkpoints.
# The four .ckpt files baked into the image are heavy (~100s MB each); make
# sure services/ppiflow-server/checkpoint/ is populated before building.
# ---------------------------------------------------------------------------
RUN cp -a /opt/ppiflow-src/tool/PPIFlow/. /opt/ppiflow/ \
    && rm -rf /opt/ppiflow-src
# Checkpoints live on NAS at /data/models/ppiflow/checkpoint/ — NOT baked
# into the image (~1.1 GB savings).  FC mounts the NAS; SIF / HPC bind:
#   apptainer run --bind /scratch/models/ppiflow:/data/models/ppiflow \
#       ppiflow-server.sif python -m server sample ...
# Pre-stage to local, then rsync to NAS / HPC.
# See engineering/decisions/2026-06-26-service-weights-externalization.md.

# ---------------------------------------------------------------------------
# Pre-compile the fastfold_layer_norm_cuda extension during image build.
#
# Upstream PPIFlow ships a stale `.so` in models/layer_norm/ that was built
# against an older torch (undefined symbol _ZN3c104cuda29c10_cuda_check_...).
# At runtime, `from models.layer_norm.layer_norm import FusedLayerNorm`:
#   1. Tries `importlib.import_module("fastfold_layer_norm_cuda")` → ImportError
#      (ABI mismatch with this env's torch 2.3.1+cu121)
#   2. Falls back to `torch.utils.cpp_extension.load(...)` → invokes nvcc to
#      rebuild the .so. This is the step that needs the devel base image.
#
# Doing this at build time has two benefits:
#   - Validates the CUDA toolchain BEFORE the image ships (build-time fail-fast
#     vs runtime job failure)
#   - Cold-start no longer pays the ~30s JIT compile cost (the .so is in the
#     image layer at /opt/ppiflow/models/layer_norm/)
#
# `TORCH_CUDA_ARCH_LIST` in torch_ext_compile.py limits arches to 7.0+8.0,
# but the file also passes `-gencode arch=compute_86 + 90` to nvcc directly.
# So the resulting .so covers V100 (7.0), A100 (8.0), L4/L40S (8.6), H100 (9.0).
# We rm the stale .so first to make sure the fallback path runs cleanly.
# ---------------------------------------------------------------------------
RUN rm -f /opt/ppiflow/models/layer_norm/fastfold_layer_norm_cuda.so \
          /opt/ppiflow/models/layer_norm/layer_norm_cuda.o \
          /opt/ppiflow/models/layer_norm/layer_norm_cuda_kernel.cuda.o \
          /opt/ppiflow/models/layer_norm/build.ninja && \
    cd /opt/ppiflow && /opt/conda/envs/ppiflow/bin/python -c \
        "from models.layer_norm.layer_norm import FusedLayerNorm; print('layer_norm pre-compile OK')"

# ---------------------------------------------------------------------------
# Shared bioagent service framework + remote-fetch deps in the same env.
# Keeping this as a late layer so the framework re-install doesn't invalidate
# the (slow) conda env build cache. uv pip resolves and installs into the
# conda env's interpreter — faster than conda env's bundled pip and benefits
# from Huawei mirror (UV_INDEX_URL set above).
# ---------------------------------------------------------------------------
COPY framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/ppiflow/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# Server code last — explicit file list so test changes don't bust cache.
COPY services/ppiflow-server/__init__.py \
     services/ppiflow-server/__main__.py \
     services/ppiflow-server/VERSION \
     services/ppiflow-server/app.py \
     services/ppiflow-server/adapter.py \
     services/ppiflow-server/models.py \
     services/ppiflow-server/settings.py \
     services/ppiflow-server/tools.py \
     /opt/ppiflow/server/

# Settings via env (see services/ppiflow-server/settings.py).
ENV PYTHONPATH=/opt/ppiflow
ENV PPIFLOW_ROOT=/opt/ppiflow
ENV PPIFLOW_CKPT_DIR=/data/models/ppiflow/checkpoint
ENV PPIFLOW_CONFIG_DIR=/opt/ppiflow/configs
ENV PPIFLOW_JOBS_BASE_DIR=/data/ppiflow_jobs
RUN mkdir -p /data/ppiflow_jobs

# Output-sink: 当 gateway 传 X-Bioagent-Oss-Prefix 时，framework 把 job dir 镜像到此挂载点。
# 需在 FC 控制台把数据面 OSS bucket 挂到 /mnt/oss。缺挂载 => no-op（仅 NAS）。
ENV PPIFLOW_OSS_OUTPUT_MOUNT=/mnt/oss

# Alibaba Cloud FC plumbing.

# Keepalive pings this URL through FC's gateway so the instance stays alive
# during long GPU jobs. Override at deploy time if the URL changes.
# ENV PPIFLOW_KEEPALIVE_URL=https://fc-ppiflow-lufflhmlaw.cn-hangzhou-vpc.fcapp.run
# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance. Requires FC console config too.
ENV PPIFLOW_SESSION_HEADER_NAME=bioagent-session-id
ENV PPIFLOW_MAX_CONCURRENT_JOBS=1
ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/ppiflow/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
