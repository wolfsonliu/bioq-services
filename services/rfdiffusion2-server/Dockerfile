# Build from bioagent project root:
#   docker build --platform linux/amd64 -t rfdiffusion2-server -f services/rfdiffusion2-server/Dockerfile .
#
# RFdiffusion2 source must be at opensource/RFdiffusion2/ (weights baked into image).
# Run `cd opensource/RFdiffusion2 && python setup.py` first to download the weights.
#
# Multistage layout
# -----------------
#   source-prep = trims the RFdiffusion2 source tree: drops `test_data/`,
#                 `goldens/`, `benchmark/rotamer_library/`, `dev/`, top-level
#                 docs/notebooks/images/assets, and the apptainer .sif files
#                 we don't use. About 350 MB of dead weight.
#   builder     = heavy: bzip2/curl/wget/git + micromamba + conda env create
#                 (~10 GB env including pyrosetta, torch+cu124, dgl, openbabel).
#                 Final RUN strips `__pycache__` / `*.pyc` from the env.
#   runtime     = lean : only build-essential + ca-certificates. The finished
#                 conda env + trimmed source + framework are copied over.
#
# Why two stages on top of source-prep: bzip2/curl/wget/git/micromamba aren't
# needed once the env is built, but in a single Dockerfile `apt-get remove`
# after install can't reclaim the bytes from the install layer. Multistage is
# the only way to leave those packages behind. Same playbook as
# rfdiffusion-server (see comments there).
#
# Base: CUDA 12.4 -devel (REQUIRED, do not switch to -runtime). rf2aa imports
# ipd.dev.cuda.cudabuild which calls `torch.utils.cpp_extension.load(...)` —
# JIT-compiles a CUDA kernel on first model construction. That path needs
# nvcc + cuda_runtime_api.h, only present in -devel. We also keep
# build-essential in the runtime stage so the JIT compile succeeds.
# Same lesson as ppiflow-server (see engineering/guides/uv-dockerfile-patterns.md).
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04


# ===========================================================================
# Stage 1: source-prep — trim the RFdiffusion2 source tree.
#
# Use the same CUDA base so we avoid pulling an extra image. The contents of
# this stage never reach the final image; only the trimmed /src/ at the end
# does, via COPY --from below.
# ===========================================================================
FROM ${BASE_IMAGE} AS source-prep

COPY opensource/RFdiffusion2 /src/

# Drop runtime-irrelevant data:
#   rf_diffusion/test_data/                 ~118 MB — referenced only by tests
#   rf_diffusion/goldens/                    ~97 MB — test fixtures
#   rf_diffusion/benchmark/rotamer_library/ ~100 MB — used by pipeline.py only
#   rf_diffusion/exec/                       ~35 MB — apptainer images + spec
#                                                     (FC can't run them; the
#                                                     entire dir is build-time
#                                                     packaging, not runtime)
#   rf_diffusion/aa_dataset_256_subsampled_10.pkl    74 MB \  training/eval-only
#   rf_diffusion/test_allatom_dataset_100.pkl        16 MB  ) datasets — only
#   rf_diffusion/aa_dataset_256_subsampled_10_2.pkl  11 MB  | reachable from
#   rf_diffusion/test_dataset_100.pkl               1.8 MB  | training configs,
#   rf_diffusion/test_fake_allatom_dataset.pkl       860 KB /  not inference
#   rf_diffusion/*.pse                              ~15 MB — pymol sessions
#                                                            (visualization)
#   rf_diffusion/dev/                      ~300 KB — dev scripts
#   notebooks / images / doc / assets / pymol_config — top-level docs (~1.5 MB)
#
# We KEEP rf_diffusion/benchmark/input/ (44 MB) — referenced as default
# `inference.input_pdb` by several configs under config/inference/. Even
# though our /api/generate always overrides this default, leaving it in
# means custom-config-name requests without an explicit input_pdb still
# work end-to-end instead of failing on a missing file.
RUN rm -rf \
        /src/rf_diffusion/test_data \
        /src/rf_diffusion/goldens \
        /src/rf_diffusion/benchmark/rotamer_library \
        /src/rf_diffusion/exec \
        /src/rf_diffusion/dev \
        /src/notebooks \
        /src/images \
        /src/doc \
        /src/assets \
        /src/pymol_config \
    && rm -f \
        /src/rf_diffusion/aa_dataset_256_subsampled_10.pkl \
        /src/rf_diffusion/aa_dataset_256_subsampled_10_2.pkl \
        /src/rf_diffusion/test_allatom_dataset_100.pkl \
        /src/rf_diffusion/test_dataset_100.pkl \
        /src/rf_diffusion/test_fake_allatom_dataset.pkl \
        /src/rf_diffusion/diffuse_fullatom.pse \
        /src/rf_diffusion/fwd_diffusion_example.pse


# ===========================================================================
# Stage 2: builder — install the rfd2 conda env on top of the trimmed source.
# ===========================================================================
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Pin torch's CUDA discovery to the system toolkit (same lesson as ppiflow):
# torch._find_cuda_home() reads CUDA_HOME / CUDA_PATH before walking PATH, so
# if conda puts an nvcc on PATH without matching headers, set these.
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda

# Build-only apt deps + micromamba bootstrap.
#   bzip2     — to extract the micromamba tarball
#   curl/wget — used by micromamba env-create (download.pytorch.org, data.dgl.ai)
#   git       — for the two `pip install git+...` deps below
#   build-essential — required by the JIT compile path in ipd.dev.cuda.cudabuild
#                     (kept in the runtime stage too — see below)
RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 curl git wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
       | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && /usr/local/bin/micromamba --version

ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/rfd2/bin:$PATH

# ---------------------------------------------------------------------------
# Mirror configuration for pip + conda, one combined layer (both files are
# < 1 KB; splitting them earlier just made the layer manifest noisier).
#
# pip: Aliyun mirror. Configure BEFORE micromamba env-create — env.yml's
# `pip:` section invokes pip during env create and would otherwise hit pypi.org.
#
# conda: TUNA (Tsinghua) for every channel env.yml touches, EXCEPT pyrosetta
# (no Chinese mirror exists for conda.rosettacommons.org — that ~600 MB
# download stays as the dominant single-package cost on a cold build).
#
# TUNA mirrors labelled sub-channels under the same path, so e.g. the env's
# `nvidia/label/cuda-12.4.0` resolves to
# `mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/nvidia/label/cuda-12.4.0`.
# ---------------------------------------------------------------------------
RUN mkdir -p /root/.config/pip \
    && printf '[global]\nindex-url = https://mirrors.aliyun.com/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
       > /root/.config/pip/pip.conf \
    && cat > /root/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  nvidia: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  dglteam: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# ---------------------------------------------------------------------------
# Create the rfd2 conda env. The env yaml's `pip:` section references
# `-r requirements_cuda124.txt`, so we must copy BOTH files to the same
# temp dir before invoking `env create`.
#
# BuildKit cache mounts (the big optimization):
#   /opt/conda/pkgs   conda package cache (~2 GB), saves re-download
#   /root/.cache/pip  pip wheel cache (~1.5 GB; torch wheels from
#                     download.pytorch.org, pyg-lib from data.pyg.org —
#                     neither has a China CDN). First build ~25 min;
#                     rebuilds drop to ~3 min.
#
# The cache-mounted dirs are excluded from the layer by BuildKit, so
# *not* doing `micromamba clean --all` here doesn't bloat the image.
# ---------------------------------------------------------------------------
COPY opensource/RFdiffusion2/envs/cuda124_env.yml /tmp/rfd2-env/cuda124_env.yml
COPY opensource/RFdiffusion2/envs/requirements_cuda124.txt /tmp/rfd2-env/requirements_cuda124.txt

RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    cd /tmp/rfd2-env \
    && micromamba env create -y -n rfd2 -f cuda124_env.yml \
    && rm -rf /tmp/rfd2-env

# ---------------------------------------------------------------------------
# Two pip extras the upstream apptainer .spec installs that aren't in
# requirements_cuda124.txt. Without these `import rf2aa` fails at runtime
# (rf2aa imports `ipd.dev` and uses PyTimer).
#
# Separate layer because these git-clone installs are slow (~30s each) and
# rarely change — combining them with the framework install (next layer)
# would re-clone on every framework change.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/conda/envs/rfd2/bin/pip install \
        git+https://github.com/RalphMao/PyTimer.git \
        git+https://github.com/baker-laboratory/ipd.git

# ---------------------------------------------------------------------------
# Pull in the pre-trimmed source from source-prep, plus the framework, and
# strip __pycache__/*.pyc from the env in the SAME layer. Cleanup in a later
# RUN would leak the bytes through the previous layer.
#
# Why the trimmed COPY: see source-prep stage above. About 350 MB saved.
# Why __pycache__ cleanup: conda installs ship pre-compiled .pyc bytecode,
# ~200 MB across the env. Python regenerates these on first import — minor
# cold-start cost, big image-size win.
# ---------------------------------------------------------------------------
COPY --from=source-prep /src /opt/rfdiffusion2
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/conda/envs/rfd2/bin/pip install \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2 \
    && rm -rf /tmp/service-framework \
    && find /opt/conda/envs/rfd2 -depth -type d -name '__pycache__' -exec rm -rf {} + \
    && find /opt/conda/envs/rfd2 -name '*.pyc' -delete


# ===========================================================================
# Stage 3: runtime — keep the JIT-compile toolchain, drop the rest.
# ===========================================================================
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/rfd2/bin:$PATH

# Runtime apt is minimal:
#   build-essential   — REQUIRED. ipd.dev.cuda.cudabuild does runtime JIT compile
#                       via torch.cpp_extension.load on first model construction.
#                       Without gcc the inference call crashes.
#   ca-certificates   — httpx HTTPS for OSS / signed URL inputs.
#
# We drop bzip2/curl/wget/git/micromamba (~150 MB on disk) and any other apt
# headers from the builder. The env is fully self-contained at this point.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Bring over the conda env + trimmed source as two separate COPYs so the
# big (and stable) one — the env — caches independently of source changes.
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /opt/rfdiffusion2 /opt/rfdiffusion2

# WORKDIR MUST be set in the runtime stage (it doesn't propagate from the
# builder stage). Without it `cwd` defaults to `/`, and uvicorn's
# `server.app:app` import looks for `server` under cwd + sys.path → fails with
# `ModuleNotFoundError: No module named 'server'` at FC cold start.
WORKDIR /opt/rfdiffusion2

# Server code last so iterating on the FastAPI layer only rebuilds the tail.
COPY services/rfdiffusion2-server/ /opt/rfdiffusion2/server/

# Settings (env-driven; see services/rfdiffusion2-server/settings.py).
ENV RFDIFFUSION2_ROOT=/opt/rfdiffusion2
ENV RFDIFFUSION2_MODELS_DIR=/opt/rfdiffusion2/rf_diffusion/model_weights
ENV RFDIFFUSION2_INFERENCE_SCRIPT=/opt/rfdiffusion2/rf_diffusion/run_inference.py
ENV RFDIFFUSION2_PYTHON=/opt/conda/envs/rfd2/bin/python
ENV RFDIFFUSION2_PYTHONPATH=/opt/rfdiffusion2
ENV RFDIFFUSION2_JOBS_BASE_DIR=/data/rfdiffusion2_jobs
RUN mkdir -p /data/rfdiffusion2_jobs

# Alibaba Cloud FC: port 9000, keep-alive ≥ 15 min, /healthz must be ready < 120 s.
ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/rfd2/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
