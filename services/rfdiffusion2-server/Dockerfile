# Build from bioagent project root:
#   docker build --platform linux/amd64 -t rfdiffusion2-server -f services/rfdiffusion2-server/Dockerfile .
#
# RFdiffusion2 source must be at opensource/RFdiffusion2/ (weights baked into image,
# ~2.6 GB across RFD_140.pt + RFD_173.pt; the 2 LigandMPNN weights are copied too
# even though run_inference.py doesn't use them — downstream pipelines need them).
# Run `python setup.py` inside opensource/RFdiffusion2/ first to download the weights.
#
# Base: CUDA 12.4 -devel. RFdiffusion2 mirrors the rf_diffusion_aa.spec apptainer
# image (cuda 12.4 + pytorch 2.4 + cu124). The DEVEL variant is required because
# rf2aa imports `cuequivariant`/`pyg_lib` which can JIT-compile CUDA kernels on
# first model construction. Same lesson as ppiflow-server (see
# engineering/guides/uv-dockerfile-patterns.md for the diagnosis).
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Pin torch's CUDA discovery to the system toolkit (same lesson as ppiflow-server):
# torch._find_cuda_home() reads CUDA_HOME / CUDA_PATH before walking PATH, so
# if a conda env puts an nvcc on PATH without the matching headers, set these.
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda

# ---------------------------------------------------------------------------
# Base tooling: micromamba (conda env manager). The conda env's bundled pip
# (`/opt/conda/envs/rfd2/bin/pip`) handles the wheel-only installs in later
# layers — no uv. Keeps the toolchain to a single package manager so the env
# manifest stays the source of truth for what's installed.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 curl git wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
       | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && /usr/local/bin/micromamba --version

ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/rfd2/bin:$PATH

# ---------------------------------------------------------------------------
# Mirror configuration — pip + conda.
#
# pip: Huawei Cloud is the project-wide mirror. Configure BEFORE micromamba
# creates the env (env.yml has a `pip:` section that triggers pip resolution
# during conda env create).
#
# conda: TUNA (Tsinghua) mirror for the standard channels (conda-forge,
# pytorch, pkgs/main). The two upstream-only channels we keep as-is:
#   - https://conda.rosettacommons.org   (pyrosetta; not mirrored anywhere)
#   - nvidia/label/cuda-12.4.0           (cuda packages; nvidia channel
#                                          isn't on TUNA, falls through to
#                                          upstream conda.anaconda.org/nvidia)
#   - dglteam/label/th24_cu124           (dgl 2.4 + cu124; falls through to
#                                          conda.anaconda.org/dglteam)
# ---------------------------------------------------------------------------
RUN mkdir -p /root/.config/pip && \
    printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
    > /root/.config/pip/pip.conf

RUN cat > /root/.condarc <<'EOF'
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
EOF

# ---------------------------------------------------------------------------
# Create the rfd2 conda env from upstream's envs/cuda124_env.yml.
#
# The env yaml `pip:` section references `-r requirements_cuda124.txt`, so we
# must copy BOTH files to the same temp dir before invoking `env create`.
#
# BuildKit cache mounts (same optimization rationale as ppiflow-server):
#   - /opt/conda/pkgs : conda package cache (~2 GB), saves re-download.
#   - /root/.cache/pip : torch wheels, pyg-lib wheels (~1.5 GB combined), no
#                        China CDN for download.pytorch.org or data.pyg.org.
# ---------------------------------------------------------------------------
WORKDIR /opt/rfdiffusion2

COPY opensource/RFdiffusion2/envs/cuda124_env.yml /tmp/rfd2-env/cuda124_env.yml
COPY opensource/RFdiffusion2/envs/requirements_cuda124.txt /tmp/rfd2-env/requirements_cuda124.txt

RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    cd /tmp/rfd2-env && \
    micromamba env create -y -n rfd2 -f cuda124_env.yml && \
    rm -rf /tmp/rfd2-env

# ---------------------------------------------------------------------------
# Two pip extras the spec file installs that aren't in requirements_cuda124.txt
# but ARE in the apptainer .spec (see opensource/RFdiffusion2/rf_diffusion/exec/
# rf_diffusion_aa.spec). Without these, `import rf2aa` fails at runtime because
# rf2aa pulls in ipd.dev.
#
#   - ipd: baker_laboratory's "IPD" dev utilities, github-only package.
#   - PyTimer: timing helpers used by rf2aa.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/conda/envs/rfd2/bin/pip install \
        git+https://github.com/RalphMao/PyTimer.git \
        git+https://github.com/baker-laboratory/ipd.git

# ---------------------------------------------------------------------------
# Copy RFdiffusion2 source + model weights. Heavy COPY late so wheel layers
# above stay cached when only source changes.
#
# Weights are ~2.6 GB total (RFD_140.pt + RFD_173.pt; the two ligand_mpnn
# weights at ~70 MB are also copied for downstream pipeline use even though
# run_inference.py doesn't load them). Make sure
# `cd opensource/RFdiffusion2 && python setup.py` ran locally before building.
# ---------------------------------------------------------------------------
COPY opensource/RFdiffusion2 /opt/rfdiffusion2

# ---------------------------------------------------------------------------
# Shared bioagent service framework + remote-fetch deps. Late layer so a
# framework re-install doesn't invalidate the slow conda-env build above.
# Reuses the same pip cache mount as the env.yml step.
# ---------------------------------------------------------------------------
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/conda/envs/rfd2/bin/pip install \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2 \
    && rm -rf /tmp/service-framework

# Server code as `server` package — uvicorn imports `server.app:app`.
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
