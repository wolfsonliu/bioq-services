# Build from bioagent project root:
#   docker build --platform linux/amd64 -t ppiflow-server -f services/ppiflow-server/Dockerfile .
#
# PPIFlow source must be at opensource/PPIFlow/ (weights baked into image).
# This image only supports structure generation (sample_*.py). Sequence design,
# scoring, packing, and Rosetta steps are NOT included — they belong in
# separate bioagent services.
FROM mambaorg/micromamba:1.5.10-cuda12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Switch to root for system installs; micromamba is the default conda manager.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Create the `ppiflow` conda env from upstream's environment.yml.
# Cache-friendly layer: env.yml rarely changes vs the PPIFlow source itself.
# ---------------------------------------------------------------------------
WORKDIR /opt/ppiflow

COPY opensource/PPIFlow/tool/PPIFlow/environment.yml /tmp/ppiflow-env.yml
RUN micromamba env create -y -n ppiflow -f /tmp/ppiflow-env.yml \
    && micromamba clean --all --yes \
    && rm /tmp/ppiflow-env.yml

ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/ppiflow/bin:$PATH

# ---------------------------------------------------------------------------
# Copy the PPIFlow tool source + pretrained checkpoints.
# The four .ckpt files baked into the image are heavy (~100s MB each); make
# sure opensource/PPIFlow/checkpoint/ is populated before building.
# ---------------------------------------------------------------------------
COPY opensource/PPIFlow/tool/PPIFlow /opt/ppiflow
COPY opensource/PPIFlow/checkpoint /opt/ppiflow/checkpoint

# ---------------------------------------------------------------------------
# Shared bioagent service framework + remote-fetch deps in the same env.
# Keeping this as a late layer so the framework re-install doesn't invalidate
# the (slow) conda env build cache.
# ---------------------------------------------------------------------------
COPY services/_framework /tmp/service-framework
RUN /opt/conda/envs/ppiflow/bin/pip install /tmp/service-framework \
    && /opt/conda/envs/ppiflow/bin/pip install httpx alibabacloud-oss-v2

# Server code as `server` package — uvicorn imports `server.app:app`. Mounted
# alongside PPIFlow's tool directory so subprocess_cwd resolves correctly.
COPY services/ppiflow-server/ /opt/ppiflow/server/

# Settings via env (see services/ppiflow-server/settings.py).
ENV PPIFLOW_ROOT=/opt/ppiflow
ENV PPIFLOW_CKPT_DIR=/opt/ppiflow/checkpoint
ENV PPIFLOW_CONFIG_DIR=/opt/ppiflow/configs
ENV PPIFLOW_JOBS_BASE_DIR=/data/ppiflow_jobs
RUN mkdir -p /data/ppiflow_jobs

# Alibaba Cloud FC plumbing.
ENV PORT=9000
EXPOSE 9000
CMD ["/opt/conda/envs/ppiflow/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
