# Build from bioagent project root:
#   docker build --platform linux/amd64 -t ppiflow-server -f services/ppiflow-server/Dockerfile .
#
# PPIFlow source must be at opensource/PPIFlow/ (weights baked into image).
# This image only supports structure generation (sample_*.py). Sequence design,
# scoring, packing, and Rosetta steps are NOT included — they belong in
# separate bioagent services.
#
# Base: same as genie3-server so the CI image-layer cache is shared across the
# two GPU services. CUDA 12.4 runtime is forward-compatible with the 12.1 PyTorch
# wheels PPIFlow's environment.yml pulls in (same CUDA major). The conda env is
# materialized via micromamba — PPIFlow's upstream env.yml ships a bunch of
# bioconda / ostrokach packages with no pip equivalents, so pure pip isn't viable.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Install micromamba on top of the nvidia/cuda base. We pin via the public
# tarball URL rather than apt to avoid pulling in conda's whole channel cache.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 curl git wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
       | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && /usr/local/bin/micromamba --version

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
# ---------------------------------------------------------------------------
RUN mkdir -p /root/.config/pip && \
    printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
    > /root/.config/pip/pip.conf

# ---------------------------------------------------------------------------
# Create the `ppiflow` conda env from upstream's environment.yml.
# Cache-friendly layer: env.yml rarely changes vs the PPIFlow source itself.
# ---------------------------------------------------------------------------
WORKDIR /opt/ppiflow

COPY opensource/PPIFlow/tool/PPIFlow/environment.yml /tmp/ppiflow-env.yml
RUN micromamba env create -y -n ppiflow -f /tmp/ppiflow-env.yml \
    && micromamba clean --all --yes \
    && rm /tmp/ppiflow-env.yml

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
