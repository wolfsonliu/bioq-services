# Build from bioagent project root:
#   docker build --platform linux/amd64 -t proteinmpnn-server -f services/proteinmpnn-server/Dockerfile .
#
# ProteinMPNN source must be at opensource/ProteinMPNN/ with the 4 weight folders
# (vanilla_model_weights, soluble_model_weights, ca_model_weights, AbMPNN_model_weights)
# populated. Total weight bake-in ~1 GB.
#
# Base: same nvidia/cuda 12.4.1 image as ppiflow-server / genie3-server to share CI
# layer cache. CUDA 12.x runtime is forward-compatible with the torch wheels we install.
# `-runtime` (not `-devel`) — ProteinMPNN install path is wheel-only (torch + numpy +
# framework), no nvcc / CUDA headers required. Save ~3.6 GB vs `-devel`.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

# Huawei Cloud pip mirror (matches project-wide convention)
RUN mkdir -p /root/.config/pip && \
    printf '[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple/\nextra-index-url = https://pypi.org/simple\ntimeout = 120\nretries = 5\n' \
    > /root/.config/pip/pip.conf

# Torch first (heavy layer, rarely changes). CUDA 12.1 wheel works on 12.4 runtime.
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
       torch==2.4.1 \
    && python -m pip install numpy

# ProteinMPNN source + all four weight folders.
COPY opensource/ProteinMPNN /opt/proteinmpnn

# Shared bioagent service framework + remote-fetch deps (late layer).
COPY services/_framework /tmp/service-framework
RUN python -m pip install "/tmp/service-framework[mcp]" \
    && python -m pip install httpx alibabacloud-oss-v2

# Server code as `server` package — uvicorn imports `server.app:app`. Mounted
# alongside ProteinMPNN's repo so subprocess_cwd resolves correctly.
COPY services/proteinmpnn-server/ /opt/proteinmpnn/server/

ENV PROTEINMPNN_ROOT=/opt/proteinmpnn
ENV PROTEINMPNN_WEIGHTS_DIR=/opt/proteinmpnn
ENV PROTEINMPNN_JOBS_BASE_DIR=/data/proteinmpnn_jobs
RUN mkdir -p /data/proteinmpnn_jobs

ENV PORT=9000
EXPOSE 9000
CMD ["python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
