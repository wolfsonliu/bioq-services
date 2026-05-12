# Build from bioagent project root:
#   docker build --platform linux/amd64 -t genie3-server -f services/genie3-server/Dockerfile .
#
# Genie3 source must be at opensource/genie3/ (pretrained v1 weights baked into image).
# This image only supports `genie3 generate` — evaluation deps (ColabFold/ESMFold/FoldSeek/
# ProteinMPNN/TMscore/DSSP/IPSAE) are intentionally excluded to keep the image small.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    git wget curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

# --- Install Genie3 (generation deps only) ---
WORKDIR /opt/genie3

# Bootstrap build deps before installing the package (some deps need numpy/Cython at build time)
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install "numpy>=2.0.2,<3" Cython

# Genie3 generation runtime deps. Keep this list in sync with opensource/genie3/setup.py.
# Notes:
#   - torch==2.7.1 ships with CUDA 12.6 wheels (compatible with the 12.4 base CUDA driver runtime).
#   - PyYAML is required by genie3/cli.py for iterative configs.
#   - We omit wandb/tensorboard server features but keep the python packages so imports work.
RUN python -m pip install \
        torch==2.7.1 \
        tqdm scipy pandas \
        lightning \
        "biopython<1.86" \
        ml-collections zstandard \
        huggingface_hub \
        wandb tensorboard \
        PyYAML

# Copy genie3 source + pretrained v1 weights (drop training data / examples to slim the image)
COPY opensource/genie3/setup.py /opt/genie3/setup.py
COPY opensource/genie3/src /opt/genie3/src
COPY opensource/genie3/pretrained/v1 /opt/genie3/pretrained/v1

# Apply local patches to upstream genie3. See services/genie3-server/patches/README.md.
COPY services/genie3-server/patches /tmp/genie3-patches
RUN cd /opt/genie3 && for p in /tmp/genie3-patches/*.patch; do \
        echo "Applying $p"; patch -p1 < "$p"; \
    done

# Editable install of genie3 (registers the `genie3` console entry point)
RUN python -m pip install --no-build-isolation -e .

# --- Server dependencies ---
RUN python -m pip install fastapi "uvicorn[standard]" python-multipart httpx alibabacloud-oss-v2

# --- Server code ---
COPY services/genie3-server/ /opt/genie3/server/

ENV GENIE3_ROOT=/opt/genie3
ENV GENIE3_BIN=genie3
ENV GENIE3_JOBS_DIR=/data/genie3_jobs
RUN mkdir -p /data/genie3_jobs

# --- Alibaba Cloud FC configuration ---
# CAPort: FC default listening port is 9000
# Keep-alive: FC requires >= 15 minutes (900s)
# Startup: HTTP server must be ready within 120s
ENV PORT=9000
EXPOSE 9000

CMD ["python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--timeout-keep-alive", "900"]
