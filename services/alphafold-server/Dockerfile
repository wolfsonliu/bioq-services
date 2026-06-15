# Build from bioagent project root:
#   docker build --platform linux/amd64 -t alphafold-server -f services/alphafold-server/Dockerfile .
#
# AlphaFold source must be at opensource/alphafold/.
# Databases and weights are NOT bundled — mount NAS at /data/models/alphafold/.
#
# Base: nvidia/cuda 12.2 + cuDNN8, matching the official AlphaFold Dockerfile.
ARG CUDA=12.2.2
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu20.04
ARG CUDA

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --yes --quiet \
        build-essential \
        cmake \
        cuda-command-line-tools-$(cut -f1,2 -d- <<< ${CUDA//./-}) \
        git \
        hmmer \
        kalign \
        tzdata \
        wget \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove --yes \
    && apt-get clean

# HH-suite from source.
RUN git clone --branch v3.3.0 --single-branch https://github.com/soedinglab/hh-suite.git /tmp/hh-suite \
    && mkdir /tmp/hh-suite/build \
    && pushd /tmp/hh-suite/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
    && make -j && make install \
    && ln -s /opt/hhsuite/bin/* /usr/bin \
    && popd \
    && rm -rf /tmp/hh-suite

# Miniconda.
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS="yes"
RUN conda install --quiet --yes conda==24.11.1 pip python=3.11 \
    && conda install --quiet --yes --channel nvidia cuda=${CUDA_VERSION} \
    && conda clean --all --force-pkgs-dirs --yes

# AlphaFold source.
COPY opensource/alphafold/ /opt/alphafold/
RUN wget -q -P /opt/alphafold/alphafold/common/ \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# Python dependencies (AlphaFold requirements + JAX CUDA).
RUN pip3 install --upgrade pip setuptools --no-cache-dir \
    && pip3 install -r /opt/alphafold/requirements.txt --no-cache-dir \
    && pip3 install --upgrade --no-cache-dir \
      jax==0.4.26 \
      jaxlib==0.4.26+cuda12.cudnn89 \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Fix SETUID for ldconfig (GPU visibility in non-root).
RUN chmod u+s /sbin/ldconfig.real
RUN ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 /opt/conda/lib/libffi.so.7

# Shared bioagent service framework + remote-fetch deps.
RUN --mount=type=bind,source=services/_framework,target=/tmp/service-framework \
    pip3 install --no-cache-dir \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# Server code — explicit file list so test/doc changes don't bust cache.
COPY services/alphafold-server/__init__.py \
     services/alphafold-server/settings.py \
     services/alphafold-server/models.py \
     services/alphafold-server/adapter.py \
     services/alphafold-server/tools.py \
     services/alphafold-server/uris.py \
     services/alphafold-server/app.py \
     services/alphafold-server/__main__.py \
     services/alphafold-server/VERSION \
     /opt/alphafold/server/

# Environment.
ENV ALPHAFOLD_ROOT=/opt/alphafold
ENV ALPHAFOLD_PYTHON=/opt/conda/bin/python
ENV ALPHAFOLD_JOBS_BASE_DIR=/data/alphafold_jobs
ENV ALPHAFOLD_DATA_DIR=/data/models/alphafold
ENV ALPHAFOLD_N_CPU=8
ENV ALPHAFOLD_MAX_CONCURRENT_JOBS=1
ENV TF_FORCE_UNIFIED_MEMORY=1
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=4.0
ENV OPENMM_CPU_THREADS=8
ENV PYTHONPATH=/opt/alphafold


# Keepalive pings this URL through FC's gateway so the instance stays alive
# during long GPU jobs. Override at deploy time if the URL changes.
# ENV ALPHAFOLD_KEEPALIVE_URL=
# FC session affinity: inject this header in POST responses so FC binds
# follow-up requests to the same instance. Requires FC console config too.
ENV ALPHAFOLD_SESSION_HEADER_NAME=bioagent-session-id

WORKDIR /opt/alphafold

# ldconfig must run before GPU access.
RUN echo $'#!/bin/bash\nldconfig\nexec "$@"' > /opt/entrypoint.sh \
    && chmod +x /opt/entrypoint.sh
ENTRYPOINT ["/opt/entrypoint.sh"]

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
