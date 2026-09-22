# 从仓库根构建：
#   docker build --platform linux/amd64 -t bindcraft2-server \
#       -f services/bindcraft2-server/Dockerfile .
#
# 构建前置（各跑一次；升级 pin 时重跑）：
#   ./services/bindcraft2-server/scripts/vendor.sh
#   ./services/bindcraft2-server/scripts/fetch_weights.sh   # 或 NAS 软链，见设计文档 P1
#
# 上游源码来自 services/bindcraft2-server/upstream/（vendor.sh 产物，固定 SHA）。
# AlphaFold 参数 5.3 GB **不烘焙**——挂 NAS 到 /data/models/bindcraft2/。
#
# 基础镜像用纯 ubuntu:24.04（与上游 containers/Dockerfile 一致）：jax cuda13
# wheels 自带 CUDA/cuDNN，宿主机只需提供 NVIDIA 驱动 + container toolkit。
# 用 nvidia/cuda base 不但冗余，还可能引入版本冲突。
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
ENV UV_HTTP_TIMEOUT=600
ENV UV_CONCURRENT_DOWNLOADS=4

# python3 = 3.12（ubuntu:24.04 默认），满足上游 requires-python >=3.12。
# 不装 git：上游源码已 vendor；不装 build-essential：amd64 上全是 wheel。
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# --- 上游源码 ---
# editable install：settings/ 与 scaffolds/ 在仓库根（不在包里），运行期按仓库根
# 相对定位，所以源码树必须留在 /opt/bindcraft 并留在 import path 上。
COPY services/bindcraft2-server/upstream /opt/bindcraft

# docs/ 与 containers/ 是构建期无关的大块，剪掉省体积；results/ 不该存在。
RUN rm -rf /opt/bindcraft/docs /opt/bindcraft/containers /opt/bindcraft/.github \
    && find /opt/bindcraft -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

RUN uv venv /opt/venv --python python3

# cuda13 extra：需要 compute capability >= 7.5（T4/A10/L20/A100）。
# V100(7.0) 及更老改用 [cuda12]。
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python -e "/opt/bindcraft[cuda13]"

# jax 的 cuda 库在 site-packages/nvidia/*/lib 下，动态链接器默认不查那里；
# 同时镜像必须显式声明这两个变量，NVIDIA container hook 才会注入宿主机驱动
# （纯 ubuntu base 不会像 CUDA base 那样预置）。
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# 必须用 venv 解释器：nvidia-* 只装在 /opt/venv，系统 python3 看不到它们。
RUN /opt/venv/bin/python -c "import nvidia, pathlib; print('\n'.join(sorted(str(p) for root in nvidia.__path__ for p in pathlib.Path(root).glob('*/lib'))))" \
        > /etc/ld.so.conf.d/bindcraft-cuda.conf \
 && ldconfig && ldconfig -p | grep -q libcupti

# --- 共享服务框架 ---
# COPY（不是 bind-mount）：bind mount 的内容不进 BuildKit 层缓存键，会发出陈旧框架。
COPY framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# --- 服务代码 ---
# 显式文件清单：测试/文档改动不破坏这一层缓存。
COPY services/bindcraft2-server/__init__.py \
     services/bindcraft2-server/__main__.py \
     services/bindcraft2-server/VERSION \
     services/bindcraft2-server/app.py \
     services/bindcraft2-server/adapter.py \
     services/bindcraft2-server/campaigns.py \
     services/bindcraft2-server/models.py \
     services/bindcraft2-server/settings.py \
     services/bindcraft2-server/tools.py \
     /opt/bindcraft2/server/

# --- 构建期自检 ---
# 让"wheel 没对上"和"包没带上权重"在这里失败，而不是在别人第一次 campaign 时。
RUN test -f /opt/bindcraft/bindcraft/weights/proteinmpnn/weights_neutral/v_48_020.npz \
 && test -f /opt/bindcraft/bindcraft/weights/proteinmpnn/weights_negative/v_48_020.npz \
 && test -f /opt/bindcraft/bindcraft/weights/proteinmpnn/weights_positive/v_48_020.npz \
 && test -d /opt/bindcraft/settings/target \
 && test -f /opt/bindcraft/bindcraft/cli.py \
 && /opt/venv/bin/python -c "import jax; print('jax', jax.__version__)" \
 && /opt/venv/bin/python -c "import bindcraft.proteinmpnn; print('proteinmpnn ok')" \
 && cd / && /opt/venv/bin/python -m bindcraft.cli --help > /dev/null

# --- 运行期环境 ---
ENV PYTHONPATH=/opt/bindcraft2
ENV BINDCRAFT2_ROOT=/opt/bindcraft
ENV BINDCRAFT2_PYTHON=/opt/venv/bin/python
ENV BINDCRAFT2_JOBS_BASE_DIR=/data/bindcraft2_jobs
ENV BINDCRAFT2_ALPHAFOLD_PARAMS_DIR=/data/models/bindcraft2/alphafold
ENV BINDCRAFT2_COMPILE_CACHE_DIR=/data/models/bindcraft2/xla_cache
ENV BINDCRAFT2_MAX_CONCURRENT_JOBS=1
ENV BINDCRAFT2_WORKERS_PER_GPU=1
ENV BINDCRAFT2_MAX_WORKERS_PER_GPU=1
# 输出回传：网关带 X-Bioagent-Oss-Prefix 时框架把完成的 job 目录镜像到这里。
# FC 控制台需要把数据面 OSS bucket 挂到 /mnt/oss（RW）；没挂则是 no-op。
ENV BINDCRAFT2_OSS_OUTPUT_MOUNT=/mnt/oss
# FC 会话亲和：在返回 job_id 的 POST 响应里回显这个 header。
ENV BINDCRAFT2_SESSION_HEADER_NAME=bioagent-session-id
RUN mkdir -p /data/bindcraft2_jobs

WORKDIR /opt/bindcraft

# ldconfig 必须在访问 GPU 之前跑过（非 root 场景下 /sbin/ldconfig 需要 setuid）。
# 用 printf 而非 $'...'：Ubuntu 的 /bin/sh 是 dash，不支持 ANSI-C 引用，
# 会把字面量写进文件导致 exec format error。
RUN printf '#!/bin/bash\nldconfig\nexec "$@"\n' > /opt/entrypoint.sh \
 && chmod +x /opt/entrypoint.sh
ENTRYPOINT ["/opt/entrypoint.sh"]

EXPOSE 9000
CMD ["/opt/venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "9000", "--timeout-keep-alive", "900"]
