# Dockerfile — 构建骨架

日期: 2026-07-14
适用: [新增 bioagent service cookbook](./index.md) 的镜像构建部分
相关: [conda-pitfalls](./conda-pitfalls.md) · [skeleton](./skeleton.md) · [总览](./index.md)

> ← 返回 [新增 service cookbook 总览](./index.md)

本页覆盖 `vendor.sh` / `fetch_weights.sh` / uv venv 与 conda/micromamba 两套 Dockerfile
骨架 / 上游修改的 wrapper vs patch 决策。包装 conda-based upstream 的踩坑单独见
[conda-pitfalls](./conda-pitfalls.md)。

### 8. `services/<svc>/Dockerfile`

#### 8.0 前置：vendor.sh + fetch_weights.sh（必看）

**新约定（2026-06 起）**：Docker build 期不再 clone 上游 / 不再烘焙权重。
两件事在 build 前由 host 上的脚本完成：

| 步骤 | 脚本 | 目的 |
|---|---|---|
| 1. Vendor 上游源码 | `scripts/vendor.sh` | 把 upstream git repo at pinned SHA `git clone` 到 `services/<svc>/upstream/` |
| 2.（可选）下载权重 | `scripts/fetch_weights.sh` | 下载到 `services/<svc>/weights/` 或直接 NAS（`WEIGHTS_DST=` env） |

**why**：

- ✅ Build 期 0 网络访问 github → 国内构建稳定（不被 TLS 抽风影响）
- ✅ 服务自包含（`services/<svc>/` 含 upstream + scripts + 框架 wrapper）
- ✅ 上游 SHA 写在脚本里显式可见 + 重试 + 校验
- ✅ 镜像里只有代码栈 → 缩到 1.5-3 GB（vs 5-18 GB），FC 拉取更快、权重独立打版

详情见 Service 权重 NAS 外置化设计。

#### 8.1 vendor.sh 模板

```bash
#!/usr/bin/env bash
# Vendor the upstream <name> source into services/<svc>/upstream/ at a
# pinned SHA, so `docker build` does no network access.
#
#   ./services/<svc>/scripts/vendor.sh
#
# Github mirror override (CN networks, flaky TLS):
#
#   <NAME>_REPO=https://ghproxy.cn/https://github.com/<owner>/<repo>.git \
#       ./services/<svc>/scripts/vendor.sh
#
# To bump the upstream pin, edit <NAME>_SHA below.

set -euo pipefail

<NAME>_REPO="${<NAME>_REPO:-https://github.com/<owner>/<repo>.git}"
<NAME>_SHA="${<NAME>_SHA:-<40-char-sha>}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DST="$PROJECT_ROOT/services/<svc>/upstream"
TMP="$(mktemp -d -t <svc>-vendor.XXXXXX)"
trap "rm -rf '$TMP'" EXIT

mkdir -p "$DST"

# 重试 5 次：CN 构建机偶发 GnuTLS TLS reset
for i in 1 2 3 4 5; do
    rm -rf "$TMP/repo"
    if git clone --filter=blob:none --no-checkout "${<NAME>_REPO}" "$TMP/repo"; then
        break
    fi
    [ "$i" = "5" ] && { echo "ERROR: git clone failed after 5 attempts" >&2; exit 1; }
    echo "  clone failed, retrying in $((i*10))s ..."
    sleep $((i*10))
done

cd "$TMP/repo"
git checkout "${<NAME>_SHA}"
actual="$(git rev-parse HEAD)"
[[ "$actual" = "${<NAME>_SHA}" ]] || {
    echo "ERROR: HEAD mismatch (got $actual, expected ${<NAME>_SHA})" >&2
    exit 1
}
rm -rf .git

rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    "$TMP/repo/" "$DST/"

echo "Vendored ${<NAME>_REPO} @ ${<NAME>_SHA}"
echo "  -> $DST"
du -sh "$DST"
```

**多 upstream**（如 promera-server 需要 tinyprot + promera + LigandMPNN）：把
`vendor_one()` 抽成函数，串行 clone。完整范例：[promera-server/scripts/vendor.sh](../../services/promera-server/scripts/vendor.sh)。

把 `services/<svc>/upstream/` 加进项目根 `.gitignore`。

#### 8.2 fetch_weights.sh 模板（如服务需要本地权重）

```bash
#!/usr/bin/env bash
# Download <svc> weights.
#
# 权重不再 baked 到 image —— 走 NAS（FC）或 apptainer --bind（SIF）。
# 本脚本只负责下载到 stage 目录；正式部署再 rsync 到 NAS / HPC scratch。
#
# Default（本地 stage）:
#   ./services/<svc>/scripts/fetch_weights.sh
#       → services/<svc>/weights/
#
# 直接下到 NAS / HPC scratch:
#   WEIGHTS_DST=/mnt/nas/data/models/<svc> \
#       ./services/<svc>/scripts/fetch_weights.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DST="${WEIGHTS_DST:-$SCRIPT_DIR/../weights}"
mkdir -p "$DST"

# wget -c 支持断点续传
for f in <list-of-files>; do
    wget -c -O "$DST/$f" "${BASE_URL}/$f"
done
```

##### 权重外置的两个常见坑

**坑 1：checkpoint 目录下的 `args.yaml` / `config.json` 必须一起 stash**

多数 pytorch-lightning / 自研训练框架在 checkpoint 目录**旁边**留一份训练超参
（`args.yaml` / `config.json` / `hparams.yaml`），推理端加载模型时会**先读它**
才知道怎么构造网络。DiffDock-PP、boltz、DeepRank-Ab 都是这个模式。

`fetch_weights.sh` 必须把这些辅助文件一起 cp 到 NAS，`/healthz/detail` 探针里
也应列进 `expected` 字典：

```bash
# 例：DiffDock-PP fold_0 目录 layout
CKPT_DIR="$SRC/large_model_dips/fold_0"
DST_DIR="$DST/large_model_dips/fold_0"
mkdir -p "$DST_DIR"
cp "$CKPT_DIR"/model_best_*.pth "$DST_DIR/"    # 主 checkpoint
cp "$SRC/large_model_dips/args.yaml" "$DST/large_model_dips/"  # 训练超参，别忘！
```

漏掉 `args.yaml` 会让服务在**第一次推理时**报"missing key in checkpoint" 或
"unexpected keyword argument"，而 build / import / healthz 全绿——非常难查。

**坑 2：上游用 `torch.hub` / `transformers`AutoModel 拉辅助权重（ESM / T5 / 等）**

如果上游 pipeline 里有 `torch.hub.load("facebookresearch/esm:main", ...)` 或
`AutoModel.from_pretrained("...")`，容器**运行时会尝试网络下载**——FC 出口
受限、SIF 常无网络，必须 pre-cache 到 NAS。参考 deeprank-ab-server（ESM-2）
和 alphafold-server（参数 tar）的做法：

```bash
# fetch_weights.sh 里加一段：把 ESM-2 stash 到 torch.hub 期望的 cache 结构
mkdir -p "$DST/esm_cache/hub/checkpoints"
wget -c -O "$DST/esm_cache/hub/checkpoints/esm2_t33_650M_UR50D.pt" \
    "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt"
# 上游 torch.hub API 还需要 hubconf.py，也 stash 一份
git clone --depth 1 https://github.com/facebookresearch/esm.git \
    "$DST/esm_cache/hub/facebookresearch_esm_main"
```

服务端在 `settings.py` / `Dockerfile` 里把 `TORCH_HOME` / `HF_HOME` /
`TRANSFORMERS_CACHE` 指向该 NAS 子目录，运行时就走 offline 缓存分支：

```python
# settings.py 追加
torchhub_dir: Path = Field(default=Path("/data/models/<svc>/esm_cache"))
```

```dockerfile
# Dockerfile 追加（让上游库看到 offline cache）
ENV TORCH_HOME=/data/models/<svc>/esm_cache
ENV HF_HOME=/data/models/<svc>/hf_cache
```

`/healthz/detail` 探针里要把这些辅助 checkpoint 也列进 `expected`：

```python
expected = {
    "score_ckpt": settings.weights_dir / "large_model_dips/fold_0/model_best.pth",
    "score_args": settings.weights_dir / "large_model_dips/args.yaml",
    "esm2_weight": settings.weights_dir / "esm_cache/hub/checkpoints/esm2_t33_650M_UR50D.pt",
}
```

#### 8.3 Dockerfile

默认采用「uv venv + Huawei 镜像」骨架。其他场景（多阶段构建 / system Python /
conda 依赖）见 uv-dockerfile-patterns 的模式对照表。

```dockerfile
# Build from bioagent project root:
#   docker build --platform linux/amd64 -t <svc> -f services/<svc>/Dockerfile .
#
# Prerequisites (一次性，重跑可升级 SHA / 权重):
#   ./services/<svc>/scripts/vendor.sh           # vendor upstream/
#   ./services/<svc>/scripts/fetch_weights.sh    # 下到 weights/，再 rsync 到 NAS
#
# 上游源码从 services/<svc>/upstream/ 注入（vendor.sh 产物）；
# 权重从 NAS at /data/models/<svc>/ 加载（不 baked 到 image）。
# SIF / HPC 用 `apptainer run --bind /scratch/models/<svc>:/data/models/<svc>` 注入。

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04   # 选 CUDA major；GPU 算法用 runtime 不用 devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Huawei Cloud pip 镜像 —— 国内构建必备，PyPI 默认 torch wheel 已含 CUDA 12.x。
ENV UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
# 注意：apt 里**通常不装 git** —— vendor.sh 在 host 上跑，image 内不需要。
# **例外**：如果后面某个 `uv pip install` 有 `"<pkg> @ git+https://..."`
# VCS specifier（如 openfold, dllogger, github fork），必须加 `git` 到 apt
# 列表 —— uv 0.11+ 不再 bundle libgit2，运行时 shell out 到 `git` CLI。
# 缺 git 报错信号："Git executable not found"。

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# 在算法目录下建 venv —— uv venv 必须先于 COPY 源码，Docker COPY 不会删除已存在的 .venv/
WORKDIR /opt/<svc>
RUN uv venv .venv --python python3.10

# Heavy deps 先装（torch / numpy / 其他重量级 wheel）—— 这层变动少，缓存命中率高
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python torch numpy

# 上游算法源码 —— **从 vendor.sh 产物 COPY**，不在 image 内 git clone
# （如果上游是 Python 包，可 editable install）
COPY services/<svc>/upstream /opt/<svc>/upstream
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python -e ./upstream

# Shared bioagent service framework + remote-fetch deps —— 单独一层，框架变动不会
# 让算法层缓存失效
COPY services/_framework /tmp/service-framework
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# Server code as `server` package — uvicorn imports `server.app:app`.
COPY services/<svc>/ /opt/<svc>/server/

# ---- Weights：NAS 挂载，**不 COPY 进 image** ----
# /data/models/<svc>/ 由 FC 自动挂载；SIF / HPC 用 --bind 注入。
# 权重缺失时 /healthz/detail 报 weights_loaded=false（见 app.py healthz_detail 实现）。

# Settings via env (see services/<svc>/settings.py).
# PYTHONPATH 让 `server` 包在任何 CWD 下都能 import（Docker WORKDIR 保证
# 了 Docker 本身能用，但 Apptainer/Singularity 不尊重 WORKDIR）。
ENV PYTHONPATH=/opt/<svc>
ENV <SVC>_ROOT=/opt/<svc>
ENV <SVC>_JOBS_BASE_DIR=/data/<svc>_jobs
ENV <SVC>_WEIGHTS_DIR=/data/models/<svc>   # 与 settings.py default 一致
RUN mkdir -p /data/<svc>_jobs

# ---- Output-sink（经 gateway 调用必备）----
# gateway 每次 run 都下发 X-Bioagent-Oss-Prefix；job 完成后 framework 把整个 NAS
# job dir 镜像到 <此挂载>/users/<user>/<job_id>/ 并写 results.zip，gateway 据此
# 走 OSS 302 提供 download。framework 默认值已是 /mnt/oss，此处显式声明作自文档。
# 需在 FC 控制台把数据面 OSS bucket 挂到 /mnt/oss（见「FC 控制台配置」）；缺挂载
# => no-op（仅 NAS）。详见 迁移到 OSS mount。
ENV <SVC>_OSS_OUTPUT_MOUNT=/mnt/oss

ENV PORT=9000
EXPOSE 9000
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "9000", "--timeout-keep-alive", "900"]
```

关键约束（详情见 uv-dockerfile-patterns）：

- **`-runtime` 而非 `-devel`** —— 算法走 prebuilt wheel 时不需要 nvcc / CUDA headers，可省 ~3.6 GB
- **通常不在 apt 装 git** —— vendor.sh 在 host 上跑，image 内不需要。装了反而误导未来 contributor 以为可以 in-build clone。**例外**：`uv pip install "<pkg> @ git+https://..."`（openfold / dllogger / github fork 等）需要 `git` CLI，此时**必须** apt-install git —— uv 0.11+ 不再 bundle libgit2，缺 git 报 "Git executable not found"（见 [diffdock-server Dockerfile](../../services/diffdock-server/Dockerfile) 参考）
- **WORKDIR 必须在 `server/` 父目录** —— 否则 Docker 模式下 `server.app` import 失败
- **`ENV PYTHONPATH=<workdir>` 必填** —— Apptainer/Singularity 不尊重 WORKDIR，必须通过 PYTHONPATH 让 `server` 包在任何 CWD 下可导入（见 apptainer-compatibility）
- **`uv venv` 先于 `COPY services/<svc>/upstream`** —— `.venv/` 不会被源码覆盖
- **CMD 用 `.venv/bin/python` 绝对路径** —— 不依赖 PATH
- **每个 RUN 加 `--mount=type=cache,target=/root/.cache/uv`** —— 加速重建，缓存不写镜像层（见 uv-cache-mount）
- **不要 `COPY services/<svc>/weights/`** —— 权重在 NAS。如果一定要烘焙小权重（如 < 100 MB 且无版本管理需求），可以例外，但需在 Dockerfile 注释里写明原因

#### 8.4 上游需要修改：wrapper vs patch 决策

上游代码几乎总要"稍微改一下"才能挂进服务框架（改 argparse、暴露硬编码路径、
关掉训练期专属的 wandb 分支等）。项目里目前共存三种做法，选错了会导致
vendor 升级 → 补丁 rebase → 每次都痛。**决策标准**：

| 方案 | 典型场景 | 优点 | 缺点 | 例子 |
|---|---|---|---|---|
| **A. `server/inference.py` wrapper**（首选） | 需要暴露参数化路径 / 加自定义 argparse / 前后端粘合（数据布局 shim / 结果后处理） | 上游 0 修改；wrapper 全在 `services/<svc>/` 里可审计 | 多一个文件；wrapper 需要清楚上游入口的签名 | [diffusion-hopping](../../services/diffusion-hopping-server/inference.py)、[turbohopp](../../services/turbohopp-server/inference.py)、[drughive](../../services/drughive-server/) |
| **B. `patches/*.patch` + Dockerfile apply** | 上游多处小改（sed 太脆 / wrapper 覆盖不了源码内部路径），且改动系统性、跨文件 | 补丁明确、可 `git diff` 审 | 上游 SHA 升级要 rebase 补丁；补丁越多越脆 | [genie3-server/patches/](../../services/genie3-server/patches/) |
| **C. runtime monkey-patch**（`importlib` + 属性替换） | 上游内部函数有环境相关 bug（如 `os.getenv` 期望 CWD 里有某文件）；且**不能** patch 源码（例如上游用了 `__file__` 相对路径依赖的二进制） | 上游源码不动，patch 逻辑在 server 代码可测 | 多一个 python 层；patch 依赖上游模块结构，重构就失效 | [deeprank-ab-server/run_inference.py](../../services/deeprank-ab-server/) |

**默认走 A**。只有当"参数化 + 前后处理"覆盖不到的深层修改时才升级到 B，
只有"上游代码不能被物理搬动"（例如二进制资源的相对路径）时才用 C。

**A 的常见 wrapper 模板**：

```python
# services/<svc>/inference.py
import argparse
import sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    # ... 参数化上游硬编码的路径 / 关掉训练期专属选项
    return p.parse_args()

def validate(args):
    # 权重 / 输入存在性 + 后缀检查 → 早失败，错误信息干净落到 stderr 尾
    ...

def main():
    args = parse_args()
    validate(args)
    # **关键：重 import 上游模块必须在 validate() 之后**，避免因为路径/权重
    # 缺失，用户看到的是 "ModuleNotFoundError: 上游库缺 CUDA 扩展" 而不是
    # "checkpoint not found: ..."
    from <upstream_pkg> import ...
    ...

if __name__ == "__main__":
    sys.exit(main())
```

**deferred import 是 A 的核心**：上游库常有 heavy import（pytorch + PyG + rdkit +
openbabel），如果 `import` 在文件顶端，任何输入校验错误都会先付一次 20-30s 的
import 成本；把 import 放到 `validate()` 之后，参数错时错误消息秒回。

**B 的操作方式**（放 Dockerfile 而非 vendor.sh）：

```dockerfile
COPY services/<svc>/upstream /opt/<svc>/upstream
COPY services/<svc>/patches /tmp/<svc>-patches
RUN cd /opt/<svc>/upstream && for p in /tmp/<svc>-patches/*.patch; do \
        echo "Applying $p"; patch -p1 < "$p"; \
    done && rm -rf /tmp/<svc>-patches
```

不要把 patch 应用放进 vendor.sh —— 那样改 patch 必须重 vendor（慢 + 网络）。
放 Dockerfile 让 patch 迭代只触发一层 rebuild。

#### Conda / micromamba 替代骨架

当上游依赖**必须**通过 conda 管理（例如 PyTorch + PyG 组合、CUDA 版本特殊限制、`environment*.yml` 中有 conda-only 包），应使用 **micromamba + multistage** 构建，而非 uv venv。典型场景：DeepRank-Ab（Python 3.9 + PyTorch 2.0.1 + CUDA 11.8 + PyG）、PPIFlow。

```dockerfile
# ---- builder ----
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

# Aliyun pip mirror + TUNA conda mirror（按需选择国内镜像）
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN micromamba config append channels conda-forge \
    && micromamba config set channel_alias https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud

# micromamba + conda env from vendored upstream yml
# (vendor.sh 必须先在 host 上跑过，产物在 services/<svc>/upstream/)
RUN ... install micromamba ...
COPY services/<svc>/upstream/environment*.yml /tmp/
RUN micromamba env create -n <env-name> -f /tmp/environment*.yml

# Additional pip deps not in yml
RUN micromamba run -n <env-name> pip install anarci  # example

# Algorithm source + sed patches (if any)
COPY services/<svc>/upstream /opt/<algo>
RUN sed -i 's/NUM_WORKERS = 96/NUM_WORKERS = 8/' /opt/<algo>/scripts/inference.py  # example

# Framework
COPY services/_framework /tmp/service-framework
RUN micromamba run -n <env-name> pip install "/tmp/service-framework[mcp]" httpx alibabacloud-oss-v2

# ---- runtime ----
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
COPY --from=builder /opt/conda/envs/<env-name> /opt/conda/envs/<env-name>
COPY --from=builder /opt/<algo> /opt/<algo>
COPY services/<svc>/ /opt/<algo>/server/

WORKDIR /opt/<algo>
ENV PYTHONPATH=/opt/<algo>
ENV PATH="/opt/conda/envs/<env-name>/bin:$PATH"
ENV <SVC>_PYTHON=/opt/conda/envs/<env-name>/bin/python
ENV <SVC>_WEIGHTS_DIR=/data/models/<svc>   # NAS 挂载，不烘焙
ENV <SVC>_OSS_OUTPUT_MOUNT=/mnt/oss        # output-sink 挂载点（见 uv 骨架注释）
# ... other <SVC>_ env vars ...

# **必备**：FC 容器默认 LANG 未设置，Python 的 locale.getpreferredencoding()
# 返回 'ascii'。任何用 `open()` 无 encoding= 读带 UTF-8 字符的文件的 upstream
# 代码都会崩 UnicodeDecodeError。设成 C.UTF-8 关掉整类问题。
# 详见 conda-pitfalls.md（包装 conda-based upstream 的常见陷阱）。
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

CMD ["python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "9000", "--timeout-keep-alive", "900"]
```

**何时选 conda**：(1) 上游提供 `environment*.yml` 且包含 pytorch/pyg/nvidia channels；(2) Python 版本 < 3.10（uv wheel 生态不完整）；(3) 依赖 conda-only 包（如某些 bioconda 工具）。其余情况优先用 uv venv 骨架。

