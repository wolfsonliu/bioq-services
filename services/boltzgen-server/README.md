# BoltzGen Server

基于 FastAPI 的 [BoltzGen](https://github.com/HannesStark/boltzgen) HTTP 服务，
封装 binder design 流水线（扩散生成 + inverse fold + refolding + filtering）。
**构建在 [bioq-service-framework](../_framework/) 之上**：HTTP / job /
错误 / 持久化由框架统一提供。

镜像 base：`nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`；torch 2.4.1 cu124
(HPC 驱动兼容性所需，详见 Dockerfile 注释)。

## API 速览

```
POST /api/design          binder design pipeline（设计目标 + 多步生成）
POST /api/inverse_fold    单步 inverse folding（结构 → 序列）
GET  /api/manifest        Agent 协议描述
GET  /healthz, /healthz/detail
GET  /api/jobs/{id}/...   JobInfo / files / log / download
```

## 配置

`pydantic-settings`，`BOLTZGEN_` 前缀。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `BOLTZGEN_JOBS_BASE_DIR` | `/data/boltzgen_jobs` | NAS 上的 job 根目录 |
| `BOLTZGEN_ROOT` | `/opt/boltzgen` | BoltzGen 安装根（subprocess cwd）|
| `BOLTZGEN_PYTHON` | `/opt/boltzgen/.venv/bin/python` | venv 解释器 |
| `BOLTZGEN_CLI` | `/opt/boltzgen/.venv/bin/boltzgen` | BoltzGen CLI |
| `BOLTZGEN_WEIGHTS_DIR` | `/data/models/boltzgen/weights` | 模型权重目录（NAS 挂载，5 个 .ckpt ~10 GB）|
| `BOLTZGEN_MOLDIR` | `/data/models/boltzgen/moldir` | CCD molecule library（NAS 挂载，~6 GB）|
| `BOLTZGEN_MAX_CONCURRENT_JOBS` | `1` | 单卡串行 |
| `BOLTZGEN_OSS_REGION` | `cn-hangzhou` | OSS URI 下载区域 |

## Weights

v0.0.10 起 5 个 checkpoint (~10 GB) + CCD moldir (~6 GB) **不再 baked 到镜像**，
从 NAS 加载。期望布局：

```
/data/models/boltzgen/
├── weights/
│   ├── boltzgen1_diverse.ckpt
│   ├── boltzgen1_adherence.ckpt
│   ├── boltzgen1_ifold.ckpt
│   ├── boltz2_conf_final.ckpt
│   └── boltz2_aff.ckpt
└── moldir/
    └── *.pkl                    ← extracted from mols.zip
```

### Pre-stage（一次性）

```bash
# 下载到本地 stage 目录
./services/boltzgen-server/scripts/fetch_weights.sh
# → services/boltzgen-server/{weights,moldir}/

# 上传到 NAS
rsync -av services/boltzgen-server/weights/ <NAS-mount>:/data/models/boltzgen/weights/
rsync -av services/boltzgen-server/moldir/  <NAS-mount>:/data/models/boltzgen/moldir/
```

或直接下到 NAS：

```bash
WEIGHTS_DST=/mnt/nas/data/models/boltzgen/weights \
MOLDIR_DST=/mnt/nas/data/models/boltzgen/moldir \
    ./services/boltzgen-server/scripts/fetch_weights.sh
```

### FC

NAS 自动挂载到 `/data/models/boltzgen/`。验证：

```bash
curl https://fc-boltzgen-karzusdoih.cn-hangzhou-vpc.fcapp.run/healthz/detail
# 期望：{"status":"ok","weights_loaded":true,"weights_missing":{}}
```

### SIF / HPC

```bash
apptainer run --nv \
    --bind /scratch/models/boltzgen:/data/models/boltzgen \
    boltzgen-server.sif python -m server design ...
```

详见 [weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)。

## Docker 构建

```bash
# 1. Vendor 上游源（一次性，重跑可升级 SHA）
./services/boltzgen-server/scripts/vendor.sh

# 2. 构建（~2.5 GB 镜像，无权重）
make build-boltzgen-server

# 本地运行（需 GPU + NAS / 本地 --bind）
docker run --gpus all -p 9000:9000 \
    -v $(pwd)/weights:/data/models/boltzgen/weights \
    -v $(pwd)/moldir:/data/models/boltzgen/moldir \
    boltzgen-server
```

构建上下文必须是项目根目录（Dockerfile 同时 `COPY framework`）。

## 相关文档

- [Service 框架抽象设计](../../engineering/decisions/2026-05-12-service-framework-design.md)
- [Weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)
- [boltzgen-server 设计文档](../../engineering/decisions/2026-05-25-boltzgen-server-design.md)
