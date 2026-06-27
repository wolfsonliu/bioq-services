# DeepRank-Ab Server

基于 FastAPI 的 [DeepRank-Ab](https://github.com/haddocking/DeepRank-Ab) HTTP
服务，封装 EGNN + ESM-2 抗体-抗原对接打分。**构建在
[bioagent-service-framework](../_framework/) 之上**。

镜像 base：`nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`；torch + ESM-2 via
conda。

## API 速览

```
POST /api/score           score 1 antibody-antigen complex
GET  /api/manifest        Agent 协议描述
GET  /healthz, /healthz/detail
GET  /api/jobs/{id}/...   JobInfo / files / log / download
```

## 配置

`pydantic-settings`，`DEEPRANK_AB_` 前缀。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `DEEPRANK_AB_JOBS_BASE_DIR` | `/data/deeprank_ab_jobs` | NAS 上的 job 根目录 |
| `DEEPRANK_AB_ROOT` | `/opt/deeprank-ab` | DeepRank-Ab 安装根 |
| `DEEPRANK_AB_PYTHON` | `/opt/conda/envs/deeprank-ab/bin/python` | conda env Python |
| `DEEPRANK_AB_INFERENCE_SCRIPT` | `/opt/deeprank-ab/server/run_inference.py` | 推理包装脚本 |
| `DEEPRANK_AB_WEIGHTS_DIR` | `/data/models/deeprank-ab/esm` | ESM-2 权重目录（NAS 挂载，~2.6 GB 外置）|
| `WEIGHT_PATH` | `<weights_dir>/esm2_t33_650M_UR50D.pt` | ESM-2 主权重（向后兼容）|
| `REG_WEIGHT_PATH` | `<weights_dir>/...contact-regression.pt` | contact regression head |
| `DEEPRANK_AB_NUM_WORKERS` | `4` | DataLoader workers |
| `DEEPRANK_AB_OSS_REGION` | `cn-hangzhou` | OSS URI 下载区域 |

## Weights

v0.0.10 起 ESM-2 权重（~2.6 GB）**不再 baked 到镜像**，从 NAS 加载。
deeprank-ab 有自己单独的 NAS 路径（**不与其它服务共享**），由
`scripts/fetch_esm_weights.sh` 从 Meta AI fair-esm CDN 下载：

```
/data/models/deeprank-ab/
└── esm/
    ├── esm2_t33_650M_UR50D.pt                       ← 主权重
    └── esm2_t33_650M_UR50D-contact-regression.pt    ← contact head
```

### Pre-stage（一次性）

```bash
# 下载到本地 stage 目录
./services/deeprank-ab-server/scripts/fetch_esm_weights.sh
# → services/deeprank-ab-server/weights/esm/

# 上传到 NAS
rsync -av services/deeprank-ab-server/weights/esm/ \
    <NAS-mount>:/data/models/deeprank-ab/esm/
```

或直接下到 NAS：

```bash
WEIGHTS_DST=/mnt/nas/data/models/deeprank-ab/esm \
    ./services/deeprank-ab-server/scripts/fetch_esm_weights.sh
```

### FC

NAS 自动挂载到 `/data/models/deeprank-ab/`。验证：

```bash
curl https://fc-deeprank-ab-lxzlfasfol.cn-hangzhou-vpc.fcapp.run/healthz/detail
# 期望：{"status":"ok","weights_loaded":true,"weights_missing":{}}
```

### SIF / HPC

```bash
apptainer run --nv \
    --bind /scratch/models/deeprank-ab:/data/models/deeprank-ab \
    deeprank-ab-server.sif python -m server score ...
```

详见 [weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)。

## Docker 构建

```bash
# 1. Vendor 上游源（一次性）
./services/deeprank-ab-server/scripts/vendor.sh

# 2. 构建（~2.5 GB 镜像，无权重）
make build-deeprank-ab-server

# 本地运行（需 GPU + NAS / 本地 --bind）
docker run --gpus all -p 9000:9000 \
    -v $(pwd)/esm_weights:/data/models/deeprank-ab/esm \
    deeprank-ab-server
```

构建上下文必须是项目根目录（Dockerfile 同时 `COPY services/_framework`）。

## 相关文档

- [Service 框架抽象设计](../../engineering/decisions/2026-05-12-service-framework-design.md)
- [Weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)
- [deeprank-ab-server 设计文档](../../engineering/decisions/2026-05-24-deeprank-ab-server-design.md)
