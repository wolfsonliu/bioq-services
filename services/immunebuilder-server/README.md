# ImmuneBuilder Server

基于 FastAPI 的 [ImmuneBuilder](https://github.com/brennanaba/ImmuneBuilder)
HTTP 服务，封装免疫受体结构预测（抗体 / 纳米抗体 / TCR 三端点）。**构建在
[bioq-service-framework](../_framework/) 之上**。

底层：ABodyBuilder2 / NanoBodyBuilder2 / TCRBuilder2，conda env (Python 3.10
+ OpenMM + pdbfixer + HMMER) + uv pip (torch 2.7.1 + ANARCI)。

## API 速览

```
POST /api/predict_antibody       H + L 链
POST /api/predict_nanobody       H 链
POST /api/predict_tcr            A + B 链
POST /api/tasks/predict_*        async task mode 版本

GET  /api/manifest               Agent 协议描述
GET  /healthz, /healthz/detail
GET  /api/jobs/{id}/...          JobInfo / files / log / download
```

## 配置

`pydantic-settings`，`IMMUNEBUILDER_` 前缀。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `IMMUNEBUILDER_JOBS_BASE_DIR` | `/data/immunebuilder_jobs` | NAS 上的 job 根目录 |
| `IMMUNEBUILDER_VENV_BIN` | `/opt/conda/envs/immunebuilder/bin` | conda env bin/ |
| `IMMUNEBUILDER_WEIGHTS_DIR` | `/data/models/immunebuilder/trained_model` | 训练模型目录（v0.0.5 起 NAS 挂载） |
| `IMMUNEBUILDER_MAX_CONCURRENT_JOBS` | `1` | 单卡串行 |
| `IMMUNEBUILDER_OSS_REGION` | `cn-hangzhou` | OSS URI 下载区域 |

## Weights

v0.0.5 起 16 个 `.pt` 训练权重（~600 MB）**不再 baked 到镜像**，从 NAS 加载。

⚠️ ImmuneBuilder 上游用包内**相对路径** `ImmuneBuilder/trained_model/`
查找权重。镜像里这条路径**是个 symlink** 指向 NAS 挂载点：

```
/opt/immunebuilder/ImmuneBuilder/ImmuneBuilder/trained_model
    → /data/models/immunebuilder/trained_model        (symlink)
```

NAS 缺失时 symlink target 失效，第一次推理时上游代码 crash；用
`/healthz/detail` 提前发现。

期望布局：

```
/data/models/immunebuilder/
└── trained_model/
    ├── antibody_model_{1..4}      ← ABodyBuilder2
    ├── nanobody_model_{1..4}      ← NanoBodyBuilder2
    ├── tcr_model_{1..4}           ← TCRBuilder2+ (default)
    └── tcr2_model_{1..4}          ← TCRBuilder2 original
```

### Pre-stage（一次性）

从 Zenodo 下载（脚本自动处理 4 个 Zenodo records）：

```bash
# 下载到本地 stage 目录
./services/immunebuilder-server/scripts/fetch_weights.sh
# → services/immunebuilder-server/trained_model/

# 上传到 NAS
rsync -av services/immunebuilder-server/trained_model/ \
    <NAS-mount>:/data/models/immunebuilder/trained_model/
```

或直接下到 NAS：

```bash
WEIGHTS_DST=/mnt/nas/data/models/immunebuilder/trained_model \
    ./services/immunebuilder-server/scripts/fetch_weights.sh
```

### FC

NAS 自动挂载到 `/data/models/immunebuilder/`。验证：

```bash
curl https://fc-immuebuilder-mhxbldkfhc.cn-hangzhou-vpc.fcapp.run/healthz/detail
# 期望：{"status":"ok","weights_loaded":true,"files_found":16}
```

### SIF / HPC

```bash
apptainer run --nv \
    --bind /scratch/models/immunebuilder:/data/models/immunebuilder \
    immunebuilder-server.sif python -m server predict_antibody ...
```

详见 [weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)。

## Docker 构建

```bash
# 1. Vendor 上游源（一次性）
./services/immunebuilder-server/scripts/vendor.sh

# 2. 构建（~2.5 GB 镜像，无权重）
make build-immunebuilder-server

# 本地运行（需 GPU + NAS / 本地 --bind）
docker run --gpus all -p 9000:9000 \
    -v $(pwd)/trained_model:/data/models/immunebuilder/trained_model \
    immunebuilder-server
```

构建上下文必须是项目根目录（Dockerfile 同时 `COPY services/_framework`）。

## 相关文档

- [Service 框架抽象设计](../../engineering/decisions/2026-05-12-service-framework-design.md)
- [Weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)
- [immunebuilder-server 设计文档](../../engineering/decisions/2026-05-27-immunebuilder-server-design.md)
