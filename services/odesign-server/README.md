# ODesign Server

基于 FastAPI 的 [ODesign](https://github.com/OTeam-AI4S/ODesign) HTTP 服务，
封装跨模态生物分子设计（蛋白 / 配体 / 核酸 binder + motif/atom scaffolding）。
**构建在 [bioagent-service-framework](../_framework/) 之上**。

镜像 base：`nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`；conda env (Python
3.10) + torch 2.3.1+cu121 + PyG + CUTLASS (DS4Sci_EvoformerAttention + fast
layernorm)。

## API 速览

```
POST /api/design           cross-modality biomolecular design
POST /api/tasks/design     async task mode 版本

GET  /api/manifest         Agent 协议描述
GET  /healthz, /healthz/detail
GET  /api/jobs/{id}/...    JobInfo / files / log / download
```

## 配置

`pydantic-settings`，`ODESIGN_` 前缀。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `ODESIGN_JOBS_BASE_DIR` | `/data/odesign_jobs` | NAS 上的 job 根目录 |
| `ODESIGN_ROOT` | `/opt/odesign/ODesign` | ODesign 源码根 |
| `ODESIGN_PYTHON` | `/opt/conda/envs/odesign/bin/python` | conda env Python |
| `ODESIGN_INFERENCE_SCRIPT` | `/opt/odesign/ODesign/scripts/inference.py` | Hydra 入口 |
| `ODESIGN_CKPT_ROOT_DIR` | `/data/models/odesign/ckpt` | 模型权重目录（NAS 挂载，v0.0.5 起外置） |
| `ODESIGN_DATA_ROOT_DIR` | `/data/models/odesign/data` | CCD 数据目录（NAS 挂载，v0.0.5 起外置） |
| `CUTLASS_PATH` | `/kernels/cutlass` | CUTLASS header-only library（镜像内）|
| `ODESIGN_MAX_CONCURRENT_JOBS` | `1` | 单卡串行 |
| `ODESIGN_OSS_REGION` | `cn-hangzhou` | OSS URI 下载区域 |

## Weights

v0.0.5 起所有权重 + CCD 数据（~537 MB）**不再 baked 到镜像**，从 NAS 加载。
期望布局：

```
/data/models/odesign/
├── ckpt/                                     ← HF 模型检查点 + grnade.h5
│   ├── odesign_base_prot_flex.pt
│   ├── odesign_base_prot_rigid.pt
│   ├── odesign_base_ligand_rigid.pt
│   ├── odesign_base_na_rigid.pt
│   ├── oinvfold_{protein,ligand,dna,rna}.ckpt
│   ├── v_48_020.pt                           ← ProteinMPNN
│   └── grnade.h5                             ← upstream-tracked，从 vendored 源 cp
└── data/                                     ← CCD molecule data
    ├── components.v20240608.cif              ← ~1.7 GB
    └── components.v20240608.cif.rdkit_mol.pkl ← ~300 MB
```

### Pre-stage（一次性）

**HF 模型 + grnade.h5**（脚本自动从 vendored upstream 取 grnade.h5）：

```bash
# 先 vendor 上游（提供 grnade.h5）
./services/odesign-server/scripts/vendor.sh

# 下载 HF 模型 + 复制 grnade.h5 到 stage 目录
./services/odesign-server/scripts/fetch_weights.sh
# → services/odesign-server/weights/ckpt/

# 或直接到 NAS
./services/odesign-server/scripts/fetch_weights.sh /mnt/nas/data/models/odesign/ckpt
```

**CCD 数据**（需手动从 Google Drive 下载）：

```bash
# 脚本打印 Google Drive 链接，下载 2 个文件到 stage 目录
./services/odesign-server/scripts/fetch_ccd_data.sh

# 上传到 NAS
rsync -av services/odesign-server/weights/{ckpt,data}/ \
    <NAS-mount>:/data/models/odesign/{ckpt,data}/
```

### FC

NAS 自动挂载到 `/data/models/odesign/`。验证：

```bash
curl https://fc-odesign-???.cn-hangzhou-vpc.fcapp.run/healthz/detail
# 期望：{"status":"ok","weights_loaded":true,"weights_missing":{}}
```

### SIF / HPC

```bash
apptainer run --nv \
    --bind /scratch/models/odesign:/data/models/odesign \
    odesign-server.sif python -m server design ...
```

详见 [weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)。

## Docker 构建

```bash
# 1. Vendor 上游源（一次性）
./services/odesign-server/scripts/vendor.sh

# 2. 构建（~2.5 GB 镜像，无权重）
make build-odesign-server

# 本地运行（需 GPU + NAS / 本地 --bind）
docker run --gpus all -p 9000:9000 \
    -v $(pwd)/weights:/data/models/odesign \
    odesign-server
```

构建上下文必须是项目根目录（Dockerfile 同时 `COPY services/_framework`）。

## 相关文档

- [Service 框架抽象设计](../../engineering/decisions/2026-05-12-service-framework-design.md)
- [Weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)
- [odesign-server 设计文档](../../engineering/decisions/2026-05-26-odesign-server-design.md)
