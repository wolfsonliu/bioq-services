# diffusion-hopping-server

基于 FastAPI 的 [DiffHopp](https://github.com/jostorge/diffusion-hopping)
HTTP 服务——把图扩散模型的 **scaffold hopping**（候选骨架生成）功能包装成 FC
GPU 服务。**构建在 [bioq-service-framework](../_framework/) 之上**。

DiffHopp：给定**蛋白口袋** + **参考配体**，生成 N 个保留结合姿势但骨架不同的
候选分子（arXiv:2308.07416，MIT license）。

镜像 base：`nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`；conda env
(Python 3.10 + pytorch 1.13.1 + cu117 + PyG 2.2 + rdkit + openbabel + reduce)。

## 架构

```
客户端 / Agent
  ↓ HTTP (multipart upload: protein.pdb + reference_ligand.sdf)
┌────────────────────────────────────────────────────────────────┐
│  FastAPI + bioq-service-framework  (port 9000)             │
│                                                                │
│  服务专属                                                      │
│    POST /api/generate         (submit/poll)                    │
│    POST /api/tasks/generate   (FC async task mode)             │
│                                                                │
│  框架统一提供                                                  │
│    GET  /api/manifest                                          │
│    GET  /openapi.json                                          │
│    GET  /healthz, /healthz/detail   (后者带权重就位探针)       │
│    GET  /api/jobs/{id}/...                                     │
└────────────────────────────────────────────────────────────────┘
  ↓ subprocess (cwd=/opt/diffusion-hopping, 单卡串行)
inference.py → ObabelTransform + ReduceTransform → DiffHopp.sample()
  ↓ 输出
NAS: <jobs_base_dir>/<job_id>/{input/, output/output_<i>.sdf, logs/run.log, job.json}
```

## API 速览

### `POST /api/generate`

multipart/form-data：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `protein` | file (.pdb) | ✓(或 `protein_uri`) | — | 蛋白口袋 |
| `reference_ligand` | file (.sdf / .mol2 / .pdb) | ✓(或 `reference_ligand_uri`) | — | 参考配体 |
| `num_samples` | int (1–100) | — | `10` | 生成的候选数量（一次 batched sampling） |
| `model_variant` | enum | — | `gvp_conditional` | 见下表 |

4 个模型变体：

| 变体 | 论文里叫 | 何时用 |
|---|---|---|
| `gvp_conditional` | DiffHopp（main） | **默认**；保留参考配体的 functional-group conditioning |
| `gvp_unconditional` | DiffHopp inpainting | 想让模型 ignore 参考配体的官能团（更纯 de novo） |
| `egnn_conditional` | DiffHopp-EGNN | EGNN backbone 变体，conditional |
| `egnn_unconditional` | DiffHopp-EGNN inpainting | EGNN backbone + inpainting |

示例：

```bash
curl -X POST $URL/api/generate \
    -F protein=@1abc_pocket.pdb \
    -F reference_ligand=@ref.sdf \
    -F num_samples=10 \
    -F model_variant=gvp_conditional
# → { "job_id": "...", "status": "pending", ... }

# poll
curl $URL/api/jobs/<job_id>
# completed 后：
curl $URL/api/jobs/<job_id>/files
# → ["output/output_0.sdf", "output/output_1.sdf", ...]
curl -O $URL/api/jobs/<job_id>/file/output/output_0.sdf
```

### `POST /api/tasks/generate`

同步阻塞 + FC async task mode（HTTP 立即返回 202，FC 把请求和计算的生命
周期绑在一起）。控制台需要打开「异步任务模式」。

详见 [FC 异步任务模式设计](../../engineering/decisions/2026-06-17-fc-async-task-mode.md)。

## 配置

`pydantic-settings`，`DIFFUSION_HOPPING_` 前缀。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `DIFFUSION_HOPPING_JOBS_BASE_DIR` | `/data/diffusion_hopping_jobs` | NAS 上的 job 根目录 |
| `DIFFUSION_HOPPING_ROOT` | `/opt/diffusion-hopping` | upstream 源码根（subprocess cwd） |
| `DIFFUSION_HOPPING_PYTHON` | `/opt/conda/envs/diffusion_hopping/bin/python` | conda env Python |
| `DIFFUSION_HOPPING_INFERENCE_SCRIPT` | `/opt/diffusion-hopping/server/inference.py` | 我们的包装脚本（不是上游的 `generate_scaffolds.py`） |
| `DIFFUSION_HOPPING_WEIGHTS_DIR` | `/data/models/diffusion-hopping/checkpoints` | 4 个 .ckpt（NAS 挂载） |
| `DIFFUSION_HOPPING_MAX_CONCURRENT_JOBS` | `1` | 单卡串行 |
| `DIFFUSION_HOPPING_OSS_REGION` | `cn-hangzhou` | OSS URI 下载区域 |
| `DIFFUSION_HOPPING_SESSION_HEADER_NAME` | `bioagent-session-id` | FC session affinity header |

## Weights

4 个 checkpoint（~189 MB 总）**不再 baked 到镜像**，从 NAS 加载。
（虽然上游 git 自带 ckpts，我们按项目约定统一走 NAS——便于独立打版本）。

期望布局：

```
/data/models/diffusion-hopping/
└── checkpoints/
    ├── gvp_conditional.ckpt      ~63 MB   DiffHopp main
    ├── gvp_unconditional.ckpt    ~52 MB   DiffHopp inpainting
    ├── egnn_conditional.ckpt     ~5 MB    DiffHopp-EGNN
    └── egnn_unconditional.ckpt   ~69 MB   DiffHopp-EGNN inpainting
```

### Pre-stage（一次性）

```bash
# 1. Vendor 上游源码（含 4 个 ckpts）
./services/diffusion-hopping-server/scripts/vendor.sh

# 2. 把 ckpts 从 upstream/ cp 到 stage 目录
./services/diffusion-hopping-server/scripts/fetch_weights.sh
# → services/diffusion-hopping-server/weights/

# 3. 上传到 NAS
rsync -av services/diffusion-hopping-server/weights/ \
    <NAS-mount>:/data/models/diffusion-hopping/checkpoints/
```

或直接 cp 到 NAS：

```bash
./services/diffusion-hopping-server/scripts/vendor.sh
WEIGHTS_DST=/mnt/nas/data/models/diffusion-hopping/checkpoints \
    ./services/diffusion-hopping-server/scripts/fetch_weights.sh
```

### FC

NAS 自动挂载到 `/data/models/diffusion-hopping/`。验证：

```bash
curl https://fc-diffusion-hopping-XXX.cn-hangzhou-vpc.fcapp.run/healthz/detail
# 期望：{"status":"ok","weights_loaded":true,"weights_missing":{}}
```

### SIF / HPC

```bash
apptainer run --nv \
    --bind /scratch/models/diffusion-hopping:/data/models/diffusion-hopping \
    diffusion-hopping-server.sif python -m server generate \
    --protein 1a0q.pdb \
    --reference-ligand ref.sdf \
    --output-dir results/
```

详见 [weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)。

## Docker 构建与运行

```bash
# 1. Vendor 上游源（一次性，重跑可升级 SHA）
./services/diffusion-hopping-server/scripts/vendor.sh

# 2. 构建（~3 GB 镜像，无权重）
make build-diffusion-hopping-server

# 本地运行（需 GPU + NAS / 本地 --bind 注入权重）
docker run --gpus all -p 9000:9000 --memory 16g \
    -v $(pwd)/services/diffusion-hopping-server/weights:/data/models/diffusion-hopping/checkpoints:ro \
    diffusion-hopping-server
```

构建上下文必须是项目根目录（Dockerfile 同时 `COPY services/_framework`）。

## CLI 批处理模式

```bash
docker run --rm \
    -v /data:/data \
    -v /path/to/ckpts:/data/models/diffusion-hopping/checkpoints:ro \
    diffusion-hopping-server \
    .venv/bin/python -m server generate \
    --protein /data/input/pocket.pdb \
    --reference-ligand /data/input/ref.sdf \
    --output-dir /data/results/ \
    --params-json '{"num_samples": 10, "model_variant": "gvp_conditional"}'
```

Slurm sbatch 范式见 [apptainer-compatibility.md](../../engineering/guides/apptainer-compatibility.md)。

## 离线测试

```bash
# 单元测试（无 GPU、无权重；subprocess stub /bin/true）
uv run python -m pytest services/diffusion-hopping-server/tests/test_app.py -v
uv run python -m pytest services/diffusion-hopping-server/tests/test_cli.py -v

# FC 集成测试（部署后）
RUN_FC_TESTS=1 uv run python -m pytest -m fc \
    services/diffusion-hopping-server/tests/test_fc.py -v
```

## 阿里云函数计算部署

| 配置项 | 推荐值 |
|---|---|
| 运行时 | 自定义容器镜像 |
| GPU 配置 | `fc.gpu.tesla.1`（T4 8 GB 即可——模型不大，~70 MB ckpt）/ Ada / Blackwell 都可 |
| 监听端口 | `9000` |
| 函数超时 | ≥ 600 秒（diffusion sampling 单批次） |
| 内存 | `16384` MB |
| CPU | `4` vCPU |
| 磁盘 | `10240` MB |
| NAS 挂载 | `/fc → /data`（jobs + 权重 `/data/models/diffusion-hopping/`） |
| 异步任务模式 | **启用**（解锁 `/api/tasks/generate`） |

详细 deploy + 异步任务模式控制台配置见
[新增服务 cookbook](../../engineering/guides/adding-a-new-service.md)
的"部署到 FC"和"FC 异步任务模式控制台配置"章节。

## 相关文档

- [Service 框架抽象设计](../../engineering/decisions/2026-05-12-service-framework-design.md)
- [Weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)
- [新增服务 cookbook](../../engineering/guides/adding-a-new-service.md)
- 上游：[jostorge/diffusion-hopping](https://github.com/jostorge/diffusion-hopping)（MIT）
