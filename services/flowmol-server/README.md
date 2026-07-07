# flowmol-server

基于 FastAPI 的 [FlowMol3](https://github.com/Dunni3/FlowMol) HTTP 服务——把
flow matching 的 **无条件 3D 小分子生成**功能包装成 FC GPU 服务。**构建在
[bioagent-service-framework](../_framework/) 之上**。

FlowMol3：flow matching 模型，从 $\mathcal{N}(0, I)$ 先验直接生成 3D 全原子
（含 H）的 valid drug-like 小分子（原子坐标 + 类型 + 电荷 + 键序）。**6M
参数**在 GEOM-Drugs benchmark 上全面 SOTA（%Valid 99.9, %PB-Valid 91.9）——
arXiv:2508.12629，MIT license，Koes Lab @ University of Pittsburgh。

镜像 base：`nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`；conda env
(Python 3.10 + pytorch 2.2 + cu121 + dgl 2.0.0.cu121 + PyG scatter/cluster +
rdkit)。

## 架构

```
客户端 / Agent
  ↓ HTTP (form-encoded params, no file uploads — unconditional)
┌────────────────────────────────────────────────────────────────┐
│  FastAPI + bioagent-service-framework  (port 9000)             │
│                                                                │
│  服务专属                                                       │
│    POST /api/generate         (submit/poll)                    │
│    POST /api/tasks/generate   (FC async task mode)             │
│                                                                │
│  框架统一提供                                                    │
│    GET  /api/manifest                                          │
│    GET  /openapi.json                                          │
│    GET  /healthz, /healthz/detail   (后者带权重就位探针)         │
│    GET  /api/jobs/{id}/...                                     │
└────────────────────────────────────────────────────────────────┘
  ↓ subprocess (cwd=/opt/flowmol, 单卡串行)
inference.py → FlowMol.load_from_checkpoint(...) → sample_random_sizes()
  ↓ 输出
NAS: <jobs_base_dir>/<job_id>/{output/{molecules.sdf, sampling_stats.json}, logs/run.log, job.json}
```

## API 速览

### `POST /api/generate`

application/x-www-form-urlencoded 或 multipart/form-data（**无文件上传**）：

| 字段 | 类型 | 默认 | 约束 | 说明 |
|---|---|---|---|---|
| `n_mols` | int | `100` | `1–1000` | 生成分子数量 |
| `n_timesteps` | int | `250` | `50–500` | Euler 积分步数 |
| `n_atoms_per_mol` | int? | `null` | `5–100` | 固定原子数（null = 按训练分布采样） |
| `model_variant` | enum | `flowmol3` | 22 选 1 | 见下表 |
| `seed` | int? | `null` | — | pytorch-lightning seed |
| `stochasticity` | float? | `null` | ≥0 | CTMC 采样 stochasticity |
| `hc_thresh` | float? | `null` | `[0, 1]` | CTMC high-confidence threshold |
| `max_batch_size` | int | `128` | `1–512` | 内部 batching 上限 |

**模型变体**（22 个 FlowMol3 pretrained checkpoints，见
[flowmol/trained_models/readme.md](https://github.com/Dunni3/FlowMol/blob/main/flowmol/trained_models/readme.md)）：

| 类别 | 变体 | 用途 |
|---|---|---|
| **主模型** | `flowmol3`（default） | 论文 SOTA，生产用 |
| Self-correction 消融 | `fm3_nodistort`, `fm3_none` | 无 distortion / 无三大 correction |
| Loss weight 消融 | `fm3_ahigh` / `alow` / `chigh` / `clow` / `ehigh` / `elow` / `xhigh` / `xlow` | atom-type / charge / bond / position 4 特征扫描 |
| Distortion 参数 | `fm3_distort_extreme`, `_highp`, `_hight`, `_lowp`, `_lowt` | geometry distortion p/t 扫描 |
| Fake-atom | `fm3_fa_highp`, `_highstd`, `_lowp`, `_lowstd` | 假原子概率/噪声扫描 |
| Self-conditioning | `fm3_scprop_high`, `fm3_scprop_low` | SC 比例扫描 |

**默认 pre-stage 到 NAS 的只有前 4 个主要变体**；其他消融变体按需通过
`fetch_weights.sh --variants` 手工 stage（见 [Weights](#weights) 章节）。请求
使用未 pre-stage 的变体会在 `inference.py` 校验阶段 fail-fast（exit 2）。
通过 `/healthz/detail` 的 `staged_variants` 查看当前可用集合。

示例：

```bash
# basic — 100 mols, default flowmol3, ~30-60 s on T4
curl -X POST $URL/api/generate -F n_mols=100 -F n_timesteps=250
# → { "job_id": "...", "status": "pending", ... }

# fast smoke — 10 mols, ~5-10 s
curl -X POST $URL/api/generate -F n_mols=10 -F n_timesteps=100

# fixed size
curl -X POST $URL/api/generate -F n_mols=50 -F n_atoms_per_mol=25 -F seed=42

# ablation variant (must be pre-staged)
curl -X POST $URL/api/generate -F n_mols=100 -F model_variant=fm3_nodistort

# poll
curl $URL/api/jobs/<job_id>
# completed →
curl -O $URL/api/jobs/<job_id>/file/molecules.sdf
curl $URL/api/jobs/<job_id>/file/sampling_stats.json
```

### `POST /api/tasks/generate`

FC async task mode（HTTP 立即返回 202，FC 把请求和计算的生命周期绑在一起）。
控制台需要打开「异步任务模式」。详见 [FC 异步任务模式设计](../../engineering/decisions/2026-06-17-fc-async-task-mode.md)。

## 输出

```
<jobs_base_dir>/<job_id>/
├── output/
│   ├── molecules.sdf          # 单一 SDF, N ≤ n_mols 个 valid mols (kekulize=False)
│   └── sampling_stats.json    # {n_requested, n_written, invalid_count, sampling_time_seconds, ...}
├── logs/
│   └── run.log
└── job.json
```

**`n_written < n_mols`** 是正常的：sampling 会产生一些无法装配成 valid RDKit
Mol 的图（FlowMol3 论文 SOTA %Valid 99.9%，所以在 100 mols 里通常 100 或 99
mol valid）。`n_written == 0` 会让服务返回 rc=1 + `status=failed`。

## 配置

`pydantic-settings`，`FLOWMOL_` 前缀。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `FLOWMOL_JOBS_BASE_DIR` | `/data/flowmol_jobs` | NAS 上的 job 根目录 |
| `FLOWMOL_ROOT` | `/opt/flowmol` | upstream 源码根（subprocess cwd） |
| `FLOWMOL_PYTHON` | `/opt/conda/envs/flowmol/bin/python` | conda env Python |
| `FLOWMOL_INFERENCE_SCRIPT` | `/opt/flowmol/server/inference.py` | 我们的包装脚本（不走上游 `test.py`） |
| `FLOWMOL_WEIGHTS_DIR` | `/data/models/flowmol` | NAS 根目录，含 `trained_models/<variant>/` 子目录 |
| `FLOWMOL_DEFAULT_VARIANT` | `flowmol3` | fallback variant |
| `FLOWMOL_MAX_CONCURRENT_JOBS` | `1` | 单卡串行 |
| `FLOWMOL_OSS_REGION` | `cn-hangzhou` | 保留字段（v0.0.1 无 URI 输入） |
| `FLOWMOL_SESSION_HEADER_NAME` | `bioagent-session-id` | FC session affinity header |

## Weights

22 个变体的 checkpoints **不 baked 到镜像**，全部从 NAS 加载。每个变体是
`<weights_dir>/trained_models/<variant>/` 下一个目录，含 `checkpoints/last.ckpt`
+ `config.yaml`（**两个文件都必须一起 stash**——upstream `FlowMol.load_from_
checkpoint` 只需 ckpt，但我们的 `/healthz/detail` 探针会检查 config.yaml 是否
一并到位，避免"就位一半"的部署错误）。

期望布局：

```
/data/models/flowmol/
└── trained_models/
    ├── flowmol3/
    │   ├── checkpoints/last.ckpt          ~13 MB
    │   └── config.yaml
    ├── fm3_nodistort/
    │   ├── checkpoints/last.ckpt
    │   └── config.yaml
    ├── fm3_none/
    │   ├── checkpoints/last.ckpt
    │   └── config.yaml
    └── fm3_ahigh/                         # 其他 18 个消融变体按需 pre-stage
        ├── checkpoints/last.ckpt
        └── config.yaml
```

### Pre-stage（一次性）

```bash
# 1. 下载 4 个 primary 到 stage 目录
./services/flowmol-server/scripts/fetch_weights.sh
# → services/flowmol-server/weights/trained_models/{flowmol3,fm3_nodistort,fm3_none,fm3_ahigh}/

# 2. rsync 到 NAS
rsync -av services/flowmol-server/weights/trained_models/ \
    <NAS-mount>:/data/models/flowmol/trained_models/

# 或者直接下到 NAS
WEIGHTS_DST=/mnt/nas/data/models/flowmol \
    ./services/flowmol-server/scripts/fetch_weights.sh

# 下全部 22 个变体（~500 MB）
FLOWMOL_VARIANTS=all \
    WEIGHTS_DST=/mnt/nas/data/models/flowmol \
    ./services/flowmol-server/scripts/fetch_weights.sh

# 只下指定子集
FLOWMOL_VARIANTS="flowmol3,fm3_fa_highp,fm3_scprop_high" \
    ./services/flowmol-server/scripts/fetch_weights.sh
```

### FC

NAS 自动挂载到 `/data/models/flowmol/`。验证：

```bash
curl https://fc-flowmol-XXX.cn-hangzhou-vpc.fcapp.run/healthz/detail
# 期望：{"status":"ok","weights_loaded":true,"weights_missing":{}, "staged_variants":["flowmol3","fm3_ahigh","fm3_nodistort","fm3_none"]}
```

### SIF / HPC

```bash
apptainer run --nv \
    --bind /scratch/models/flowmol:/data/models/flowmol \
    flowmol-server.sif python -m server generate \
    --n-mols 100 --n-timesteps 250 \
    --output-dir results/
```

详见 [weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)。

## Docker 构建与运行

```bash
# 1. Vendor 上游源（一次性；重跑可升级 pinned SHA）
./services/flowmol-server/scripts/vendor.sh

# 2. 构建（~3 GB 镜像，无权重）
make build-flowmol-server

# 本地运行（需 GPU + NAS / 本地 --bind 注入权重）
docker run --gpus all -p 9000:9000 --memory 16g \
    -v $(pwd)/services/flowmol-server/weights:/data/models/flowmol:ro \
    flowmol-server
```

构建上下文必须是项目根目录（Dockerfile 同时 `COPY services/_framework`）。

## CLI 批处理模式

```bash
docker run --rm \
    -v /data:/data \
    -v /path/to/weights:/data/models/flowmol:ro \
    flowmol-server \
    /opt/conda/envs/flowmol/bin/python -m server generate \
    --n-mols 100 --n-timesteps 250 \
    --model-variant flowmol3 \
    --output-dir /data/results/

# 复杂参数可以用 --params-json
python -m server generate \
    --params-json '{"n_mols": 50, "n_timesteps": 150, "seed": 42, "n_atoms_per_mol": 25}' \
    --output-dir ./output/
```

Slurm sbatch 范式见 [apptainer-compatibility.md](../../engineering/guides/apptainer-compatibility.md)。

## 离线测试

```bash
# 单元测试（无 GPU、无权重；subprocess stub /bin/true）
uv run python -m pytest services/flowmol-server/tests/test_app.py -v
uv run python -m pytest services/flowmol-server/tests/test_cli.py -v

# FC 集成测试（部署后）
RUN_FC_TESTS=1 uv run python -m pytest -m fc \
    services/flowmol-server/tests/test_fc.py -v

# FC 异步任务模式测试（控制台开启异步任务模式后）
RUN_FC_TESTS=1 uv run python -m pytest -m fc \
    services/flowmol-server/tests/test_fc_task.py -v
```

## 阿里云函数计算部署

| 配置项 | 推荐值 |
|---|---|
| 运行时 | 自定义容器镜像 |
| GPU 配置 | `fc.gpu.tesla.1`（T4 8 GB 绰绰有余——6M 参数模型）/ Ada / L20 |
| 监听端口 | `9000` |
| 函数超时 | ≥ 600 秒（覆盖 n_mols=1000 + n_timesteps=500 的最长采样） |
| 内存 | `16384` MB |
| CPU | `4` vCPU |
| 磁盘 | `10240` MB |
| NAS 挂载 | `/fc → /data`（jobs + 权重 `/data/models/flowmol/`） |
| 异步任务模式 | **启用**（解锁 `/api/tasks/generate`） |
| Keepalive URL | 清空（走 task endpoint，不需要 legacy keepalive） |

详细 deploy + 异步任务模式控制台配置见
[新增服务 cookbook](../../engineering/guides/adding-a-new-service.md) 的
"部署到 FC" 和 "FC 异步任务模式控制台配置" 章节。

## 相关文档

- [flowmol-server 设计](../../engineering/decisions/2026-07-06-flowmol-server-design.md)
- [FlowMol3 wiki](../../wiki/small-molecule-design/flowmol3.md) — 上游算法 + 三个 self-correction 技巧详解
- [Service 框架抽象设计](../../engineering/decisions/2026-05-12-service-framework-design.md)
- [Weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)
- [新增服务 cookbook](../../engineering/guides/adding-a-new-service.md)
- 上游：[Dunni3/FlowMol](https://github.com/Dunni3/FlowMol)（MIT）
- 论文：[Dunn & Koes, arXiv:2508.12629 (2025)](https://arxiv.org/abs/2508.12629)
