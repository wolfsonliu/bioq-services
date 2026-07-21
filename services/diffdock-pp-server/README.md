# diffdock-pp-server

基于 FastAPI 的 [DiffDock-PP](https://github.com/ketatam/DiffDock-PP)
HTTP 服务——把刚性蛋白-蛋白对接（rigid protein-protein docking）功能包装成
FC GPU 服务。**构建在 [bioq-service-framework](../_framework/) 之上**。

DiffDock-PP：给定一对未结合蛋白结构（receptor + ligand，两者都是 protein），
用 e3nn 等变扩散网络采样 N 个候选相对姿势，再用 confidence model 打分排序
（arXiv:2304.03889，MIT license）。

镜像 base：`nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`；conda env
(Python 3.10 + pytorch 1.13.0 + cu117 + PyG 2.2 + e3nn + biopython + fair-esm)。

> **命名说明**：`ligand` / `receptor` 沿用 EquiDock 术语——两者都是蛋白，不是
> 小分子。"ligand" = 被旋转/平移的那个蛋白，"receptor" = 坐标锚点。若需要
> 蛋白-小分子对接，用 [odesign-server](../odesign-server/) 或其它服务。

## 架构

```
客户端 / Agent
  ↓ HTTP (multipart upload: receptor.pdb + ligand.pdb)
┌────────────────────────────────────────────────────────────────┐
│  FastAPI + bioq-service-framework  (port 9000)             │
│                                                                │
│  服务专属                                                      │
│    POST /api/dock              (submit/poll)                   │
│    POST /api/tasks/dock        (FC async task mode)            │
│                                                                │
│  框架统一提供                                                  │
│    GET  /api/manifest                                          │
│    GET  /openapi.json                                          │
│    GET  /healthz, /healthz/detail   (后者带权重就位探针)       │
│    GET  /api/jobs/{id}/...                                     │
└────────────────────────────────────────────────────────────────┘
  ↓ subprocess (cwd=/opt/diffdock-pp, 单卡串行)
inference.py wrapper:
    1. 把 receptor + ligand PDB 复制成 DB5-style pair_{r,l}_b.pdb 布局
    2. in-process 调 upstream main_inf.main() → 生成 N 个采样 pickle
    3. 按 confidence 排序，写 top-K 为 dock_pose_<rank>.pdb
  ↓ 输出
NAS: <jobs_base_dir>/<job_id>/{
    input/{receptor,ligand}.pdb,
    output/{dock_pose_<rank>.pdb, confidence_scores.json, raw_samples.pkl},
    logs/run.log, job.json
}
```

## API 速览

### `POST /api/dock`

multipart/form-data：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `receptor` | file (.pdb) | ✓(或 `receptor_uri`) | — | 受体蛋白 |
| `ligand` | file (.pdb) | ✓(或 `ligand_uri`) | — | 配体蛋白（被旋转/平移的一方） |
| `num_samples` | int (1–200) | — | `40` | reverse diffusion 采样数（论文默认 40） |
| `actual_steps` | int (10–40) | — | `40` | denoise 步数（上游 num_steps=40 + 早停） |
| `top_k` | int (1–40) | — | `5` | 按 confidence 排序输出的姿势数 |
| `use_confidence_model` | bool | — | `true` | false → 跳过 confidence 打分（~30% 更快） |
| `seed` | int \| null | — | `null` | 随机种子；null 时框架自动填 |
| `mirror_ligand` | bool | — | `false` | 上游选项：镜像一半采样（额外多样性） |
| `no_final_noise` | bool | — | `true` | 上游默认；最后一步不加噪 |

示例：

```bash
# 基本调用（默认 40 采样，top-5，含 confidence）
curl -X POST $URL/api/dock \
    -F receptor=@receptor.pdb \
    -F ligand=@ligand.pdb \
    -F num_samples=40 \
    -F top_k=5
# → { "job_id": "...", "status": "pending", ... }

# poll
curl $URL/api/jobs/<job_id>
# 完成后拿姿势 PDB：
curl $URL/api/jobs/<job_id>/files
# → ["output/dock_pose_1.pdb", ..., "output/confidence_scores.json", "output/raw_samples.pkl"]
curl -O $URL/api/jobs/<job_id>/file/output/dock_pose_1.pdb
```

**Fast mode**（不用 confidence model，~30% 更快，无排序信号）：

```bash
curl -X POST $URL/api/dock \
    -F receptor=@receptor.pdb \
    -F ligand=@ligand.pdb \
    -F num_samples=20 -F top_k=5 \
    -F use_confidence_model=false
```

### `POST /api/tasks/dock`

同步阻塞 + FC async task mode（HTTP 立即返回 202，FC 把请求和计算的生命
周期绑在一起）。控制台需要打开「异步任务模式」。

详见 [FC 异步任务模式设计](../../engineering/decisions/2026-06-17-fc-async-task-mode.md)。

## 输出契约

```
<jobs_base_dir>/<job_id>/output/
├── dock_pose_1.pdb          ← top-1（按 confidence 排序；或采样顺序若 use_confidence_model=false）
├── dock_pose_2.pdb          ← top-2
├── ...
├── dock_pose_<top_k>.pdb    ← top-K
├── confidence_scores.json   ← [{rank, confidence, sample_file, note}, ...]
└── raw_samples.pkl          ← 上游原始 pickle（所有 N 个采样，reanalysis / RMSD 打分用）
```

每个 `dock_pose_<rank>.pdb` 是 receptor + ligand 的合成 PDB，两条 chain：`R`（receptor）+ `L`（ligand）。坐标是**中心化坐标**（DiffDock-PP 内部把 receptor mean 减到 0），如果需要世界坐标可用 `raw_samples.pkl` 里的 `graph.original_center` 加回来。

## 配置

`pydantic-settings`，`DIFFDOCK_PP_` 前缀。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `DIFFDOCK_PP_JOBS_BASE_DIR` | `/data/diffdock_pp_jobs` | NAS 上的 job 根目录 |
| `DIFFDOCK_PP_ROOT` | `/opt/diffdock-pp` | upstream 源码根（subprocess cwd） |
| `DIFFDOCK_PP_PYTHON` | `/opt/conda/envs/diffdock_pp/bin/python` | conda env Python |
| `DIFFDOCK_PP_INFERENCE_SCRIPT` | `/opt/diffdock-pp/server/inference.py` | 我们的包装脚本 |
| `DIFFDOCK_PP_CONFIG_YAML` | `/opt/diffdock-pp/server/single_pair_inference.yaml` | 上游论文调优 config（打包进镜像） |
| `DIFFDOCK_PP_WEIGHTS_DIR` | `/data/models/diffdock-pp` | 权重根（NAS 挂载） |
| `DIFFDOCK_PP_MAX_CONCURRENT_JOBS` | `1` | 单卡串行 |
| `DIFFDOCK_PP_OSS_REGION` | `cn-hangzhou` | OSS URI 下载区域 |
| `DIFFDOCK_PP_SESSION_HEADER_NAME` | `bioagent-session-id` | FC session affinity header |
| `TORCH_HOME` | `/data/models/diffdock-pp/esm_cache` | torch.hub 离线缓存（Dockerfile 设置） |

## Weights

DiffDock-PP 有**三块**外部权重，全部走 NAS，**不 baked 到镜像**（见
[Service 权重 NAS 外置化设计](../../engineering/decisions/2026-06-26-service-weights-externalization.md)）：

```
/data/models/diffdock-pp/
├── large_model_dips/
│   ├── args.yaml                                         # 训练超参（模型构造必需）
│   └── fold_0/
│       └── model_best_338669_140_31.084_30.347.pth       # 19 MB — score model
├── confidence_model_dips/
│   ├── args.yaml                                         # 训练超参
│   └── fold_0/
│       └── model_best_0_6_0.241_0.887.pth                # 3 MB — confidence model
└── esm_cache/                                            # ~2.5 GB — ESM-2 residue embedding
    ├── hf_cache/                                         # (HF_HOME 别名，兼容用)
    └── hub/
        ├── checkpoints/
        │   └── esm2_t33_650M_UR50D.pt                    # ~2.5 GB — 主权重
        └── facebookresearch_esm_main/                    # torch.hub source dir
            ├── hubconf.py
            └── esm/                                       # ESM python 包
```

### 预下载 + 上传

```bash
# 1. Vendor upstream（一次性；含 checkpoints/）
./services/diffdock-pp-server/scripts/vendor.sh

# 2. 本地 stage（含 score + confidence + args.yaml + 下载 ESM-2 + clone esm src）
./services/diffdock-pp-server/scripts/fetch_weights.sh
#   → services/diffdock-pp-server/weights/  (~2.7 GB)

# 3. rsync 到 NAS
rsync -av services/diffdock-pp-server/weights/ \
    <NAS>:/data/models/diffdock-pp/

# 也可以直接下到 NAS，跳过本地 stage：
WEIGHTS_DST=/mnt/nas/data/models/diffdock-pp \
    ./services/diffdock-pp-server/scripts/fetch_weights.sh
```

### FC 部署后验证

```bash
curl https://fc-diffdock-pp-XXX.cn-hangzhou-vpc.fcapp.run/healthz/detail
# 期望：{
#   "status": "ok",
#   "weights_dir": "/data/models/diffdock-pp",
#   "weights_loaded": true,
#   "weights_missing": {}
# }
```

如果 `weights_missing` 里有 `esm2_checkpoint` 或 `esm_source_dir`，说明 NAS
上的 `esm_cache/` 没有 stage 完整——回到 §预下载 + 上传 重跑。

### SIF / HPC (Apptainer)

```bash
apptainer run --nv \
    --bind /scratch/models/diffdock-pp:/data/models/diffdock-pp \
    diffdock-pp-server.sif \
    .venv/bin/python -m server dock \
        --receptor /data/input/receptor.pdb \
        --ligand /data/input/ligand.pdb \
        --output-dir /scratch/results/
```

见 [apptainer-compatibility.md](../../engineering/guides/apptainer-compatibility.md)。

## 本地开发

```bash
# 前置：确认 upstream 已 vendor
./services/diffdock-pp-server/scripts/vendor.sh

# lint + offline 单测（不需要 GPU / 权重，subprocess 被 stub）
uvx ruff check services/diffdock-pp-server/
uv run python -m pytest services/diffdock-pp-server/tests/test_app.py -v
uv run python -m pytest services/diffdock-pp-server/tests/test_cli.py -v

# import smoke
uv run python -c "from server.app import app; print(app.title)"
```

## Docker 构建

```bash
# 从项目根：
docker build --platform linux/amd64 \
    -t diffdock-pp-server \
    -f services/diffdock-pp-server/Dockerfile .
```

或用 Makefile：

```bash
make build-diffdock-pp-server
make push-diffdock-pp-server   # 需要 VERSION 已 bump
```

## FC 部署

参考 [genie3-server README §阿里云函数计算部署](../genie3-server/README.md)。
关键项：

| 项 | 推荐 |
|---|---|
| 实例 | `fc.gpu.tesla.1`（T4 16 GB） |
| 超时 | 1800 s |
| 内存 | 32 GB |
| CPU | 8 vCPU |
| NAS jobs | `/fc → /data` |
| NAS weights | `/fc → /data/models/diffdock-pp`（只读） |
| 异步任务模式 | 启用 |
| Session affinity | 启用 (`bioagent-session-id`) |
| PreStop hook | 保留 |

## 参考

- **设计文档**：[engineering/decisions/2026-07-03-diffdock-pp-server-design.md](../../engineering/decisions/2026-07-03-diffdock-pp-server-design.md)
- **上游**：https://github.com/ketatam/DiffDock-PP （MIT, SHA `25a28900736c0730821e45265ee8e409751c358a`）
- **论文**：Ketata et al., arXiv:2304.03889, ICLR 2023 MLDD Workshop
