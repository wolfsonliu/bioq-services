# RFdiffusion Server

基于 FastAPI 的 RFdiffusion HTTP 服务，构建在 [bioq-service-framework](../_framework/) 之上：
HTTP 层 / job 生命周期 / 错误处理 / 持久化 / 多实例一致性 / Agent 协议描述均由框架统一提供，服务自身只
负责把 RFdiffusion 的 5 种典型用法（unconditional / motif / binder / symmetry / 自定义）映射为
`scripts/run_inference.py` 的 Hydra override argv。

上游：[RosettaCommons/RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)（[paper](https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1)）。

## 架构

```
客户端 / Agent
  ↓ HTTP
┌────────────────────────────────────────────────────────────────┐
│  FastAPI + bioq-service-framework  (port 9000)             │
│                                                                │
│  服务专属（rfdiffusion-server 注册）                           │
│    POST /api/generate/unconditional   (无条件 / 单体)          │
│    POST /api/generate/motif           (motif scaffolding)      │
│    POST /api/generate/binder          (PPI binder + hotspots)  │
│    POST /api/generate/symmetry        (对称 oligomer)          │
│    POST /api/generate                 (自定义 contig + 透传)   │
│                                                                │
│  框架统一提供                                                  │
│    GET  /api/manifest                 (Agent 协议描述)         │
│    GET  /openapi.json                 (字段 schema)            │
│    GET  /healthz, /healthz/detail                              │
│    GET  /api/jobs/{id}                (JobInfo 含 error 详情)  │
│    GET  /api/jobs/{id}/files / log / download / file/{path}    │
│    DELETE /api/jobs/{id}                                       │
└────────────────────────────────────────────────────────────────┘
  ↓ subprocess (单卡串行)
RFdiffusion: scripts/run_inference.py <Hydra overrides...>
  ↓ 输出
NAS: <jobs_base_dir>/<job_id>/{input/, output/design_<N>.{pdb,trb}, logs/run.log, job.json}
```

## API 速览

每个 POST 端点接受 `multipart/form-data`：文件字段 + pydantic 参数。返回 `JobInfo`（含 `job_id`），
实际计算在后台异步执行。

| 端点 | 必填输入 | 用途 |
|---|---|---|
| `POST /api/generate/unconditional` | — | 给定长度区间，无条件生成 backbone（可选 `cyclic=true` 走 RFpeptides 大环模式）|
| `POST /api/generate/motif` | PDB + contigs | 在 motif 周围设计 scaffold（contigs 内字母引用 PDB 的 chain+residue）|
| `POST /api/generate/binder` | target PDB + contigs | PPI binder 设计；推荐配合 `hotspots=A59,A83,A91` |
| `POST /api/generate/symmetry` | symmetry + total_length | 对称 oligomer（`c6` / `d2` / `tetrahedral`），可加 olig_contacts potential |
| `POST /api/generate` | contigs（+ 可选 PDB）| 自定义：透传任意 Hydra override（如 `diffuser.partial_T`），用于 partial diffusion / fold conditioning |

### 公共参数（_GenerationCommon）

| 字段 | 默认 | 说明 |
|---|---|---|
| `num_designs` | 10 | 每次提交生成多少个 design |
| `diffuser_t` | 50 | 反向扩散步数；50 是上游默认值 |
| `final_step` | 1 | 提前终止 trajectory（1 = 跑完）|
| `write_trajectory` | false | 是否保留完整 trajectory（默认关闭以省磁盘）|
| `deterministic` | false | RNG 固定，复现实验时打开 |
| `noise_scale` | 1.0 | 同时设置 `noise_scale_ca` 与 `noise_scale_frame`；binder 模式默认 0.0 |
| `model` | (auto) | 显式 checkpoint 名（`base` / `complex_base` / `active_site` / ...）；不填则脚本自动选 |

### 端点 1: unconditional

```bash
# 10 个长度 = 150aa 的单体
curl -X POST $URL/api/generate/unconditional \
  -F min_length=150 -F max_length=150 -F num_designs=10

# RFpeptides 大环（含 inference.cyclic=True）
curl -X POST $URL/api/generate/unconditional \
  -F min_length=12 -F max_length=18 -F cyclic=true -F num_designs=4
```

### 端点 2: motif scaffolding

```bash
# 在 5TPN 的 A163-181 motif 周围生成 10-40 aa 的 scaffold
curl -X POST $URL/api/generate/motif \
  -F input_pdb=@5TPN.pdb \
  -F 'contigs=10-40/A163-181/10-40' \
  -F num_designs=10

# 也可加 inpaint_seq（自动切到 InpaintSeq 模型）
curl -X POST $URL/api/generate/motif \
  -F input_pdb=@input.pdb \
  -F 'contigs=10-40/A30-50/10-40' \
  -F 'inpaint_seq=A1/A40-45'
```

### 端点 3: binder design

```bash
# 针对 insulin receptor (A1-150) 设计 70-100 aa binder，hotspots = A59/A83/A91
curl -X POST $URL/api/generate/binder \
  -F input_pdb=@insulin_target.pdb \
  -F 'contigs=A1-150/0 70-100' \
  -F 'hotspots=A59,A83,A91' \
  -F num_designs=10
```

binder 模式默认 `noise_scale=0.0`（per upstream README），可通过参数覆盖。

### 端点 4: symmetric oligomer

```bash
# C6 对称 480aa 共 6 链 × 80aa，配合 olig_contacts potential
curl -X POST $URL/api/generate/symmetry \
  -F symmetry=c6 -F total_length=480 -F num_designs=10 \
  -F 'guiding_potentials=["type:olig_contacts,weight_intra:1,weight_inter:0.1"]' \
  -F guide_scale=2.0 -F guide_decay=quadratic \
  -F olig_intra_all=true -F olig_inter_all=true
```

### 端点 5: 自定义 / 透传

最大灵活性，适合：partial diffusion、scaffold-guided、provide_seq 等 base.yaml 里能写但
没单独暴露的高级用法。

```bash
# Partial diffusion：从 2KL8 出发噪声 10 步再去噪
curl -X POST $URL/api/generate \
  -F input_pdb=@2KL8.pdb \
  -F 'contigs=79-79' \
  -F num_designs=10 \
  -F 'extra_overrides={"diffuser.partial_T": 10}'

# Scaffold-guided binder
curl -X POST $URL/api/generate \
  -F input_pdb=@target.pdb \
  -F 'contigs=A1-150/0 60-100' \
  -F 'extra_overrides={"scaffoldguided.scaffoldguided": true, "scaffoldguided.target_pdb": true, "ppi.hotspot_res": "[A59,A83]"}'
```

### 支持的 `input_uri` schemes

适用于 `/motif`、`/binder`、`/api/generate` 的 PDB 输入：

| 形式 | 说明 |
|---|---|
| `job://<job_id>/<filename>` | 拉取本 service 另一 job 的 `output/` 文件（**推荐用于链式调用**）|
| `file:///abs/path` 或 `/abs/path` | NAS 上的绝对路径（跨 service 共享）|
| `oss://<bucket>/<key>` | 阿里云 OSS 对象 |
| `http(s)://...` | 任意 HTTP(S) URL，包括 OSS 签名 URL |

### 输出文件

所有端点都把 design 落到 `<job>/output/`：

```
output/
├── design_0.pdb       # 第 0 个 trajectory 的最终 backbone（全 Gly）
├── design_0.trb       # metadata pickle（contig、config、resmap）
├── design_1.pdb
├── design_1.trb
├── ...
└── traj/              # 仅 write_trajectory=true 时存在
    └── design_0_*.pdb
```

```bash
# 列输出
curl $URL/api/jobs/<job_id>/files

# 下载单个 PDB
curl -o my.pdb $URL/api/jobs/<job_id>/file/design_0.pdb

# 打包下载
curl -O $URL/api/jobs/<job_id>/download
```

### 下游 pipeline 提示

RFdiffusion 是 backbone 扩散模型，输出全是 poly-Gly，**没有 sidechain**。完整 de novo 流程一般是：

```
RFdiffusion (本服务)
  ↓ design_<N>.pdb
ProteinMPNN (proteinmpnn-server)   ← 序列设计
  ↓ <name>.fasta
AlphaFold2 / RF2                    ← 重新 fold 校验
```

ProteinMPNN server 接受 `input_uri=job://<rfdiffusion_job_id>/design_0.pdb`，可直接零拷贝链式调用。

## 配置

环境变量，env_prefix=`RFDIFFUSION_`：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `RFDIFFUSION_JOBS_BASE_DIR` | `/data/rfdiffusion_jobs` | NAS 上的 job 根目录 |
| `RFDIFFUSION_ROOT` | `/opt/rfdiffusion` | RFdiffusion 源码根（subprocess cwd）|
| `RFDIFFUSION_MODELS_DIR` | `/data/models/rfdiffusion/models` | 权重目录（NAS 挂载；v0.0.9 起从镜像外置，~3.9 GB） |
| `RFDIFFUSION_INFERENCE_SCRIPT` | `/opt/rfdiffusion/scripts/run_inference.py` | Hydra 驱动的入口 |
| `RFDIFFUSION_PYTHON` | `/opt/rfdiffusion/.venv/bin/python` | venv 解释器 |
| `RFDIFFUSION_PORT` | `9000` | uvicorn 监听端口（FC CAPort）|
| `RFDIFFUSION_KEEP_ALIVE_SEC` | `900` | uvicorn `--timeout-keep-alive` |
| `RFDIFFUSION_MAX_CONCURRENT_JOBS` | `1` | 单卡建议保持 1 |
| `RFDIFFUSION_OSS_REGION` | `cn-hangzhou` | OSS URI 下载区域 |

OSS 凭证按阿里云 SDK 约定走 `OSS_ACCESS_KEY_ID` + `OSS_ACCESS_KEY_SECRET`。

## 本地开发

```bash
# 1. 准备 RFdiffusion（先下载权重）
cd opensource/RFdiffusion
bash scripts/download_models.sh models
conda env create -f env/SE3nv.yml
conda activate SE3nv
cd env/SE3Transformer && pip install -r requirements.txt && python setup.py install && cd ../..
pip install -e .

# 2. 安装服务框架
pip install ../../framework

# 3. 软链 server 包
ln -s ../../services/rfdiffusion-server server
export RFDIFFUSION_ROOT=$(pwd)
export RFDIFFUSION_MODELS_DIR=$(pwd)/models
export RFDIFFUSION_PYTHON=$(which python)
export RFDIFFUSION_JOBS_BASE_DIR=/tmp/rfdiffusion_jobs

# 4. 启动
uvicorn server.app:app --host 0.0.0.0 --port 9000 --reload
```

## Weights

v0.0.9 起 RFdiffusion checkpoints（~3.9 GB，8 个 .pt）**不再 baked 到镜像**，
而是从 NAS 加载。期望布局：

```
/data/models/rfdiffusion/
└── models/
    ├── Base_ckpt.pt
    ├── Complex_base_ckpt.pt
    ├── Complex_beta_ckpt.pt
    ├── Complex_Fold_base_ckpt.pt
    ├── InpaintSeq_ckpt.pt
    ├── InpaintSeq_Fold_ckpt.pt
    ├── ActiveSite_ckpt.pt
    └── Base_epoch8_ckpt.pt
```

### 下载（一次性）

```bash
# 先 vendor 上游源（含 download_models.sh）
./services/rfdiffusion-server/scripts/vendor.sh

# 用 upstream 下载脚本，输出到本地 stage 目录
bash services/rfdiffusion-server/upstream/scripts/download_models.sh \
    services/rfdiffusion-server/models

# 上传到 NAS
rsync -av services/rfdiffusion-server/models/ \
    <NAS-mount>:/data/models/rfdiffusion/models/
```

### FC

NAS 自动挂载到 `/data/models/rfdiffusion/`。验证：

```bash
curl https://fc-rfdiffusion-cdskxiqtnk.cn-hangzhou-vpc.fcapp.run/healthz/detail
# 期望：{"status":"ok","weights_loaded":true,"ckpts_found":8}
```

### SIF / HPC (apptainer)

```bash
apptainer run --nv \
    --bind /scratch/models/rfdiffusion:/data/models/rfdiffusion \
    rfdiffusion-server.sif python -m server unconditional ...
```

详见 [weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)。

## Docker 构建与运行

```bash
# 1. Vendor 上游源码（一次性，重跑可升级 SHA）
./services/rfdiffusion-server/scripts/vendor.sh

# 2. 构建（CUDA 11.8 base，~3 GB 镜像，无权重）
docker build --platform linux/amd64 -t rfdiffusion-server -f services/rfdiffusion-server/Dockerfile .

# 或通过 Makefile
make build-rfdiffusion-server

# 本地运行（需 GPU + NAS / 本地 --bind 注入权重）
docker run --gpus all -p 9000:9000 --memory 16g \
    -v $(pwd)/models:/data/models/rfdiffusion/models \
    rfdiffusion-server
```

构建上下文必须是项目根目录（Dockerfile 同时 `COPY framework`）。

## 阿里云函数计算部署

| 配置项 | 推荐值 |
|---|---|
| 运行时 | 自定义容器镜像 |
| GPU 配置 | `fc.gpu.tesla.1` 起步（8 GB 显存可跑常规设计；大蛋白上 `fc.gpu.ampere.1`）|
| 监听端口 CAPort | `9000` |
| 函数超时 | ≥ 600 秒（partial diffusion / 大 num_designs 可能数百秒）|
| 内存 | `16384` MB |
| CPU | `4` vCPU |
| 磁盘 | `10240` MB |
| NAS 挂载 | 推荐挂到 `/data` —— 跨实例 / 跨服务共享 job 文件 |

## FC 平台限制与适配

| 限制项 | 值 | 适配 |
|---|---|---|
| 启动超时 | 120 s | `/healthz` 不加载模型，立即响应 |
| Keep-alive | ≥ 15 min | `--timeout-keep-alive=900` |
| 监听地址 | `0.0.0.0:CAPort` | uvicorn `--host 0.0.0.0 --port 9000` |
| 可写磁盘 | 512 MB ~ 10 GB | 框架自动清理已完成 job + 推荐挂 NAS |
| GPU 镜像大小 | ≤ 15 GB | base + torch + dgl + 4 GB 权重，总计约 10 GB |
| 镜像架构 | AMD64 only | `--platform linux/amd64` |

## 设计要点

- **5 个结构化端点 + 1 个透传**：覆盖 RFdiffusion README 的 5 大典型用法；冷门用法
  （partial diffusion / fold conditioning / 自定义 potential）走 `/api/generate` 的
  `extra_overrides`，避免给参数表加噪。
- **共享 NAS 链式调用**：相邻步骤通过 `job://<id>/<file>` 引用上一步输出，避免 PDB 在
  HTTP 上下行。
- **错误信息丰富**：失败 job 的 `JobInfo` 自带 `error_summary` + `error_tail`，无需另调
  `/log` 即可判定原因。
- **多实例 FC 友好**：sidecar 持久化 + read-through cache，submit/poll 落到不同实例都 OK。
- **快速启动**：`/healthz` 无模型加载，FC 120 s 启动探测通过；权重烘焙到镜像避免冷启动下载。

## 相关文档

- [bioq-service-framework](../_framework/README.md) — 通用 HTTP / job / 错误处理 / manifest 层
- [Service 框架抽象设计](../../engineering/decisions/2026-05-12-service-framework-design.md)
- [Tool 抽象层设计](../../engineering/decisions/2026-04-23-tool-abstraction-design.md)
- [rfantibody-server](../rfantibody-server/README.md) — 抗体专用变体（RFdiffusion_Ab + RF2 + 抗体专用 MPNN）
- [genie3-server](../genie3-server/README.md) — 类似设计的通用扩散生成 server
