# chembounce-server

基于 FastAPI 的 [ChemBounce](https://github.com/jyryu3161/chembounce) HTTP
服务——把经典（非 ML）的 **ligand-based scaffold hopping** 工具包装成 FC
CPU 服务。**构建在 [bioagent-service-framework](../_framework/) 之上**。

ChemBounce：给定一个分子的 SMILES，做分子骨架替换（fragmentation +
相似性搜索 + 药物规则过滤），输出 N 个保留主体但骨架不同的候选。

⚠️ **LICENSE 注意**：upstream 仓库**无 LICENSE 文件**。本镜像**仅限项目
内部研究使用，不对外发布、不商业分发**。详见
[设计文档 §10.1](../../engineering/decisions/2026-06-28-chembounce-server-design.md)。

镜像 base：`python:3.11-slim`；conda env (Python 3.8 + RDKit 2020.09.5 +
scaffoldgraph + oddt + molvs)。**CPU-only，无 GPU 依赖**。

## 与 diffusion-hopping-server 的区别

两个服务都做 "scaffold hopping" 但根本不同：

| | **ChemBounce**（本服务） | DiffHopp |
|---|---|---|
| 输入 | SMILES 字符串 | 蛋白 PDB + 配体 SDF |
| 蛋白口袋约束 | ❌ | ✅ |
| 方法 | 经典相似性搜索 | 图扩散神经网络 |
| 算力 | CPU | GPU |
| 适用 | 早期 hit-to-lead、抗专利、无靶点结构时 | 已知靶点结构、要求保留结合姿势 |

## 架构

```
客户端 / Agent
  ↓ HTTP (form: input_smiles=... + thresholds)
┌────────────────────────────────────────────────────────────────┐
│  FastAPI + bioagent-service-framework  (port 9000)             │
│                                                                │
│  服务专属                                                      │
│    POST /api/scaffold_hop         (submit/poll)                │
│    POST /api/tasks/scaffold_hop   (FC async task mode)         │
│                                                                │
│  框架统一提供                                                  │
│    GET  /api/manifest                                          │
│    GET  /openapi.json                                          │
│    GET  /healthz, /healthz/detail   (后者带 DB 探针)           │
│    GET  /api/jobs/{id}/...                                     │
└────────────────────────────────────────────────────────────────┘
  ↓ subprocess (cwd=/opt/chembounce/upstream, 单卡串行)
chembounce.py → fragmentation → similarity search → drug-likeness filter
  ↓ 输出
NAS: <jobs_base_dir>/<job_id>/{input/, output/{overall_result.txt, fragment_*.tsv, resource_cost.json}, logs/run.log, job.json}
```

## API 速览

### `POST /api/scaffold_hop`

form 字段（**没有文件上传**——SMILES 是字符串）：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `input_smiles` | str | ✓（或 `input_smiles_uri`） | — | 目标分子 SMILES（1-500 字符） |
| `database` | enum | — | `250mw` | `250mw`（快）/ `full`（论文级，需 ≥64 GB） |
| `core_smiles` | str | — | — | 必须保留的核心子结构 SMILES |
| `frag_max_n` | int | — | `100` | 每个 fragment 的最大候选数 |
| `overall_max_n` | int | — | — | 总候选硬上限 |
| `scaffold_top_n` | int | — | — | 每个 fragment 测多少 scaffold |
| `cand_max_n__rplc` | int | — | `10` | 每个替换 scaffold 最多候选数 |
| `tanimoto_threshold` | float | — | `0.5` | 0.0-1.0 |
| `qed_min/max`, `sa_min/max`, `logp_min/max`, `mw_min/max`, `h_donor_min/max`, `h_acceptor_min/max` | float/int | — | — | 药物属性阈值 |
| `wo_lipinski` | bool | — | `false` | 关闭 Lipinski 五规则（大分子/肽用） |
| `low_mem` | bool | — | `false` | 低内存模式 |

示例：

```bash
curl -X POST $URL/api/scaffold_hop \
    -F 'input_smiles=CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl' \
    -F 'frag_max_n=100' \
    -F 'tanimoto_threshold=0.5'
# → { "job_id": "...", "status": "pending", ... }

# 等完成
curl $URL/api/jobs/<job_id>

# 拿主结果 TSV
curl -O $URL/api/jobs/<job_id>/file/output/overall_result.txt
```

锁定 core + 全库搜索：

```bash
curl -X POST $URL/api/scaffold_hop \
    -F 'input_smiles=CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl' \
    -F 'database=full' \
    -F 'tanimoto_threshold=0.7' \
    -F 'core_smiles=C4=NNN=N4' \
    -F 'frag_max_n=50'
```

### `POST /api/tasks/scaffold_hop`

FC 异步任务模式——HTTP 立即返回 202，FC 把请求和计算生命周期绑死。
控制台需要开"异步任务模式"。

## 配置

`pydantic-settings`，`CHEMBOUNCE_` 前缀。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `CHEMBOUNCE_JOBS_BASE_DIR` | `/data/chembounce_jobs` | NAS 上的 job 根目录 |
| `CHEMBOUNCE_ROOT` | `/opt/chembounce/upstream` | upstream 源码根（subprocess cwd） |
| `CHEMBOUNCE_PYTHON` | `/opt/conda/envs/chembounce/bin/python` | conda env Python |
| `CHEMBOUNCE_ENTRYPOINT` | `/opt/chembounce/upstream/chembounce.py` | upstream CLI |
| `CHEMBOUNCE_WEIGHTS_DIR` | `/data/models/chembounce/data` | scaffold + fingerprint DB 目录（NAS） |
| `CHEMBOUNCE_MAX_CONCURRENT_JOBS` | `1` | 单实例并发 |
| `CHEMBOUNCE_OSS_REGION` | `cn-hangzhou` | OSS URI 下载区域 |
| `CHEMBOUNCE_SESSION_HEADER_NAME` | `bioagent-session-id` | FC session affinity header |

字段名 `weights_dir` 沿用全项目约定（即使内容不是 ML 权重而是 scaffold
DB），让 `/healthz/detail` 探针逻辑统一。

## Data（NAS 挂载，**非镜像 baked**）

期望布局：

```
/data/models/chembounce/
└── data/
    ├── scaffolds.txt                         ← 全量 ~4M scaffolds SMILES
    ├── scaffolds_250mw.txt                   ← 250 MW 子集
    ├── scaffold_fingerprints.npz             ← 全量 Morgan FP
    └── scaffold_fingerprints_250mw.npz       ← 子集 Morgan FP
```

### Pre-stage（一次性）

```bash
# 1. Vendor 上游源码
./services/chembounce-server/scripts/vendor.sh

# 2. 下载 scaffold DB（从 Zenodo / Dropbox）
./services/chembounce-server/scripts/fetch_data.sh
# → services/chembounce-server/data/

# 3. 上传到 NAS
rsync -av services/chembounce-server/data/ \
    <NAS-mount>:/data/models/chembounce/data/
```

或直接下到 NAS：

```bash
DATA_DST=/mnt/nas/data/models/chembounce/data \
    ./services/chembounce-server/scripts/fetch_data.sh
```

### 验证 FC 部署

```bash
curl https://fc-chembounce-XXX.cn-hangzhou-vpc.fcapp.run/healthz/detail
# 期望：{"status":"ok","weights_loaded":true,"database_status":{"250mw":true,"full":true}}
```

### SIF / HPC

```bash
apptainer run \
    --bind /scratch/models/chembounce:/data/models/chembounce \
    chembounce-server.sif python -m server scaffold_hop \
    --input-smiles "..." \
    --output-dir results/
```

详见 [weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)。

## Docker 构建与运行

```bash
# 1. Vendor 上游源（一次性，重跑可升级 SHA）
./services/chembounce-server/scripts/vendor.sh

# 2. 构建（~1.5 GB 镜像，无 DB）
make build-chembounce-server

# 本地运行（无需 GPU + NAS / 本地 --bind 注入 DB）
docker run -p 9000:9000 --memory 64g \
    -v $(pwd)/services/chembounce-server/data:/data/models/chembounce/data:ro \
    chembounce-server
```

构建上下文必须是项目根目录（Dockerfile 同时 `COPY services/_framework`）。

## CLI 批处理模式

```bash
docker run --rm \
    -v /data:/data \
    -v /path/to/db:/data/models/chembounce/data:ro \
    chembounce-server \
    /opt/conda/envs/chembounce/bin/python -m server scaffold_hop \
    --input-smiles "CCCCC1=NC..." \
    --output-dir /data/results/ \
    --params-json '{"frag_max_n": 100, "tanimoto_threshold": 0.5, "database": "250mw"}'
```

Slurm sbatch 模板见 [apptainer-compatibility.md](../../engineering/guides/apptainer-compatibility.md)。

## 离线测试

```bash
uv run python -m pytest services/chembounce-server/tests/test_app.py -v
uv run python -m pytest services/chembounce-server/tests/test_cli.py -v

# FC 集成（部署后）
RUN_FC_TESTS=1 uv run python -m pytest -m fc \
    services/chembounce-server/tests/test_fc.py -v
```

## 阿里云函数计算部署

| 配置项 | 推荐值 |
|---|---|
| 函数类型 | **CPU**（参考 jwt-server 配置路径，不是 GPU） |
| CPU | 32 vCPU |
| 内存 | **64 GB**（同时承载 250mw + full DB） |
| 监听端口 | `9000` |
| 函数超时 | 3600 秒（chembounce 在复杂分子上可能慢） |
| NAS 挂载 | `/fc → /data` |
| 异步任务模式 | **启用** |
| Session affinity | 启用 |

⚠️ **不要 push 公开 registry** —— 见顶部 LICENSE 注意。仅内部 ACR 部署。

## 相关文档

- [chembounce-server 设计文档](../../engineering/decisions/2026-06-28-chembounce-server-design.md)
- [diffusion-hopping-server 设计](../../engineering/decisions/2026-06-28-diffusion-hopping-server-design.md) — 互补的 pocket-conditional scaffold hopping
- [Service 框架抽象设计](../../engineering/decisions/2026-05-12-service-framework-design.md)
- [新增服务 cookbook](../../engineering/guides/adding-a-new-service.md)
- 上游：[jyryu3161/chembounce](https://github.com/jyryu3161/chembounce)（**无 LICENSE**）
- Zenodo data：https://zenodo.org/records/16741967
