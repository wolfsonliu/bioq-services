# ProteinMPNN Server

基于 FastAPI 的 ProteinMPNN HTTP 服务，覆盖 4 套权重 × 3 类用途：

| Endpoint | 用途 | 上游模式 |
|---|---|---|
| `POST /api/design` | 序列设计（FASTA 输出） | 默认 |
| `POST /api/score` | 结构+序列对打分 | `--score_only 1` |
| `POST /api/probs` | 位点 AA 概率 | `--{conditional,conditional_backbone,unconditional}_probs_only` |

权重族通过 `model_variant` 字段切换：`vanilla / soluble / ca_only / abmpnn`。

## 架构

```
客户端 / Agent
  ↓ HTTP
┌─────────────────────────────────────────────────────────┐
│  FastAPI + bioq-service-framework  (port 9000)      │
│  POST /api/{design,score,probs}                         │
│  GET  /api/manifest  /api/jobs/{id}/*  /healthz         │
└─────────────────────────────────────────────────────────┘
  ↓ subprocess (cwd=/opt/proteinmpnn, 单卡串行)
helper_scripts/*.py  →  intermediates/*.jsonl
python protein_mpnn_run.py --jsonl_path ... --path_to_model_weights ...
  ↓
NAS: /data/proteinmpnn_jobs/<job_id>/output/{seqs,score_only,probs}/...
```

## API 速览

### Design（默认模式）

```bash
curl -X POST http://localhost:9000/api/design \
  -F pdb=@input.pdb \
  -F model_variant=vanilla \
  -F model_name=v_48_020 \
  -F 'chains_to_design=A C' \
  -F num_seq_per_target=8 \
  -F sampling_temp=0.1 \
  -F name=run1
```
输出：`output/seqs/run1.fa`

### Score

```bash
curl -X POST http://localhost:9000/api/score \
  -F pdb=@complex.pdb \
  -F num_seq_per_target=10 \
  -F name=s1
```
输出：`output/score_only/s1_pdb.npz`

### Probs

```bash
curl -X POST http://localhost:9000/api/probs \
  -F pdb=@input.pdb \
  -F kind=conditional \
  -F name=p1
```
输出：`output/probs/p1.npz`（每位置 21 维 log-prob 向量）

### AbMPNN（抗体专用）

```bash
curl -X POST http://localhost:9000/api/design \
  -F pdb=@antibody.pdb \
  -F model_variant=abmpnn \
  -F model_name=abmpnn \
  -F 'chains_to_design=H L' \
  -F 'fixed_positions=1 2 3 4 5, 1 2 3 4 5' \
  -F name=cdr_run
```

## 输入 URI scheme

文件字段（`pdb`）支持 `pdb_uri` 同名 Form 字段引用 NAS 上已有文件：

| scheme | 用途 |
|---|---|
| `job://<job_id>/<filename>` | 拉同 service 历史 job 的输出 |
| `file:///abs/path` | NAS 上绝对路径（跨 service 共享挂载） |
| `oss://<bucket>/<key>` | OSS 对象（需 `OSS_ACCESS_KEY_ID / _SECRET`） |
| `http(s)://...` | 任意 URL，含 OSS 签名 URL |

## 权重 / 模型矩阵

| `model_variant` | `model_name` | 权重目录 |
|---|---|---|
| `vanilla`（默认） | `v_48_002 / v_48_010 / v_48_020 / v_48_030` | `vanilla_model_weights/` |
| `soluble` | `v_48_002 / v_48_010 / v_48_020 / v_48_030` | `soluble_model_weights/` |
| `ca_only` | `v_48_002 / v_48_010 / v_48_020`（**无 v_48_030**） | `ca_model_weights/` |
| `abmpnn` | `abmpnn` | `AbMPNN_model_weights/` |

非法组合（如 `ca_only + v_48_030`、`abmpnn + v_48_020`）返回 HTTP 422。

## 配置

环境变量前缀 `PROTEINMPNN_`：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `PROTEINMPNN_JOBS_BASE_DIR` | `/data/proteinmpnn_jobs` | NAS 上 job 根 |
| `PROTEINMPNN_ROOT` | `/opt/proteinmpnn` | ProteinMPNN repo 根（subprocess cwd） |
| `PROTEINMPNN_WEIGHTS_DIR` | `/opt/proteinmpnn` | 4 个 `*_model_weights/` 的父目录 |
| `PROTEINMPNN_PORT` | `9000` | uvicorn 端口（FC CAPort） |
| `PROTEINMPNN_KEEP_ALIVE_SEC` | `900` | `--timeout-keep-alive` |
| `PROTEINMPNN_MAX_CONCURRENT_JOBS` | `1` | 单卡串行 |
| `PROTEINMPNN_DISK_LIMIT_MB` | `8000` | 自动清理阈值 |
| `PROTEINMPNN_OSS_REGION` | `cn-hangzhou` | OSS 区域 |

OSS 凭证走 `OSS_ACCESS_KEY_ID` + `OSS_ACCESS_KEY_SECRET`。

## 本地开发

```bash
# 1. 装 framework + 算法栈
cd opensource/ProteinMPNN
pip install ../../framework httpx alibabacloud-oss-v2
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 numpy

# 2. 软链 server 包让 uvicorn 找到
ln -s $(pwd)/../../services/proteinmpnn-server $(pwd)/server

# 3. 跑
export PROTEINMPNN_ROOT=$(pwd)
export PROTEINMPNN_WEIGHTS_DIR=$(pwd)
export PROTEINMPNN_JOBS_BASE_DIR=/tmp/proteinmpnn_jobs
uvicorn server.app:app --host 0.0.0.0 --port 9000 --reload
```

## Docker 构建

```bash
docker build --platform linux/amd64 -t proteinmpnn-server -f services/proteinmpnn-server/Dockerfile .
# 或：
make build-proteinmpnn-server
# 本地 GPU 运行
docker run --gpus all -p 9000:9000 proteinmpnn-server
```

构建上下文必须是项目根目录，因为 Dockerfile 同时 `COPY framework` 和 `COPY opensource/ProteinMPNN`。

## FC 部署

| 配置项 | 推荐 |
|---|---|
| GPU 配置 | `fc.gpu.tesla.1`（T4 16 GB） |
| 函数超时 | 1800 秒 |
| 内存 | `16384` MB |
| 磁盘 | `5120` MB |
| NAS 挂载 | `/fc → /data` |

```bash
make push-proteinmpnn-server
# 然后到 FC 控制台把函数镜像更新到 harbor.ruosheng.bio/aliyun_fc/proteinmpnn-server:vX.Y.Z
```

## 设计要点

- **4 套权重一镜像**：vanilla / soluble / ca_only / AbMPNN 都烘焙进同一个镜像；调用方仅切换 `model_variant`。
- **helper_scripts 内化**：`fixed_positions` / `tied_positions` / `bias_AA` 等结构化字段，服务侧自动 `parse_multiple_chains` + `make_*` 生成 JSONL，agent 不必先在客户端预处理。
- **错误信息丰富**：失败 job 的 `JobInfo` 自带 `error_summary` + `error_tail` + `failure_kind`；helper_scripts 错误直接 422 同步返回。
- **单 PDB 单 job**：批量场景由 pipeline 调度多 job 并发实现，不污染 service 接口。

## 已知限制 / 后续

- v0.0.1 不含 PSSM 输入、多 PDB 批量、score-from-fasta。需要时在 v0.0.2 加。
- 单实例 GPU 串行；不适合大批量并行，需要时升级到多实例 FC + NAS 共享。

## 相关文档

- [proteinmpnn-server 设计](../../engineering/decisions/2026-05-13-proteinmpnn-server-design.md)
- [新增 bioagent service cookbook](../../engineering/guides/adding-a-new-service.md)
- [bioq-service-framework](../_framework/README.md)
- [调用 bioagent service](../../engineering/guides/calling-bioagent-services.md)
- [ProteinMPNN 上游 README](../../opensource/ProteinMPNN/README.md)
