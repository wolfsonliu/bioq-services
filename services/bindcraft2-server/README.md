# bindcraft2-server

把 [BindCraft2](https://github.com/PacesaLab/BindCraft2)（BC2 v1.0.1）包装成 bioq 服务：
一次 `design` 请求跑一场完整的蛋白 binder 设计 campaign（AlphaFold 2 hallucination 设计 →
ProteinMPNN 重设计 → 独立 AlphaFold 验证 → 过滤排序），并可用 `rank` / `filter` 对已有
campaign 换指标、换阈值，无需重跑设计。

## ⚠️ 许可证红线（先读这段）

BC2 采用 **BindCraft2 Source-Available License (Hosting-Restricted)**
（Copyright (c) 2026 Martin Pacesa, University of Zurich），**不是** OSI 开源许可证。

- **仅限组织内部使用**：本地 / 私有基础设施 / 私有云 / 为使用组织单独运营的基础设施；
  由本组织自己的自动化与 agent 工具链驱动执行属于允许范围。
- **不得对外提供工具功能**：未经 UZH 单独商业许可，不得把本服务或其修改版作为
  hosted / managed / cloud / API / web application / workflow platform / SaaS 提供给第三方
  （许可证点名了"作为 hosted platform 里可调用的 tool / agent action / connector"这一形态）。
- **不得分发预配置部署**：不得把本镜像 / VM 镜像 / 安装器交给第三方去托管。
- **命名**：不得用 "BindCraft2" 标识、宣传 Hosted Service 或衍生作品，以免暗示与 UZH 的
  官方关联或等效性；"基于 BindCraft2" 这类准确陈述是允许的。

完整条款见上游 `LICENSE`，以及仓库根 [`THIRD_PARTY_NOTICES`](../../THIRD_PARTY_NOTICES)。

## Endpoints

| Endpoint | 说明 |
|---|---|
| `POST /api/design` | 提交一场 campaign（submit/poll） |
| `POST /api/tasks/design` | 同上，FC 异步任务模式（**推荐**，campaign 小时级） |
| `POST /api/rank` | 对已有 campaign 按指标重排 → `output/ranked_by_<metric>.csv` |
| `POST /api/tasks/rank` | 同上，task 模式 |
| `POST /api/filter` | 对已有 campaign 重放/覆盖阈值 → `output/filtered.csv` |
| `POST /api/tasks/filter` | 同上，task 模式 |

请求参数、curl 示例与字段表以运行中实例的 `GET /api/manifest` 为准（`endpoint_examples()`
是单一真源）。

### 快速上手

```bash
# shipped target
curl -X POST $URL/api/design -F target_name=hPDL1 -F 'binder_lengths=[80,80]' \
     -F number_of_final_designs=10 -F max_trajectories=200

# 自己的 target
curl -X POST $URL/api/design -F target=@target.pdb -F target_chains=A \
     -F 'hotspots=54,56,66-70' -F modality=VHH -F humanize=true

# 对上一场结果换指标重排（零拷贝，共享 NAS）
curl -X POST $URL/api/rank -F campaign_uri=job://<design_job_id> \
     -F 'on=["i_pTM","i_pDAE"]'

# 换阈值重新筛选
curl -X POST $URL/api/filter -F campaign_uri=job://<design_job_id> \
     -F 'where=["i_pAE=0.45","Interface_Residues>=7"]'
```

`binder_lengths` / `on` / `where` 在 HTTP 上以 **JSON 字符串**传入（`[80,80]`），
在 CLI 上也接受逗号写法（`80,80`）。

### CLI 批处理（Slurm）

```bash
apptainer exec --nv --bind /data:/data bindcraft2-server.sif \
    /opt/venv/bin/python -m server design --target-name hPDL1 \
    --max-trajectories 200 --output-dir /data/runs/$SLURM_JOB_ID/
```

## 运行约束

- **无 CPU 模式**。cuda13 extra 需 compute capability ≥ 7.5；V100(7.0) 及更老要改成
  `[cuda12]`（改 Dockerfile 一行）。
- **显存**：单 worker 预算 `2.0*(3.4GB + 38kB*N²)`，N=256 约 11.8 GB、N=384 约 18 GB、
  N=512 约 26.7 GB，另留 4 GB headroom。服务锁死 `workers_per_gpu=1`（BC2 的宿主内存
  上限读节点可用内存，会过度 pack）。
- **时长**：FC `timeout: 36000`（10 h）是硬上限。`max_trajectories` 是唯一的主动兜底。
  超时后任务 FAILED，但 `1_Trajectories/`、`2_Refolded/` 的中间结果仍在 NAS 上。
  **v0.0.1 无自动 resume**（框架每次 submit 建新 job_dir，BC2 原生续跑跨 job 失效）。
- **进度**：无实时百分比。看 `/log` 尾部，或直接读
  `job://<id>/1_Trajectories/!_Trajectories.csv` 行数。
- **冷启动**：JAX runtime + 模型编译。`JAX_COMPILATION_CACHE_DIR` 指到 NAS 可缓解；
  缓存按卡型分目录，跨卡型不可复用。

## 输出

```
<jobs_base_dir>/<job_id>/
├── input/campaign.json          # 服务生成的 campaign（唯一真源）
├── input/target.pdb             # 上传/URI 落盘（shipped target 时不存在）
├── output/                      # == 上游 project_folder
│   ├── 1_Trajectories/!_Trajectories.csv
│   ├── 2_Refolded/!_Refolded.csv
│   ├── 3_Ranked/!_Ranked.csv    # 主产物：accepted，best-first by i_pDAE
│   ├── summary.csv
│   ├── campaign_metadata.json   # resolved 设置 + checkpoint SHA-256
│   └── workers/worker_<NN>_gpu_<id>.log
└── logs/run.log
```

`detect_outputs` 接受 `3_Ranked/!_Ranked.csv`、`summary.csv`、`ranked_by_*.csv`、
`filtered.csv` 任一非空。**包含 `summary.csv` 是刻意的**：accepted=0 的合法 campaign
不会写 `!_Ranked.csv`。

## Weights

AlphaFold 参数 5.3 GB，**外置 NAS**（不烘焙进镜像）：

```
/data/models/bindcraft2/alphafold/
└── params/
    ├── params_model_1_multimer_v3.npz     # 5 个 multimer（设计 + 验证）
    ├── params_model_2_multimer_v3.npz
    ├── params_model_3_multimer_v3.npz
    ├── params_model_4_multimer_v3.npz
    ├── params_model_5_multimer_v3.npz
    ├── params_model_1_ptm.npz             # 2 个 monomer（验证）
    └── params_model_2_ptm.npz
```

上游接受 `<dir>/params/params_<model>.npz` 或 `<dir>/params_<model>.npz`；服务用
`BINDCRAFT2_ALPHAFOLD_PARAMS_DIR` 指定，并把它注入子进程的 `BINDCRAFT_AF2_PARAMS`
（**必须**——否则上游会去下载 5.3 GB）。

pre-stage：

```bash
# 方式 A：NAS 上已有 alphafold-server 的参数 → 软链，零下载
ln -s /data/models/alphafold /data/models/bindcraft2/alphafold

# 方式 B：下载（只解出这 7 个文件）
WEIGHTS_DST=/data/models/bindcraft2/alphafold \
    ./services/bindcraft2-server/scripts/fetch_weights.sh
```

ProteinMPNN 三变体（`weights_{neutral,negative,positive}/v_48_020.npz`，各 ~6.6 MB）
**随包**分发，由 `vendor.sh` 校验、Dockerfile 构建期自检。

FC 验证：

```bash
curl -s $URL/healthz/detail | python3 -m json.tool
# 期望：weights_loaded=true, weights_missing=[], gpu_backend="gpu"
```

SIF / HPC：

```bash
apptainer exec --nv --bind /scratch/models:/data/models bindcraft2-server.sif \
    /opt/venv/bin/python -m server design --target-name hPDL1 --output-dir /scratch/run1/output
```

## 配置

`env_prefix = BINDCRAFT2_`，全走 pydantic-settings。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `BINDCRAFT2_JOBS_BASE_DIR` | `/data/bindcraft2_jobs` | 任务目录根 |
| `BINDCRAFT2_ROOT` | `/opt/bindcraft` | 上游源码树（editable install 依赖它） |
| `BINDCRAFT2_SHIPPED_WEIGHTS_DIR` | `/opt/bindcraft` | ProteinMPNN 权重根；探针查 `<它>/bindcraft/weights/proteinmpnn/weights_<变体>/v_48_020.npz` |
| `BINDCRAFT2_PYTHON` | `/opt/venv/bin/python` | 解释器 |
| `BINDCRAFT2_MODULE` | `bindcraft.cli` | 上游 CLI 模块；置空则只跑 `python <args>` |
| `BINDCRAFT2_ALPHAFOLD_PARAMS_DIR` | `/data/models/bindcraft2/alphafold` | AF2 参数目录 |
| `BINDCRAFT2_COMPILE_CACHE_DIR` | `/data/models/bindcraft2/xla_cache` | JAX 图缓存 |
| `BINDCRAFT2_MAX_CONCURRENT_JOBS` | `1` | 整实例独占 GPU |
| `BINDCRAFT2_WORKERS_PER_GPU` | `1` | 覆盖 BC2 自动 pack |
| `BINDCRAFT2_MAX_WORKERS_PER_GPU` | `1` | 同上 |
| `BINDCRAFT2_DEFAULT_MAX_TRAJECTORIES` | `500` | `max_trajectories` 未指定时的兜底 |
| `BINDCRAFT2_GPU_PROBE_TTL_SECONDS` | `300` | `/healthz/detail` GPU 探针缓存 |

## v0.0.1 不做

`score` / `archive` / `unarchive` / `fetch-weights` / `campaign_output` / 参数 sweep /
多 target（同源对、detarget、多链受体）/ 自定义 scaffold / wall-clock 强杀 / 跨 job resume /
实时百分比进度。边界理由见
[设计文档](../../docs/specs/2026-09-22-bindcraft2-server-design.md)。

## 设计文档

[`docs/specs/2026-09-22-bindcraft2-server-design.md`](../../docs/specs/2026-09-22-bindcraft2-server-design.md)
