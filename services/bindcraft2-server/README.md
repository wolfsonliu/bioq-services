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
| `POST /api/rank` | 对已有 campaign 按指标重排 → `output/ranked_by_<m1>[_<m2>...].csv` |
| `POST /api/tasks/rank` | 同上，task 模式 |
| `POST /api/filter` | 对已有 campaign 重放/覆盖阈值 → `output/filtered.csv` |
| `POST /api/tasks/filter` | 同上，task 模式 |

`rank` 产物文件名把 `on` 里的**全部**指标按顺序拼进去、非字母数字折成 `_`
（`on=["i_pTM","i_pDAE"]` → `ranked_by_i_pTM_i_pDAE.csv`）；只取第一个指标会让不同
tie-break 的产物撞名。完整字段表以运行中实例的 `GET /api/manifest` 与
`GET /openapi.json` 为准：**endpoint 列表来自 app 的路由，请求字段来自 OpenAPI**；
`endpoint_examples()` 只提供每个 endpoint 的 curl / python 示例，不是 endpoint 与字段的
真源。

### 快速上手

```bash
# shipped target
curl -X POST $URL/api/design -F target_name=hPDL1 -F 'binder_lengths=[80,80]' \
     -F number_of_final_designs=10 -F max_trajectories=200

# 自己的 target
curl -X POST $URL/api/design -F target=@target.pdb -F target_chains=A \
     -F 'hotspots=54,56,66-70' -F modality=VHH -F humanize=true

# 对上一场结果换指标重排（源目录通常只读；见下方"零拷贝"注意）
curl -X POST $URL/api/rank -F campaign_uri=job://<design_job_id> \
     -F 'on=["i_pTM","i_pDAE"]'

# 换阈值重新筛选
curl -X POST $URL/api/filter -F campaign_uri=job://<design_job_id> \
     -F 'where=["i_pAE=0.45","Interface_Residues>=7"]'
```

`campaign_uri` 是**目录**输入（`job://<job_id>[/<subdir>]`、`file:///abs` 或裸绝对路径；
**不接受 `oss://` / `http(s)://`**，会 422）。"零拷贝"有前提：只有当请求的源表已有记录行、且
请求指标已记录或可派生时才只读；若源表没有记录行但目录里留了结构文件（或用了 `--rescore`），
上游 `score_structure_folder` 会把 `scored.csv` 写进**源目录**。这一点尚未实测（设计文档 P3）。

`binder_lengths` / `on` / `where` 在 HTTP 上以 **JSON 字符串**传入（`[80,80]`，
逗号写法 `80,80` 在 HTTP 上会在进模型前被 `model_form_depends` 判为
`json_invalid` → 422）；在 CLI 上也接受逗号写法（`80,80`）。CLI 的 `--on` **不支持重复
传参**（后者覆盖前者），多指标要写成一个逗号串。

`modality` 接受单个名字、逗号串或 JSON 数组（`modality=VHH`、`modality=binder,VHH`、
`-F 'modality=["binder","VHH"]'`）：逗号组合会被归一成数组落盘，因为上游 campaign JSON 里
的多预设必须是数组（字符串 `'binder,VHH'` 会被当成单个 preset 名而 preflight 失败）。
`target_chains` / `hotspots` / `coldspots` 只在 `target` 上传 / `target_uri` 路径下生效；
与 `target_name` 同时给会 **422**（shipped preset 自带这三项，同给等于静默丢弃）。

### CLI 批处理（Slurm）

```bash
apptainer exec --nv --bind /data:/data bindcraft2-server.sif \
    /opt/venv/bin/python -m server design --target-name hPDL1 \
    --max-trajectories 200 --output-dir /data/runs/$SLURM_JOB_ID/
```

## 运行约束

- **无 CPU 模式**。cuda13 extra 需 compute capability ≥ 7.5；V100(7.0) 及更老要改成
  `[cuda12]`（改 Dockerfile 一行）。
- **显存**：单 worker 预算 `2.0*(3.4GB + 38kB*N²)`（上游公式），另留 4 GB headroom。
  **`N` 是 padding 后的复合物残基数，不是 `binder_lengths`**：上游按 bucket 32 把最长的
  binder 长度向上取整、乘 `copies`，加上最长 target 链长度后再向上取整一次
  （`N = padded32(padded32(max(binder_lengths)) * copies + max(target_length))`）。
  以 shipped target `hPDL1`（115 残基）为例：`[80,80]` → N=224 → 10.61 GB；
  `[60,60]` → 192 → 9.60 GB；默认 `binder` preset `[60,180]` → 320 → 14.58 GB；
  `[250,600]`（large_binder 默认）→ 736 → 47.97 GB。**16 GB 的 T4 扣掉 headroom 只剩
  12 GB，连默认 preset 都装不下**——最小可用卡是 24 GB 级（如 A10 24 GB / L20 48 GB）。
  24576 MB 卡（24 GB）预算 20 GB ⇒ `N ≤ 416`；对 115 残基 target 相当于
  `binder_lengths` 最大 **288**（289–320 都落到 448 桶 → 22.05 GB，超预算）。
  ⚠️ API 对 `binder_lengths` **没有上界**，超预算请求不会在提交时被拒，而是在数小时后
  CUDA OOM。服务锁死 `workers_per_gpu=1`（BC2 的宿主内存上限读节点可用内存，会过度 pack）。
  完整的逐场景表格见[设计文档](../../docs/specs/2026-09-22-bindcraft2-server-design.md)。
- **时长**：FC `timeout: 36000`（10 h）是硬上限。`max_trajectories` 是唯一的主动兜底。
  超时后**要等新实例启动并从磁盘恢复**，该 job 才会被标成 FAILED（框架的
  `restart_semantics`；运行中的 job 在下一次实例加载时降级为 FAILED / `interrupted`）。
  `1_Trajectories/`、`2_Refolded/` 的中间结果仍在 NAS 上——**但受 `BINDCRAFT2_DISK_LIMIT_MB`
  淘汰阈值约束**：超过阈值时框架会整目录删掉已结束的 job，所以并非永久保留。
  **v0.0.1 无自动 resume**（框架每次 submit 建新 job_dir，BC2 原生续跑跨 job 失效）。
- **进度**：无实时百分比。看 `GET /api/jobs/<id>/log` 尾部，或读
  `GET /api/jobs/<id>/file/1_Trajectories/!_Trajectories.csv` 的行数——`job://` 是
  **输入** URI（`campaign_uri` 用），不是可以直接 GET 的产物 URL。
- **冷启动**：JAX runtime + 模型编译。`JAX_COMPILATION_CACHE_DIR` 指到 NAS 可缓解。
  注意**不是**上游默认的"按卡型分目录"：服务在 `adapter.subprocess_env` 里直接设了
  `JAX_COMPILATION_CACHE_DIR`，会短路上游的 per-card 目录逻辑，所以所有卡型共用一个
  缓存目录；跨卡型能否命中由 JAX 自己的缓存 key 决定。

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
│   └── campaign_metadata.json   # resolved 设置 + checkpoint SHA-256
└── logs/run.log
```

**不存在 `output/workers/worker_<NN>_gpu_<id>.log`。** 本部署把 `workers_per_gpu=1` /
`max_workers_per_gpu=1` 写死，单卡下 worker plan 长度为 1，而上游在 plan 少于 2 项时直接
返回、campaign 在**本进程内**运行——写 worker 日志的 `launch_design_workers` 从不执行。
副作用：BC2 自己的 per-worker 显存份额守卫（`XLA_PYTHON_CLIENT_MEM_FRACTION`）也随之失效，
整个 campaign 的显存不受该守卫约束（这正是 §运行约束里要按 N 预算显存的原因）。

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

上游按四种布局查找：`<dir>/params/params_<model>.npz`、`<dir>/params_<model>.npz`、
`<dir>/params/<model>.npz`、`<dir>/<model>.npz`；服务用
`BINDCRAFT2_ALPHAFOLD_PARAMS_DIR` 指定，并把它注入子进程的 `BINDCRAFT_AF2_PARAMS`
（**必须**——否则上游会去下载 5.3 GB）。

pre-stage：

```bash
# 方式 A：NAS 上已有 alphafold-server 的参数 → 软链，零下载
ln -s /data/models/alphafold /data/models/bindcraft2/alphafold
# 也可以不软链：alphafold-server 的 params/ 与本服务要求的布局同构，
# 直接把 BINDCRAFT2_ALPHAFOLD_PARAMS_DIR 指到 /data/models/alphafold 即可。

# 方式 B：下载（只解出这 7 个文件）
WEIGHTS_DST=/data/models/bindcraft2/alphafold \
    ./services/bindcraft2-server/scripts/fetch_weights.sh
```

ProteinMPNN 三变体**随包**分发（vendor 树里每个 `weights_{neutral,negative,positive}/` 目录含
4 个 checkpoint，`du -sh` 26 MB/变体、整棵 `weights/` 77M）；本服务只用
`v_48_020.npz`（每份 6,681,030 B）。由 `vendor.sh` 校验、Dockerfile 构建期自检。

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
| `BINDCRAFT2_DISK_LIMIT_MB` | `1048576` | job 目录总占用超过它时框架删除**已结束**任务；框架默认 8000 MB 会在下一请求里删掉 `rank`/`filter` 要 chain 的源 campaign，故按 NAS 容量放宽到 1 TiB |
| `BINDCRAFT2_ROOT` | `/opt/bindcraft` | 上游源码树，也是子进程 cwd；`Popen(cwd=...)` 要求它是已存在目录（上游的 `settings/` / preset 定位用绝对路径，与此无关） |
| `BINDCRAFT2_SHIPPED_WEIGHTS_DIR` | `/opt/bindcraft` | ProteinMPNN 权重根；探针查 `<它>/bindcraft/weights/proteinmpnn/weights_<变体>/v_48_020.npz` |
| `BINDCRAFT2_PYTHON` | `/opt/venv/bin/python` | 解释器 |
| `BINDCRAFT2_MODULE` | `bindcraft.cli` | 上游 CLI 模块；置空则只跑 `python <args>` |
| `BINDCRAFT2_ALPHAFOLD_PARAMS_DIR` | `/data/models/bindcraft2/alphafold` | AF2 参数目录 |
| `BINDCRAFT2_COMPILE_CACHE_DIR` | `/data/models/bindcraft2/xla_cache` | JAX 图缓存（全卡型共用，见"冷启动"） |
| `BINDCRAFT2_MAX_CONCURRENT_JOBS` | `1` | **submit/poll 路径**的并发上限；`/api/tasks/*` 不受它约束（框架 task 路径没有容量检查），那条路径靠 FC `instanceConcurrency=1` + `sessionConcurrencyPerInstance=1` 串行化 |
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
