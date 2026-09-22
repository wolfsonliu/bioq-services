# bindcraft2-server 设计

- **日期**：2026-09-22
- **状态**：设计（未实现；v0.0.1 范围已定）
- **适用**：把 BindCraft2（BC2）包装成 bioq 服务。读者是后续实现该服务的 agent / 人；
  同时是"能否为 BindCraft2 开发服务"这一可行性问题的结论载体——许可证、运行时长、
  显存三大约束的处置方式都记在本文，尤其是 §2。
- **相关**：[新增 service cookbook](../adding-a-new-service/index.zh.md) ·
  [跨服务 URI 字段命名](./2026-08-18-cross-service-uri-field-naming-design.md) ·
  [THIRD_PARTY_NOTICES](../../THIRD_PARTY_NOTICES) · [services.yaml](../../services.yaml)

## 概述

[BindCraft2](https://github.com/PacesaLab/BindCraft2)（PacesaLab / UZH，v1.0.1）是一体化的
**蛋白 binder 设计 campaign** 套件：AlphaFold 2 hallucination 设计 → ProteinMPNN 重设计 →
用独立的 AlphaFold 模型验证 → 结构过滤器验收与排序。它覆盖 de novo binder、large_binder、
线性/环肽、同源多聚体、多结构域、VHH、ARP、scFv、Fab，以及构象设计（induced_fit /
fold_switch），并支持 hotspot 定向、humanization、蛋白酶抗性、二硫键、末端朝向等 9 个可选属性。

本服务把它包装为 `services/bindcraft2-server/`：一次 `design` 请求 = 一场完整 campaign，
产出可直接送实验的 ranked 序列与预测复合物结构。

**与已有服务的边界**（避免 scope 混淆）：

| 已有服务 | 与本服务的区别 |
|---|---|
| [rfdiffusion-server](../../services/rfdiffusion-server/) / [proteinmpnn-server](../../services/proteinmpnn-server/) / [rfantibody-server](../../services/rfantibody-server/) | 它们是**单步工具**，由客户端自行串联；BC2 是**自成闭环的 campaign**。BC2 内部的 worker 编排、autotuning、desperation ladder、resume 不拆开暴露——拆开等于重写上游编排层，与 1.x 的快速演进正面冲突 |
| [boltzgen-server](../../services/boltzgen-server/) | 同属 binder 设计，但走 Boltz 路线、没有 AF2 hallucination 阶段。两者是**并行可选的两条路线**，不是替代关系 |
| [dockq-server](../../services/dockq-server/) / [plip-server](../../services/plip-server/) | 提供结构度量；BC2 的 `score` 子命令不与它们竞争，因此本服务**不包装 `score`** |

## 许可证与合规约束（红线）

BC2 **不是** OSI 开源许可证，而是 *BindCraft2 Source-Available License
(Hosting-Restricted)*（Copyright (c) 2026 Martin Pacesa, University of Zurich）。

**允许**（本服务的落地依据）：

- 组织内部任意目的使用，包括商业内部的资产开发与药物发现；
- 在本地、私有基础设施、私有云，或"为该使用组织单独运营的基础设施"上运行；
- 由**自己的**自动化 / agent 工具链驱动执行（许可证原文明确把 internal use including
  execution directed by your own automated or agentic tooling 排除在 Hosted Service 之外）；
- 把**设计产物**（序列、结构）作为结果或服务交付给他人。

**禁止**（两条红线，必须写进 `services/bindcraft2-server/README.md`）：

1. **不得对外提供工具功能**。未经 UZH 单独商业许可，不得把本软件、其修改版，或"其实质功能
   由本软件提供/派生"的服务，作为 hosted / managed / cloud / API / web application /
   workflow platform / SaaS 提供给第三方（Third Parties = 非本人、非关联方、非"为你内部目的
   工作的员工或承包商"）。许可证**点名**了本项目的形态："Making the Software available to
   Third Parties as an invocable tool, plugin, agent action, connector, or workflow step
   within a hosted platform"——即通过网关把本服务暴露成 agent 可调用动作，一旦触达第三方即
   构成 Hosted Service。
   同理，**不得**把预配置好的部署（容器 / VM 镜像 / 安装器）分发给第三方去托管。
2. **命名限制**。不得用 "BindCraft2" 或混淆性近似的名字去标识、宣传、描述 Hosted Service /
   修改版 / 衍生作品，以免暗示与上游等效或存在官方关联。许可证**明确允许**"本工作基于 /
   派生自 / 兼容 BindCraft2"这类准确陈述——`bindcraft2-server` 这个仓库内命名与 README 的
   表述落在此允许范围内。

**合规动作**：在 [THIRD_PARTY_NOTICES](../../THIRD_PARTY_NOTICES) 的 License summary 表里
按字母序（`alphafold-server` 之后、`bindflow-server` 之前）新增一行，License 列写明
`Source-Available (Hosting-Restricted) — 仅限内部使用，禁止作为 Hosted Service 对外提供`；
并在服务 README 顶部放同样的红线说明与上游许可证链接。

## 设计目标

1. **纯 argv 包装**：不 patch、不 import 上游，`upstream/` 保持 pristine；wrapper 只负责把
   pydantic 请求编译成 campaign JSON 并构造命令行。上游升级 = 改 pin + 重跑测试。
2. **单 campaign 为主入口**：`design` 是核心；`rank` / `filter` 是**零 GPU 成本的再加工**
   入口（对已有 campaign 目录换指标、换阈值），让 agent 不必重跑设计就能迭代候选集。
3. **权重全部外置 NAS**：5.3 GB AlphaFold 参数不烘焙进镜像；ProteinMPNN 三变体（~20 MB）
   随包，是约束 7 的 <100 MB 例外，Dockerfile 注释写明理由。
4. **无界 campaign 必须被兜底**：上游 `number_of_final_designs` 只约束"收够 N 个 accepted"，
   对尝试次数**无上限**。服务侧强制 `max_trajectories`，防止把 FC 时间额度烧穿。
5. **GPU 静默降级必须可探测**：镜像内没有 `nvidia-smi`，JAX 取不到卡时会**静默降到 CPU**
   （慢约两个数量级且不报错）。`/healthz/detail` 必须暴露真实后端。
6. **双模式对齐**：HTTP submit/poll、`/api/tasks/*`、`python -m server <endpoint>` 三条路径
   共享同一套 `tools.py` argv builder 与 `adapter.detect_outputs`。

## Endpoint 拓扑

| Endpoint | 说明 |
|---|---|
| `POST /api/design` | 提交一场 campaign（submit/poll 模式） |
| `POST /api/tasks/design` | 同上，FC 异步任务模式（**推荐入口**，campaign 为小时级） |
| `POST /api/rank` | 对已有 campaign 目录按指定指标重排 → `ranked_by_<metric>.csv` |
| `POST /api/tasks/rank` | 同上，task 模式 |
| `POST /api/filter` | 对已有 campaign 目录重放 / 覆盖 acceptance 阈值 → `filtered.csv` |
| `POST /api/tasks/filter` | 同上，task 模式 |

**v0.0.1 明确不做**（防止 scope creep）：

- `score`（单结构度量，与 `dockq-server` / `plip-server` 边界模糊）；
- `archive` / `unarchive`（NAS 上无意义）、`fetch-weights`（权重在构建期就位）、
  `campaign_output`（重建摘要）、参数 sweep；
- **多 target**：同源对（`"target": ["hPDL1","mPDL1"]`）、detarget（`"objective":
  "detarget"`）、多链受体装配。v0.0.1 只支持单 target；
- **自定义 scaffold**（VHH / ARP / Fab 用上游自带 `scaffolds/`）；
- **wall-clock 强杀**（依赖上游对 SIGINT 的收尾行为，未验证）；
- **跨 job resume**（见 §11 风险）；
- **实时百分比进度**（需要 framework 改动，不在本服务范围内）。

## 请求 Schema

文件输入走路由层 `File(...)` / `Form(...)`，**不放在 model 上**。`binder_lengths` / `on` /
`where` 是复杂类型，按 framework 约定以 **JSON 字符串**表单字段传入
（`-F 'binder_lengths=[80,80]'`，见 [framework/forms.py](../../framework/src/bioq_service/forms.py)）。

### DesignRequest — `POST /api/design`

| 字段 | 类型 | 默认 | 约束 | 说明 |
|---|---|---|---|---|
| `target_name` | str? | None | 与 `target` / `target_uri` 三选一 | 上游 shipped target 名（`bindcraft design --list-targets`） |
| `target` | UploadFile? | None | 同上 | PDB / mmCIF / FASTA 目标结构 |
| `target_uri` | str? | None | 同上 | 上传字段的 URI 孪生：`job://` `file://` `oss://` `http(s)://` |
| `target_chains` | str? | None | 如 `A` / `A,B` | 目标链选择（`targets[].chains`） |
| `hotspots` | str? | None | 如 `54,56,66-70` | 结合位点残基，编号取自输入结构 |
| `coldspots` | str? | None | | 要求保持自由的区域 |
| `modality` | str | `binder` | 上游 modality 名，允许逗号组合 | 组合合法性交上游 preflight 校验，不在此处枚举 |
| `binder_lengths` | list[int]? | None | 长度 1–2；`lo<=hi` | `[80,80]` 定长 / `[60,100]` 范围；scaffold 模态（VHH / ARP / scFv / Fab）不传 |
| `number_of_final_designs` | int | `10` | 1–1000 | 收够 N 个 accepted 即停 |
| `max_trajectories` | int? | None → `settings.default_max_trajectories` | ≥1 | **服务侧新增**，见设计目标 4 |
| `forced_targeting` | bool | false | | 对应上游 `--forced-targeting` |
| `humanize` | bool | false | | 对应 `--humanize` |
| `protease_stable` | bool | false | | 对应 `--protease-stable` |
| `disulfide_staple` | bool | false | | 对应 `--disulfide-staple` |
| `mixed_topology` | bool | false | | 对应 `--mixed-topology` |
| `termini_together` | bool | false | | 对应 `--termini-together` |
| `termini_accessible` | bool | false | | 对应 `--termini-accessible` |
| `initial_guess` | bool | false | | 对应 `--initial-guess` |
| `bigbang` | bool | false | | 对应 `--bigbang` |
| `core` | str? | None | | 上游 core profile（如 `benchmark`）；未设则用上游 baseline |
| `campaign_name` | str? | None → `job_id` | `[A-Za-z0-9_-]{1,64}` | 结果命名与 `campaign_metadata.json` 元数据 |

9 个属性用**显式 bool** 而非枚举 list：与上游 campaign JSON 的顶层 key 一一对应，OpenAPI
自解释，零翻译歧义（新增上游属性时的成本由"加一个 bool 字段"承担，可接受）。

`model_validator(mode="after")` 交叉校验：`target_name` / `target` / `target_uri` **恰好一个**；
`binder_lengths` 若给必须是 1 或 2 个正整数且升序。

### RankRequest — `POST /api/rank`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `campaign_uri` | str | **必填** | 源 campaign 目录：`job://<id>`、`job://<id>/output`、`file:///abs` 或裸绝对路径 |
| `on` | list[str] | `["i_pDAE"]` | 排序指标；多个用于 tie-break。取值见 `bindcraft rank <dir> --list` |
| `lowest_first` | bool? | None | 为 None 时用上游自动方向判定 |
| `table` | enum | `accepted` | `accepted` \| `candidates` \| `trajectories` |
| `top` | int? | None | 限制控制台显示行数 |

### FilterRequest — `POST /api/filter`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `campaign_uri` | str | **必填** | 同 RankRequest |
| `where` | list[str]? | None | 阈值表达式，1:1 透传上游：`["i_pAE=0.45","Interface_Residues>=7"]`。`METRIC=VALUE` 走该指标配置方向，`>=` / `<=` 显式指定方向；无 target 后缀的阈值作用于所有 binding target 并排除 detarget 状态 |
| `table` | enum | `candidates` | `candidates` \| `accepted` \| `trajectories` |
| `top` | int? | None | 限制控制台显示行数 |

`where` 为空时等价于"用 campaign 自身的阈值重放"，与上游 `bindcraft filter <dir>` 一致。

两条 URI 约定（约束 6）：上传字段名与 URI 字段名严格配对（`target` ↔ `target_uri`）；
`campaign_uri` 是**目录**而非上传字段，故不带 `_uri` 后缀的孪生上传（详见 §7 零拷贝解析）。

## 输出

`design` job：

```
<jobs_base_dir>/<job_id>/
├── input/
│   ├── campaign.json            # 服务拼装的 campaign 文件（唯一真源，含 resolved 设置）
│   └── target.pdb               # 上传 / URI 落盘结果；target_name 路径下不存在
├── output/                      # == 上游 project_folder（绝对路径写进 campaign.json）
│   ├── 1_Trajectories/!_Trajectories.csv    # 每次尝试一行，含 terminated / autotuned
│   ├── 2_Refolded/!_Refolded.csv            # 每个 ProteinMPNN 候选一行，含 failed_filters
│   ├── 3_Ranked/!_Ranked.csv                # 主产物：accepted，best-first by i_pDAE
│   ├── 3_Ranked/<design>_seq<n>.cif         # accepted 预测复合物
│   ├── summary.csv                          # 跨 attempt / stage / candidate / accepted 汇总
│   ├── campaign_metadata.json               # resolved 设置 + checkpoint SHA-256 + 源 revision
│   ├── workers/worker_<NN>_gpu_<id>.log     # 每个 worker 的完整日志（单个 worker 失败时看它）
│   └── .campaign_state.json                 # 续跑与 worker 协调状态
└── logs/run.log                 # 子进程 stdout+stderr（框架 tee）
```

`rank` / `filter` job 的 `output/` 只有 `ranked_by_<metric>.csv` / `filtered.csv`（一律用上游
`--output` 重定向到本 job 的 `output/`，**不写回源 campaign**）。

`detect_outputs()` 判据 = 下列任一存在且非空：

```
output/3_Ranked/!_Ranked.csv
output/summary.csv
output/ranked_by_*.csv
output/filtered.csv
```

⚠️ **必须包含 `summary.csv`**：一场合法完成但 accepted = 0 的 campaign 不会写
`!_Ranked.csv`，只认它会把这个成功任务误判为 FAILED。放宽是安全的——`rc != 0` 仍由框架
`finalize_job` 判失败，`summary.csv` 只在 rc == 0 的前提下才被当作成功证据。

## 实现要点

### 包装 vs patch 决策

| 候选 | 选择 | 理由 |
|---|---|---|
| 包装 CLI（`python -m bindcraft.cli design <json>`） | **采用** | 上游 CLI 是稳定公开面；campaign JSON 是官方输入契约；`upstream/` 保持 pristine |
| patch 上游以暴露 trajectory 级流式进度 / 中途取消 | **不做** | 需要扎进 worker 内部结构，与 1.x 快速演进冲突面最大；v0.0.2 再评估 |
| import 上游 Python API（`from bindcraft import launch_campaign`） | **不做** | 在服务进程内 import 会把 JAX runtime 拉进 HTTP 进程并占用 campaign 需要的显存 |

### 关键实现点

| 决策点 | 选择 |
|---|---|
| 入口 argv | `[settings.python, "-m", "bindcraft.cli", "design", str(job_dir/"input"/"campaign.json")]`，`cwd = job_dir` |
| campaign 拼装 | `tools.build_campaign_json(req, job_dir)`：`target_name` → `"target": <name>`；上传 → `targets: [{name: req.target_name or "target", target_path: <绝对路径>, chains, hotspots, coldspots}]`（`name` 由服务生成，**不取用户文件名**，避免非法字符进入结果命名）；9 个 bool → 顶层同名 key；`project_folder` / `binder_lengths` / `number_of_final_designs` / `max_trajectories` / `core` / `campaign_name` 服务侧写入 |
| 输入落盘后缀 | 保留 `UploadFile.filename` 的后缀（`.pdb` / `.cif` / `.fasta` / `.fa`），无法识别时用 `.pdb`；`target_uri` 路径同样按 URI 后缀，无法判定时用 `.pdb`。上游按后缀/内容识别格式，服务侧不嗅探内容 |
| rank / filter argv | `[python, "-m", "bindcraft.cli", "rank", <campaign_dir>, "--on", *on, "--table", table, "--output", str(out)]` / `[... , "filter", <campaign_dir>, "--where", *where, "--output", str(out)]`；`--lowest-first` / `--top` 按需追加 |
| campaign 目录解析 | 新增 `services/bindcraft2-server/campaigns.py`：**零拷贝**返回目录路径。`job://<id>` / `job://<id>/output` → `settings.jobs_base_dir/<id>/output`；`file:///abs` / 裸 `/abs` → 原路径。**不能复用 `bioq_service.uris.resolve_uri`**——它是文件级且 `shutil.copy2`，campaign 目录可达数百 MB，拷贝不可接受 |
| `detect_outputs()` | 见 §输出 的 any-of 判据 |
| `infer_job_from_dir()` | 恢复时报告 accepted 行数（`3_Ranked/!_Ranked.csv` 数据行）；无该文件则报告 `1_Trajectories` 尝试数并注明 accepted=0 |
| 进度可见性 | v0.0.1 **不做**实时百分比。agent 通过 `/log` 尾部与 `job://<id>/1_Trajectories/!_Trajectories.csv` 行数判断进展 |
| 权重（AF2） | 外置 NAS：`settings.alphafold_params_dir` 下的 `params/params_model_{1..5}_multimer_v3.npz` + `params_model_{1,2}_ptm.npz`（7 个）。启动时**必须**设 `BINDCRAFT_AF2_PARAMS`，否则上游会去下载 5.3 GB |
| 权重（ProteinMPNN） | 随包：`upstream/bindcraft/weights/proteinmpnn/{neutral,negative,positive}/v_48_020.npz`（~20 MB，约束 7 的 <100 MB 例外）。**vendor.sh 的排除规则不得剔掉该子树** |
| 编译缓存 | `JAX_COMPILATION_CACHE_DIR=settings.compile_cache_dir`。镜像无 `nvidia-smi`，否则上游把图缓存丢进 `${TMPDIR:-/tmp}`，每次冷启动重付约 60 s/预测形状的编译成本 |
| GPU 可见性 | Dockerfile 设 `NVIDIA_VISIBLE_DEVICES` / `NVIDIA_DRIVER_CAPABILITIES`，并写 `/etc/ld.so.conf.d/bindcraft-cuda.conf` + `ldconfig`（jax cuda13 wheel 的库在 `site-packages/nvidia/*/lib`，loader 默认找不到） |
| GPU 探针 | `/healthz/detail` 用**子进程**探测 `python -c "import jax; print(jax.default_backend()); print(jax.devices())"`，结果进程内缓存（TTL 5 min）。**绝不在 HTTP 进程内 import jax** |
| 并发 | `max_concurrent_jobs = 1`。BC2 会吃掉所有可见卡并按显存 pack worker；FC 一实例一卡，必须独占 |
| worker 覆盖 | `workers_per_gpu = 1` / `max_workers_per_gpu = 1`。BC2 的宿主内存上限读**节点**可用内存而非 job 限额，在 FC 上会过度规划 worker 导致 OOM |
| `manifest_extras()` | `tool_outputs`（三阶段 CSV 路径）+ `input_uri_schemes` + `campaign_knobs`（modality / property 名清单，来自上游 preset 目录）+ `weights`（7 个 npz 的期望文件名） |
| `endpoint_examples()` | 6 个 endpoint 各 ≥1 条可跑 curl，含 `design` 的 shipped-target 路径与上传路径、`rank`/`filter` 接续上一 job 的 `campaign_uri=job://<id>` |
| `/healthz/detail` | 自定义实现（`_strip_route` 摘掉 framework 默认路由），字段见下 |

`/healthz/detail` 返回字段：`status` / `service` / `version` / `weights_dir` /
`weights_loaded` / `weights_missing`（缺失的 npz 文件名列表）/ `proteinmpnn_weights_loaded` /
`gpu_backend`（`"gpu"` / `"cpu"` / `"probe_failed"`）/ `gpu_devices` /
`active_jobs` / `max_concurrent_jobs`。权重缺失时 HTTP 200 + `weights_loaded=false`，
**不在 import 期 raise**；`gpu_backend != "gpu"` 时附一条 warning 字符串，因为这是唯一能
暴露"静默跑 CPU"的信号。

## 配置

`env_prefix = BINDCRAFT2_`（全走 pydantic-settings，无 `os.getenv`）

| 字段 | 默认 | 说明 |
|---|---|---|
| `jobs_base_dir` | `/data/bindcraft2_jobs` | 任务目录根 |
| `root` | `/opt/bindcraft` | 上游源码树；editable install 依赖它存在 |
| `python` | `/opt/venv/bin/python` | 解释器（离线测试指向 stub） |
| `alphafold_params_dir` | `/data/models/bindcraft2/alphafold` | 含 `params/` 下 7 个 npz；见 §待确认 D2 |
| `compile_cache_dir` | `/data/models/bindcraft2/xla_cache` | `JAX_COMPILATION_CACHE_DIR` |
| `max_concurrent_jobs` | `1` | 整实例独占 GPU |
| `workers_per_gpu` | `1` | 覆盖 BC2 自动 pack |
| `max_workers_per_gpu` | `1` | 同上 |
| `default_max_trajectories` | `500` | `max_trajectories` 未给时的兜底 |

## 部署目标

- FC **GPU** 实例（custom-container），`timeout: 36000`（对齐
  [alphafold-server](../../services/alphafold-server/) / [boltzgen-server](../../services/boltzgen-server/)）。
- 控制台：开启**异步任务模式**、清空 keepalive URL、`sessionAffinity` 用 header
  `bioagent-session-id` 且 `sessionConcurrencyPerInstance: 1`。
- 挂载：NAS `mountDir: /data`；OSS `mountPoints` → `/mnt/oss`（有文件输入，经网关调用时
  需要 `oss://` 输入改写为 `/mnt/oss/...` 直读）。
- 主机规格：建议 8 vCPU / 32 GB。显存按单 worker 预算
  `2.0 × (3.4 GB + 38 kB × N²)`（N = padding 后的复合物残基数）估算：
  N=256 → ~11.8 GB，N=384 → ~18.0 GB，N=512 → ~26.7 GB，另需留 4 GB headroom。
- 卡型（**待确认 D6**）：`cuda13` extra 要求 compute capability ≥ 7.5，T4(7.5) / A10(8.6) /
  L20(8.9) / A100(8.0) 均可；V100(7.0) 及更老必须改用 `cuda12` extra。
  建议 **L20 48 GB 或 A10 24 GB**；T4 16 GB 为最小可用配置（只能跑 1 worker，
  256+ 残基的 target 容易 OOM）。
- 镜像：Ubuntu 24.04 多阶段，uv venv + `pip install -e "/opt/bindcraft[cuda13]"` + framework。
  jax cuda13 + cuDNN + cuequivariance 合计数 GB，需要复用 boltzgen 的瘦身路线
  （删 `*.a`、`share/doc|man|locale`、`__pycache__`、`*.pyc`）。**不烘焙** 5.3 GB AF2 参数。
- [services.yaml](../../services.yaml) 新增条目：`bindcraft2-server:` + `url` + `tier: warm`
  + `function` + `gpu` + `oss_mount: true`。

## 测试策略

| 层 | 文件 | 内容 |
|---|---|---|
| offline HTTP | `tests/test_app.py` | `BINDCRAFT2_PYTHON` 指向 stub：`/healthz`、`/api/manifest`（endpoints + examples 齐全）、三端点各一次 submit、以及 422 矩阵（target 三选一违规、`target` 与 `target_name` 同时给、`binder_lengths` 长度/升序违规、`campaign_uri` 缺失） |
| offline CLI | `tests/test_cli.py` | endpoint 注册表完整；`build_campaign_json` 对 shipped / 上传两条路径的逐字段断言；rank / filter argv builder；`create_cli` e2e |
| FC sync | `tests/test_fc.py` | `@pytest.mark.fc`：`/healthz/detail`（断言 `weights_loaded=true` 且 `gpu_backend=="gpu"`）、`/api/manifest`、`design` 小 campaign（`number_of_final_designs=1`、`max_trajectories=1`、`binder_lengths=[60,60]`、`mini_target.pdb`）→ 断言 `summary.csv` 与 `campaign_metadata.json` 存在；再用产物目录跑 `rank` / `filter` |
| FC async | `tests/test_fc_task.py` | `@pytest.mark.fc`：`/api/tasks/{design,rank,filter}` 的 202 / 完成 / 生命周期 / 平台层 dedup |

fixture：`tests/data/mini_target.pdb`（≤60 残基单链，保证 smoke campaign 能跑完）；
`tests/data/fake_bindcraft.sh`（离线成功路径：按 endpoint 写 `output/3_Ranked/!_Ranked.csv`
或 `output/summary.csv` / `ranked_by_<metric>.csv` / `filtered.csv`）。

FC smoke 即使 1 个 trajectory 也可能是十分钟量级——`test_fc.py` 分两级，先跑 health/manifest
（`-k "not design"`），再跑完整 design。

## 成功标准

1. `services/bindcraft2-server/` 满足 [新增 service cookbook](../adding-a-new-service/index.zh.md)
   的必备文件清单与提交清单（`VERSION` / `__main__.py` / `scripts/vendor.sh` / `README.md` /
   tests 五件套 / Dockerfile 内无 `git clone`、无 `COPY .../weights/`）。
2. `uvx ruff check services/bindcraft2-server/` 通过；`tests/test_app.py` 与 `tests/test_cli.py`
   离线通过（不需要 GPU）。
3. `make build-bindcraft2-server` 成功；`/api/manifest` 的 6 个 endpoint 各有 ≥1 条可执行 curl；
   `/api/tasks/*` 与 `/api/*` 路由一一对应。
4. FC 部署后 `/healthz/detail` 返回 `weights_loaded: true` 且 `gpu_backend: "gpu"`。
5. `pytest -m fc services/bindcraft2-server/tests/test_fc.py`：`design` 小 campaign 端到端产出
   `summary.csv` + `campaign_metadata.json`，随后 `rank` / `filter` 在该产物上各自产出目标 CSV。
6. `pytest -m fc services/bindcraft2-server/tests/test_fc_task.py` 通过（三个 task endpoint）。
7. [services.yaml](../../services.yaml) 有条目；[THIRD_PARTY_NOTICES](../../THIRD_PARTY_NOTICES)
   有 BC2 行；服务 README 含 §2 的两条红线。

## 部署前置核验（决定规则已定）

实现与部署前必须核验下列环境事实。每条只决定走哪一支，处置规则已确定，不留设计空白。

| # | 核验项 | 处置规则 |
|---|---|---|
| P1 | NAS `/data/models/alphafold` 是否已由 [alphafold-server](../../services/alphafold-server/) 就位，且含 BC2 需要的 7 个 npz | 就位 → `alphafold_params_dir` 软链到它，零额外下载；缺失 → 跑 `scripts/fetch_weights.sh`，从 `alphafold_params_2022-12-06.tar` 只解出 `params_model_{1..5}_multimer_v3.npz` 与 `params_model_{1,2}_ptm.npz` |
| P2 | FC 可分配 GPU 卡型的 compute capability | cc ≥ 7.5 → `cuda13` extra（T4 / A10 / L20 / A100）；cc < 7.5 → `cuda12` extra（V100 及更老）。worker 与实例内存参数按 §部署目标 的显存公式定 |
| P3 | `rank` / `filter` 是否只写 `--output` 指向的文件 | 记录源 campaign 目录的文件清单与 mtime；未变 → 保持零拷贝；有变 → 改为"拷贝源目录到 `<job_dir>/input/campaign/` 再操作" |

## 风险 / 限制

- **运行时长天花板**：FC `timeout: 36000`（10 h）是硬上限。`max_trajectories` 是唯一主动
  兜底；超时后任务 FAILED，但 `1_Trajectories/`、`2_Refolded/` 的中间结果仍在 NAS 上，
  可人工挽救。v0.0.1 **无自动 resume**——框架每次 submit 建新 job_dir，BC2 原生的
  `.campaign_state.json` 续跑机制跨 job 失效。v0.0.2 候选方案：新增
  `resume_from_uri`，把源 campaign 的 `output/` 作为 `project_folder` 复用。
- **上游 1.x 演进快**：`settings/core/reference.json` 的 key 会增删；服务只映射稳定子集，
  未知 key 无透传通道（不提供 `--set` 直通，避免绕过类型契约）→ 需要时发版跟进。
- **`campaign_uri` 零拷贝的副作用**：上游 `rank` / `filter` 除了写 `--output`，可能仍往源
  campaign 写 cache / `.partial` / 状态文件。若 FC 测试观察到源目录被改，退化为"只读副本 +
  copy"（代价是数百 MB 拷贝）。这是本设计里**最需要在 FC 阶段实测验证**的一条。
- **显存**：T4 16 GB 上 N≥256 的 target 只能 1 worker 且容易 OOM；大 target 需要大卡。
- **冷启动**：JAX runtime + 模型编译。`JAX_COMPILATION_CACHE_DIR` 指向 NAS 可缓解，但缓存
  **按卡型分目录**，跨卡型不可复用；scale-to-zero 后首个请求仍要付 runtime 启动成本。
- **许可证**：见 §2。红线必须随 README 一起交付；对外暴露本服务前需先取得 UZH 商业许可。
- **命名**：服务与镜像名不得暗示与 UZH 的官方关联；README 只能做"基于 / 兼容 BindCraft2"
  的准确陈述。

## Sources

- 上游：[github.com/PacesaLab/BindCraft2](https://github.com/PacesaLab/BindCraft2) — pin
  **v1.0.1** = `5342aefa18dedad653f7a5f6dbee1e566ca24d8f`（default branch `main`）
- 许可证：*BindCraft2 Source-Available License (Hosting-Restricted)*,
  Copyright (c) 2026 Martin Pacesa, University of Zurich
- 上游文档：`README.md`（modality / property / input tiers）、`docs/installation.md`
  （权重、环境变量、容器、Slurm）、`docs/outputs.md`（输出文件、度量、rank/filter）
- 上游代码：`bindcraft/model_weights.py`（`CAMPAIGN_MODELS` / 参数查找路径）、
  `pyproject.toml`（`cuda12` / `cuda13` extras）、`containers/Dockerfile`（CUDA 与
  `nvidia-smi` 缺失的坑）
- 参考实现：[boltzgen-server](../../services/boltzgen-server/)（权重外置 + 镜像瘦身）、
  [alphafold-server](../../services/alphafold-server/)（JAX/AF2 + NAS 参数 + 长 GPU 任务 +
  keepalive）、[seqkit-server](../../services/seqkit-server/)（本设计文档的结构样板）
