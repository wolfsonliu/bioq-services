# bindcraft2-server 设计

- **日期**：2026-09-22
- **状态**：已实现（v0.0.1，镜像 tag 见 [`services/bindcraft2-server/VERSION`](../../services/bindcraft2-server/VERSION)）
- **适用**：把 BindCraft2（BC2）包装成 bioq 服务。读者是维护该服务的 agent / 人；
  同时是"能否为 BindCraft2 开发服务"这一可行性问题的结论载体——许可证、运行时长、
  显存三大约束的处置方式都记在本文，尤其是 §2。**本文已按 Task 13 回写为"实测 / 实现"口径**：
  每条事实都对应到文件或命令；仍未核实的条目集中在 §部署前置核验与 §已知限制。
- **相关**：[新增 service cookbook](../adding-a-new-service/index.zh.md) ·
  [跨服务 URI 字段命名](./2026-08-18-cross-service-uri-field-naming-design.md) ·
  [THIRD_PARTY_NOTICES](../../THIRD_PARTY_NOTICES) · [services.yaml](../../services.yaml) ·
  [实现计划](../plans/2026-09-22-bindcraft2-server.md)
- **引用约定**：文中「约束 N」指 [AGENTS.md](../../AGENTS.md) 的 Hard Constraints 编号
  （约束 6 = URI 字段 `_uri` 配对，约束 7 = 权重外置 NAS、<100 MB 才可随包）。

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

**合规动作（已完成）**：在 [THIRD_PARTY_NOTICES](../../THIRD_PARTY_NOTICES) 的 License
summary 表里按字母序（`alphafold-server` 之后、`bindflow-server` 之前）新增了一行，License
列写明 `Source-Available (Hosting-Restricted) — 仅限内部使用，禁止作为 Hosted Service 对外提供`；
服务 README 顶部放了同样的红线说明与上游许可证链接。

## 设计目标

1. **纯 argv 包装**：不 patch、不 import 上游，`upstream/` 保持 pristine；wrapper 只负责把
   pydantic 请求编译成 campaign JSON 并构造命令行。上游升级 = 改 pin + 重跑测试。
2. **单 campaign 为主入口**：`design` 是核心；`rank` / `filter` 是**零 GPU 成本的再加工**
   入口（对已有 campaign 目录换指标、换阈值），让 agent 不必重跑设计就能迭代候选集。
   ⚠️ "零拷贝"是有条件的，见 §已知限制与 §部署前置核验 P3。
3. **权重全部外置 NAS**：5.3 GB AlphaFold 参数不烘焙进镜像；ProteinMPNN 权重随包
   （vendor 树里三个变体各 4 个 checkpoint，`du -sh` 每个变体目录 26 MB、整棵 `weights/` 77M；
   本服务只用 `v_48_020.npz`，每份 6,681,030 B），是约束 7 的 <100 MB 例外，
   Dockerfile 注释写明理由。
4. **无界 campaign 必须被兜底**：上游 `number_of_final_designs` 只约束"收够 N 个 accepted"，
   对尝试次数**无上限**。服务侧强制 `max_trajectories`（请求上界 100000，未给时用
   `default_max_trajectories=500`），防止把 FC 时间额度烧穿。
5. **GPU 静默降级必须可探测**：镜像内没有 `nvidia-smi`，JAX 取不到卡时会**静默降到 CPU**
   （慢约两个数量级且不报错）。`/healthz/detail` 必须暴露真实后端。
6. **双模式对齐**：HTTP submit/poll、`/api/tasks/*`、`python -m server <endpoint>` 三条路径
   共享同一套 `tools.py` argv builder 与 `adapter.detect_outputs`。

## Endpoint 拓扑

| Endpoint | 说明 |
|---|---|
| `POST /api/design` | 提交一场 campaign（submit/poll 模式） |
| `POST /api/tasks/design` | 同上，FC 异步任务模式（**推荐入口**，campaign 为小时级） |
| `POST /api/rank` | 对已有 campaign 目录按指定指标重排 → `ranked_by_<m1>[_<m2>...].csv` |
| `POST /api/tasks/rank` | 同上，task 模式 |
| `POST /api/filter` | 对已有 campaign 目录重放 / 覆盖 acceptance 阈值 → `filtered.csv` |
| `POST /api/tasks/filter` | 同上，task 模式 |

`rank` 产物文件名由 `tools.rank_output_name()` 生成：**所有** `on` 指标按顺序拼接、
非字母数字折成 `_`（`on=["i_pTM","i_pDAE"]` → `ranked_by_i_pTM_i_pDAE.csv`），与上游
`rank.ranked_filename()` 同口径。只用第一个指标会让不同 tie-break 的产物撞名。

**v0.0.1 明确不做**（防止 scope creep）：

- `score`（单结构度量，与 `dockq-server` / `plip-server` 边界模糊）；
- `archive` / `unarchive`（NAS 上无意义）、`fetch-weights`（权重在构建期就位）、
  `campaign_output`（重建摘要）、参数 sweep；
- **多 target**：同源对（`"target": ["hPDL1","mPDL1"]`）、detarget（`"objective":
  "detarget"`）、多链受体装配。v0.0.1 只支持单 target；
- **自定义 scaffold**（VHH / ARP / Fab 用上游自带 `scaffolds/`）；
- **wall-clock 强杀**（依赖上游对 SIGINT 的收尾行为，未验证）；
- **跨 job resume**（见 §已知限制）；
- **实时百分比进度**（需要 framework 改动，不在本服务范围内）。

## 请求 Schema

文件输入走路由层 `File(...)` / `Form(...)`，**不放在 model 上**。因此目标结构"三选一"由
`models.validate_target_selection()` 这个**普通函数**在路由里调用，而**不是**
`model_validator`——上传参数不是 model 字段，`model_validator` 看不到它们
（`models.py:77`，路由侧调用见 `app.py:247` 与 `app.py:312`）。

### HTTP 与 CLI 的复杂字段编码契约不同（A1）

`binder_lengths` / `on` / `where` 是复杂类型（list），两条入口路径接受的写法**不一样**：

| 路径 | 机制 | 合法写法 | 非法写法 |
|---|---|---|---|
| HTTP 表单 | `framework/src/bioq_service/forms.py::model_form_depends` 在模型校验**之前**对复杂字段 `json.loads` | `-F 'binder_lengths=[80,80]'`、`-F 'on=["i_pTM"]'` | `-F binder_lengths=80,80` → **422 json_invalid**（`app.py` 的测试 `test_design_rejects_invalid_json_in_complex_field` 钉住了它） |
| CLI | `framework/src/bioq_service/cli.py::_add_model_args` 对除 bool/int/float 外一律 `type=str`，从不做 JSON 解析，原样交给 `model_validate` | `--binder-lengths 80,80`、`--on i_pTM,i_pDAE`；**JSON 写法也照收**（`--binder-lengths '[80,80]'` 一样成立） | 无——CLI 上两种写法都合法。区别只在 HTTP 侧**只**认 JSON |

模型里的 `field_validator(mode="before")`（`models.py:169`）同时解码两种写法，**但 HTTP 上逗号
分支永远不会被用到**（非法 JSON 在进模型前就已 422）。因此：**凡是 HTTP 示例/测试
（`endpoint_examples()`、README、`test_app.py`、`test_fc*.py`）里的复杂字段都必须写成 JSON
字符串**；逗号写法只出现在 CLI 示例与 CLI 测试里。

另注：**CLI 的 `--on` 不支持重复传参**——`_add_model_args` 只注册一个 `--on` flag，重复传时
argparse 后者覆盖前者（实测 `--on i_pTM --on i_pDAE` 最终 `on == ["i_pDAE"]`）。CLI 上多指标
必须写成一个逗号串 `--on i_pTM,i_pDAE`。

### DesignRequest — `POST /api/design`

| 字段 | 类型 | 默认 | 约束 | 说明 |
|---|---|---|---|---|
| `target_name` | str? | None | 与 `target` / `target_uri` 三选一 | 上游 shipped target 名（`bindcraft design --list-targets`） |
| `target` | UploadFile? | None | 同上 | PDB / mmCIF / FASTA 目标结构（路由级 `File(...)`） |
| `target_uri` | str? | None | 同上 | 上传字段的 URI 孪生：`job://` `file://` `oss://` `http(s)://` |
| `target_chains` | str? | None | **不得与 `target_name` 同用**（422） | 目标链选择（`targets[].chains`）；只在 target 上传 / URI 路径生效 |
| `hotspots` | str? | None | 同上 | 结合位点残基，编号取自输入结构 |
| `coldspots` | str? | None | 同上 | 要求保持自由的区域 |
| `modality` | `str \| list[str]` | `binder` | 单名、逗号串或 JSON 数组 | 逗号组合在 model 里被摊平成 list 落盘（上游 campaign JSON 里的多预设必须是数组）；组合合法性交上游 preflight 校验 |
| `binder_lengths` | list[int]? | None | 长度 1–2；`lo<=hi`；每项 ≥1；**无上界** | `[80,80]` 定长 / `[60,100]` 范围；scaffold 模态（VHH / ARP / scFv / Fab）不传 |
| `number_of_final_designs` | int | `10` | 1–1000 | 收够 N 个 accepted 即停 |
| `max_trajectories` | int? | None → `settings.default_max_trajectories` | 1–100000 | **服务侧新增**，见设计目标 4 |
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

**校验分布**：`binder_lengths` 的形状 / 正负 / 升序由 `model_validator(mode="after")` 校验
（`models.py:244`）；`binder_lengths` 的**元素类型**由 `field_validator(mode="before")` 严格
校验（拒绝 bool、float 与非法字符串——`int(80.5)` 会静默截断成 80，必须报错）。
`target_name` 与 `target_chains` / `hotspots` / `coldspots` 的互斥由
`_reject_target_name_with_target_fields` 校验（`models.py:219`）：shipped preset 自带这三项，
而 `build_campaign_json` 在 `target_name` 分支根本不写 `targets`，同给会被静默丢弃 → 显式 422。

### RankRequest — `POST /api/rank`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `campaign_uri` | str | **必填** | 源 campaign 目录：`job://<id>`、`job://<id>/output`、`file:///abs` 或裸绝对路径。**不接受 `oss://` / `http(s)://`**（422） |
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
`campaign_uri` 是**目录**而非上传字段，故不带 `_uri` 后缀的孪生上传（详见 §零拷贝解析）。

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
│   └── .campaign_state.json                 # 续跑与 worker 协调状态
└── logs/run.log                 # 子进程 stdout+stderr（框架 tee）
```

**不存在 `output/workers/worker_<NN>_gpu_<id>.log`**（旧稿把它列进了产物树，是错的）。本部署
把 `workers_per_gpu=1` / `max_workers_per_gpu=1` 写死，单卡下 worker plan 长度为 1，而上游
`design_workers.dispatch_design_workers` 在 `len(plan) < 2` 时直接 `return None`
（`upstream/bindcraft/design_workers.py:244`）→ campaign 在**本进程内**跑，`launch_design_workers`
（唯一写 `worker_*.log` 的地方）从不执行。副作用：上游的 `XLA_PYTHON_CLIENT_MEM_FRACTION`
per-worker 显存份额守卫也随之失效（`design_worker_memory_fractions` 在 worker 数 <2 时返回
全 0，`design_workers.py:57`）。

`rank` / `filter` job 的 `output/` 只有 `ranked_by_<m1>[_<m2>...].csv` / `filtered.csv`
（用上游 `--output` 重定向到本 job 的 `output/`）。**但"源目录只读"是有条件的**：见 §P3。

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
| 入口 argv | `[settings.python, "-m", "bindcraft.cli", "design", str(job_dir/"input"/"campaign.json")]`，**`cwd = settings.root`**（不是 `job_dir`；见下条） |
| 子进程 cwd 的真实理由 | `upstream/bindcraft/settings.py:41,353` 用 `Path(__file__).parent.parent` 绝对定位 `settings/` 与 preset 树，`read_preset` 也把 `target_path` / `binder_scaffold` 解析成绝对路径——**与 cwd 无关**。`settings.root` 之所以必须是目录，只因为 `Popen(cwd=...)` 要求它存在（不存在直接 ENOENT）；指到一个"存在但无关"的目录在功能上无害。保留 `/opt/bindcraft` 只是与镜像布局一致（editable install 也确实要求源码树在场） |
| campaign 拼装 | `tools.build_campaign_json(req, *, job_dir, target_path, max_trajectories)`：`target_name` → `"target": <name>`；上传/URI → `targets: [{name: "target", target_path: <绝对路径>, chains, hotspots, coldspots}]`（`name` **恒为 `"target"`**，不取用户文件名，避免非法字符进入结果命名）；9 个 bool → 顶层同名 key；`project_folder` / `binder_lengths` / `number_of_final_designs` / `max_trajectories` / `core` / `campaign_name` 服务侧写入 |
| 输入落盘后缀 | 保留 `UploadFile.filename` / URI 的后缀（`.pdb` / `.cif` / `.mmcif` / `.fasta` / `.fas` / `.fa`），无法识别时用 `.pdb`。上游按后缀/内容识别格式，服务侧不嗅探内容 |
| rank / filter argv | `[python, "-m", "bindcraft.cli", "rank", <campaign_dir>, "--on", *on, "--table", table, "--output", str(out)]` / `[... , "filter", <campaign_dir>, "--where", *where, "--output", str(out)]`；`--lowest-first` / `--highest-first` / `--top` 按需追加 |
| campaign 目录解析 | `services/bindcraft2-server/campaigns.py`：**零拷贝**返回目录路径。`job://<id>` / `job://<id>/output` → `settings.jobs_base_dir/<id>/output`；`file:///abs` / 裸 `/abs` → 原路径。**不能复用 `bioq_service.uris.resolve_uri`**——它是文件级且 `shutil.copy2`，campaign 目录可达数百 MB，拷贝不可接受 |
| `detect_outputs()` | 见 §输出 的 any-of 判据 |
| `infer_job_from_dir()` | 从磁盘恢复 job 时：有 `3_Ranked/!_Ranked.csv` → `progress = "<n> accepted"`（数据行数）；否则 → `progress = "finished"`。**不报告 trajectory 尝试数**（旧稿写错） |
| 进度可见性 | v0.0.1 **不做**实时百分比。agent 通过 `/log` 尾部与 `1_Trajectories/!_Trajectories.csv` 行数判断进展 |
| 权重（AF2） | 外置 NAS：`settings.alphafold_params_dir` 下的 7 个 npz（`params_model_{1..5}_multimer_v3.npz` + `params_model_{1,2}_ptm.npz`）。启动时**必须**设 `BINDCRAFT_AF2_PARAMS`，否则上游会去下载 5.3 GB。上游按四种布局查找：`params/params_<model>.npz`、`params_<model>.npz`、`params/<model>.npz`、`<model>.npz`（`upstream/bindcraft/model_weights.py:29-30`） |
| 权重（ProteinMPNN） | 随包：`upstream/bindcraft/weights/proteinmpnn/{neutral,negative,positive}/v_48_020.npz`（每份 6,681,030 B）。**vendor 树里每个变体目录含 4 个 checkpoint**（`v_48_002/010/020/030`），`du -sh` 每变体 26M、整棵 `weights/` 77M，属约束 7 的 <100 MB 例外。**vendor.sh 的排除规则不得剔掉该子树** |
| 编译缓存 | `JAX_COMPILATION_CACHE_DIR=settings.compile_cache_dir`。镜像无 `nvidia-smi`；上游 `cli.use_campaign_compile_cache` 的默认行为是"取 `nvidia-smi` 的卡名 → 在 `<project_folder>/compile_cache/<卡名>` 或 `~/.cache/bindcraft/compile_cache/<卡名>` 下建 per-card 子目录"，取不到卡名就什么都不设。**服务显式设 `JAX_COMPILATION_CACHE_DIR` 会短路这套逻辑**：`bindcraft/__init__.py:12` 把它读进 `OPERATOR_COMPILATION_CACHE`，`use_campaign_compile_cache` 命中后立即返回该值（`upstream/bindcraft/cli.py:181-183`），**不建 per-card 子目录**。实际行为 = 所有卡型共用一个目录；跨卡型能否命中是 JAX 编译缓存自己的 key 语义，不是本服务的分目录策略 |
| GPU 可见性 | Dockerfile 设 `NVIDIA_VISIBLE_DEVICES` / `NVIDIA_DRIVER_CAPABILITIES`，并写 `/etc/ld.so.conf.d/bindcraft-cuda.conf` + `ldconfig`（jax cuda13 wheel 的库在 `site-packages/nvidia/*/lib`，loader 默认找不到） |
| GPU 探针 | `/healthz/detail` 用**子进程**探测 `python -c "import jax; ..."`，结果进程内缓存（TTL 5 min）。**绝不在 HTTP 进程内 import jax**。探针不拼 `-m <module>`（否则 `-c <code>` 会被上游 CLI 当普通参数吞掉） |
| 并发 | `max_concurrent_jobs = 1`：**只约束 submit/poll 路径**（框架 `JobRunner` 的准入检查）。`/api/tasks/*` 走 `execute_task`，在请求线程里同步执行，**没有任何容量检查**——task 路径的并发只能靠 FC `instanceConcurrency: 1` + `sessionConcurrencyPerInstance: 1` 兜底（`deploy/fc.yaml`）。见 §已知限制 |
| worker 覆盖 | `workers_per_gpu = 1` / `max_workers_per_gpu = 1`。BC2 的宿主内存上限读**节点**可用内存而非 job 限额，在 FC 上会过度规划 worker 导致 OOM。副作用见 §输出（worker 日志与显存份额守卫失效） |
| `manifest_extras()` | `tool_outputs`（三阶段 CSV 路径）+ `input_uri_schemes`（**按输入面分层**：`target` 与 `campaign_uri` 各自列出接受的 scheme，避免把 `oss://` / `http(s)://` 误读成 `campaign_uri` 也支持）+ `campaign_knobs`（modality / property 名清单，来自上游 preset 目录）+ `weights`（7 个 npz 的期望文件名 + 四种候选布局） |
| `endpoint_examples()` | 6 个 endpoint 各 ≥1 条可跑 curl，含 `design` 的 shipped-target 路径与上传路径、`rank`/`filter` 接续上一 job 的 `campaign_uri=job://<id>` |
| `/healthz/detail` | 自定义实现（`_strip_route` 摘掉 framework 默认路由），字段见下 |

`/healthz/detail` 返回字段：`status` / `service` / `version` / `weights_dir` /
`weights_loaded` / `weights_missing`（缺失的 AF2 npz 名列表）/ `proteinmpnn_weights_loaded` /
`proteinmpnn_weights_missing` / `expected_alphafold_models` / `gpu_backend`
（`"gpu"` / `"cpu"` / `"probe_failed"`）/ `gpu_devices` / `active_jobs` /
`active_jobs_scope`（恒为 `"submit_poll_only"`）/ `max_concurrent_jobs` / `concurrency_note`，
以及 `gpu_backend != "gpu"` 时的 `warning`。权重缺失时 HTTP 200 + `weights_loaded=false`，
**不在 import 期 raise**。

## 配置

`env_prefix = BINDCRAFT2_`（全走 pydantic-settings，无 `os.getenv`）

| 字段 | 默认 | 说明 |
|---|---|---|
| `jobs_base_dir` | `/data/bindcraft2_jobs` | 任务目录根 |
| `disk_limit_mb` | `1048576`（1 TiB） | 覆盖框架默认 8000。框架语义：`jobs_base_dir` 总占用超阈值时，`evict_finished_until_under_limit` 逐个删掉**整个** completed/failed job 目录。8000 MB 会在下一请求里删掉 `rank`/`filter` 要 chain 的源 campaign（单场 500-trajectory campaign 的 `output/` 轻易超过 8 GB），`/api/tasks/rank` 甚至可能先删掉它正要读的目录再 404 |
| `root` | `/opt/bindcraft` | 子进程 cwd；`Popen` 要求它是已存在目录。上游的 `settings/` / preset / scaffold 定位与它无关（绝对路径） |
| `python` | `/opt/venv/bin/python` | 解释器（离线测试指向 stub） |
| `module` | `bindcraft.cli` | 上游 CLI 模块；置空则只执行 `python <args>`（离线 stub 用） |
| `alphafold_params_dir` | `/data/models/bindcraft2/alphafold` | 含 7 个 npz；见 §P1 |
| `shipped_weights_dir` | `/opt/bindcraft` | 上游源码树，ProteinMPNN 权重探针的根 |
| `compile_cache_dir` | `/data/models/bindcraft2/xla_cache` | `JAX_COMPILATION_CACHE_DIR` |
| `max_concurrent_jobs` | `1` | **submit/poll 路径**的并发上限，不覆盖 `/api/tasks/*` |
| `workers_per_gpu` | `1` | 覆盖 BC2 自动 pack |
| `max_workers_per_gpu` | `1` | 同上 |
| `default_max_trajectories` | `500` | `max_trajectories` 未给时的兜底 |
| `gpu_probe_ttl_seconds` | `300` | `/healthz/detail` GPU 探针缓存秒数，0 = 每次探测 |

## 部署目标

- FC **GPU** 实例（custom-container），`timeout: 36000`（对齐
  [alphafold-server](../../services/alphafold-server/) / [boltzgen-server](../../services/boltzgen-server/)）。
- 控制台：开启**异步任务模式**、清空 keepalive URL、`sessionAffinity` 用 header
  `bioagent-session-id` 且 `sessionConcurrencyPerInstance: 1`、`instanceConcurrency: 1`
  （后者是 task 路径"一实例一卡"的唯一保证，见 §已知限制）。
- 挂载：NAS `mountDir: /data`；OSS `mountPoints` → `/mnt/oss`（有文件输入，经网关调用时
  需要 `oss://` 输入改写为 `/mnt/oss/...` 直读）。
- 主机规格：`deploy/fc.yaml` 用 8 vCPU / 32 GB（`cpu: 8`、`memorySize: 32768`）。
- 镜像：Ubuntu 24.04 多阶段，uv venv + `pip install -e "/opt/bindcraft[cuda13]"` + framework。
  jax cuda13 + cuDNN + cuequivariance 合计数 GB，需要复用 boltzgen 的瘦身路线
  （删 `*.a`、`share/doc|man|locale`、`__pycache__`、`*.pyc`）。**不烘焙** 5.3 GB AF2 参数。
- [services.yaml](../../services.yaml) 条目当前**整条注释**（未部署，`url` 仍是
  `https://fc-bindcraft2-XXXXXX...` 这样的假值；把它原样取消注释会让 `fc_url()` 返回不可达
  主机而不是清晰报错）。首次部署后取消注释并回填真实值。
- 卡型（**未核实**，见 §P2）：`cuda13` extra 要求 compute capability ≥ 7.5，T4(7.5) / A10(8.6) /
  L20(8.9) / A100(8.0) 均可；V100(7.0) 及更老必须改用 `cuda12` extra。

### 显存预算（N 的定义见下，务必先读）

**`N` 是 padding 后的复合物残基数，不是 `binder_lengths`。** 上游按 bucket 32 对**最长的**
binder 长度先向上取整、乘 `copies`，加上最长 target 链长度后再向上取整一次：

```
N = padded32( padded32(max(binder_lengths)) × copies + max(target_length) )
padded32(x) = ceil(x / 32) × 32          # upstream/bindcraft/af2.py:52
```

（`upstream/bindcraft/protein_preparation.py:155-166` 的 `design_residue_count`；bucket 常量
`DEFAULT_LENGTH_BUCKET = 32`，`af2.py:47`。）

单 worker 显存 `est(N) = 2.0 × (3.4 GB + 38000·N²/1e9)` GB
（`upstream/bindcraft/design_workers.py:13-14,54`：`DESIGN_MEMORY_SAFETY_FACTOR=2.0`、
`DESIGN_MODEL_RESIDENT_GB=3.4`、`DESIGN_ACTIVATION_BYTES_PER_RESIDUE_PAIR=38000`），
卡上另留 **4 GB headroom**（`GPU_MEMORY_HEADROOM_GB=4.0`）。`24576 MB` 卡 = 24 GB，
可用预算 = 24 − 4 = **20 GB**。

以下表格按上述公式在 `hPDL1`（115 残基、单链 A；实测
`grep -c "^ATOM.* CA " upstream/settings/target/structures/hPDL1.pdb` = 115）上逐场景重算：

| 场景 | N | est(N) GB | fits 20 GB |
|---|---|---|---|
| `[80,80]`（quickstart） | 96 + 115 = 211 → 224 | 10.61 | yes |
| `[60,60]`（FC smoke） | 64 + 115 = 179 → 192 | 9.60 | yes |
| 默认 `binder` preset `[60,180]` | 192 + 115 = 307 → 320 | 14.58 | yes |
| `homo_oligomer [40,120]` copies=2 | 128×2 + 115 = 371 → 384 | 18.01 | marginal |
| `multidomain [120,300]` | 320 + 115 = 435 → 448 | 22.05 | **no** |
| `large_binder [250,600]`（该 modality 自身默认） | 608 + 115 = 723 → 736 | 47.97 | **no** |

算术细节（`est = 2.0 × (3.4 + 38000·N²/1e9)`）：

```
N=224  N²=  50176  38000N²/1e9=1.906688  +3.4=5.306688  ×2=10.613376 → 10.61
N=192  N²=  36864  38000N²/1e9=1.400832  +3.4=4.800832  ×2= 9.601664 →  9.60
N=320  N²= 102400  38000N²/1e9=3.891200  +3.4=7.291200  ×2=14.582400 → 14.58
N=384  N²= 147456  38000N²/1e9=5.603328  +3.4=9.003328  ×2=18.006656 → 18.01
N=448  N²= 200704  38000N²/1e9=7.626752  +3.4=11.026752 ×2=22.053504 → 22.05
N=736  N²= 541696  38000N²/1e9=20.584448 +3.4=23.984448 ×2=47.968896 → 47.97
```

**隐含上限**：`est(N) ≤ 20` ⇒ `3.4 + 0.000038·N² ≤ 10` ⇒ `N ≤ 416.7`，即 **N ≤ 416**。
对 115 残基的 target，`N` 只由 `binder_lengths` 的最大值决定，且 padding 把
289–320 都归到同一个 448 桶——所以能装下的最大 binder 长度是 **288**（`288+115=403 → 416`，
`est=19.95 GB`）；`binder_lengths=300` 落到 448 桶 → 22.05 GB，**超预算**。
旧稿写的"≲300"应理解为"约 300 但实际断点在 288"。

⚠️ **API 目前对 `binder_lengths` 没有上界**（`models.py` 只校验长度 1–2、每项 ≥1、升序）。
也就是说 `multidomain` 的默认 `[120,300]`、`large_binder` 的默认 `[250,600]` 都能提交成功，
然后在几小时后以 CUDA OOM 失败，而不是在请求时被拒。这是 §已知限制里待决的"`binder_lengths`
上界"问题的来源。

卡型建议：**L20 48 GB 或 A10 24 GB**。**T4 16 GB 不是可用最小配置**——16 GB 扣掉 4 GB
headroom 只剩 12 GB，连默认 `binder` preset `[60,180]`（14.58 GB）都装不下；只有
`[80,80]`（10.61）与 `[60,60]`（9.60）这类小 campaign 能跑。旧稿的"T4 16 GB 为最小可用配置"
是错的。

## 测试策略

| 层 | 文件 | 内容 |
|---|---|---|
| offline HTTP | `tests/test_app.py` | `BINDCRAFT2_PYTHON` 指向 stub：`/healthz`、`/api/manifest`（endpoints + examples 齐全）、三端点各一次 submit、以及 422 / 404 矩阵（target 三选一违规、`target_name` 与上传专用字段同给、`binder_lengths` 长度/升序/非整数违规、HTTP 复杂字段逗号写法 json_invalid、未知 `campaign_uri` 404） |
| offline CLI | `tests/test_cli.py` | endpoint 注册表完整；`build_campaign_json` 对 shipped / 上传两条路径的逐字段断言；rank / filter argv builder；`create_cli` 的 design / rank 两条 e2e |
| FC sync | `tests/test_fc.py` | `@pytest.mark.fc`：`/healthz/detail`（断言 `weights_loaded=true` 且 `gpu_backend=="gpu"`）、`/api/manifest`、`design` 小 campaign（**shipped target `hPDL1`** + `number_of_final_designs=1`、`max_trajectories=1`、`binder_lengths=[60,60]`）→ 断言 `summary.csv` / `campaign_metadata.json` / `1_Trajectories/!_Trajectories.csv` 存在；再用产物目录跑 `rank` / `filter` |
| FC async | `tests/test_fc_task.py` | `@pytest.mark.fc`：`/api/tasks/{design,rank,filter}` 的端点注册与**同步终态（atomic）语义**（`test_task_*_is_atomic`），以及各端点的产物存在 |

fixture：FC smoke 用 **shipped target `hPDL1`**（保证可解析）——`tests/data/mini_target.pdb`
**不存在**（`ls tests/data/` 只有 `fake_bindcraft.sh`），旧稿把它列为 fixture 是错的。离线上传
路径用测试内联的字节（如 `("target.cif", b"data_demo\n", ...)`）；
`tests/data/fake_bindcraft.sh` 供离线 HTTP / CLI 成功路径使用。

FC smoke 即使 1 个 trajectory 也是**分钟级**（计划 Task 12 的预期）——建议先跑
`pytest -m fc -k "not design"` 验证 health / manifest，再跑完整 design；`rank` / `filter`
在 design 未跑时会自行 skip（`STATE` 为空）。

### 测试覆盖的诚实说明

离线套件是**成功路径**的回归网，不能当作失败路径或真实 GPU 行为的证据。具体缺口：

- **从不跑 `rc != 0` 的 job**：stub 的每个分支都 `exit 0`，也没有"失败 stub"。因此失败终态、
  `error_summary` / `error_tail` 的填充（框架 `finalize_job`）**完全未被离线验证**，只有
  FC 集成测试能暴露。
- **CLI 的 `filter` endpoint 体从未被执行**：`test_cli.py` 只用 `create_cli` 跑了 `design` 与
  `rank`；`__main__._filter_build` 没有任何测试覆盖。
- **`build_campaign_json` 的 `core` 分支从未被置值**：测试只断言过 `"core" not in campaign`，
  没有 `core="..."` 的用例。
- **GPU 探针的代码串本身未被校验**：stub 的 `-c` 分支不执行传入的代码，只回显 `gpu`。
  测试能证明"探针走子进程"和"不拼 `-m module`"，但不能证明探针代码在真机上能 import jax
  并打印出后端——那要在 FC 上跑 `test_fc.py::test_healthz_detail_reports_weights_and_gpu`。

含义：本服务的离线绿灯只保证成功路径的请求编译 / argv / 产物判定；**失败语义、CLI filter
路径与真实 GPU 探针必须靠 FC 集成测试**来确认。

## 成功标准

1. `services/bindcraft2-server/` 满足 [新增 service cookbook](../adding-a-new-service/index.zh.md)
   的必备文件清单与提交清单（`VERSION` / `__main__.py` / `scripts/vendor.sh` / `README.md` /
   tests 五件套 / Dockerfile 内无 `git clone`、无 `COPY .../weights/`）。
2. `uv run --group dev ruff check .` 通过；离线套件 `132 passed, 12 skipped`（不需要 GPU）。
3. `make build-bindcraft2-server` 成功；`/api/manifest` 的 6 个 endpoint 各有 ≥1 条可执行 curl；
   `/api/tasks/*` 与 `/api/*` 路由一一对应。
4. FC 部署后 `/healthz/detail` 返回 `weights_loaded: true` 且 `gpu_backend: "gpu"`。
   **未执行**（本环境无 FC 部署）。
5. `pytest -m fc services/bindcraft2-server/tests/test_fc.py`：`design` 小 campaign 端到端产出
   `summary.csv` + `campaign_metadata.json`，随后 `rank` / `filter` 在该产物上各自产出目标 CSV。
   **未执行**。
6. `pytest -m fc services/bindcraft2-server/tests/test_fc_task.py` 通过（三个 task endpoint）。
   **未执行**。
7. [services.yaml](../../services.yaml) 有（当前为注释状态的）条目；[THIRD_PARTY_NOTICES](../../THIRD_PARTY_NOTICES)
   有 BC2 行；服务 README 含 §2 的两条红线。

## 部署前置核验（核验结果 + 处置）

| # | 核验项 | 结果 | 处置 |
|---|---|---|---|
| P1 | NAS `/data/models/alphafold` 是否已由 [alphafold-server](../../services/alphafold-server/) 就位，且含 BC2 需要的 7 个 npz | **已答（按文件名/字节身份论证，未在本环境实测 NAS）**：BC2 的 7 个检查点文件名与 alphafold-server 的完全一致——`params_model_{1..5}_multimer_v3.npz`（各 373,043,148 B）与 `params_model_{1,2}_ptm.npz`（各 373,103,340 B）；BC2 的第一候选路径 `<dir>/params/params_<model>.npz`（`upstream/bindcraft/model_weights.py:30`）与 alphafold 的 `params/` 布局同构。**注意：这些字节数来自 Task 13 交办说明，本环境无 NAS/权重可复核** | `ln -s /data/models/alphafold /data/models/bindcraft2/alphafold`，或设 `BINDCRAFT2_ALPHAFOLD_PARAMS_DIR=/data/models/alphafold`，都避免重新下载整个归档（上游 `ALPHAFOLD_PARAMETER_GIGABYTES=5.3`；`scripts/fetch_weights.sh` 注释实测归档 5,587,968,000 B ≈ 5.59 GB，其中本服务只用 7 个、合计约 2.6 GB）。BC2 **只读**该目录（`model_weights.alphafold_parameters` 在 `BINDCRAFT_AF2_PARAMS` 已设时直接返回、不写盘；编译缓存在另一个目录），共享不会污染 alphafold 的 params。无 NAS 时才跑 `scripts/fetch_weights.sh` |
| P2 | FC 可分配 GPU 卡型 / compute capability | **未确认**。真实卡型从未建立：`deploy/fc.yaml` 保留 `gpuType: fc.gpu.ada.1` + `gpuMemorySize: 24576`，文件自身注释写明该值**未对照控制台合法档位核验**；[promera-server](../../services/promera-server/deploy/fc.yaml) 用同一 `gpuType` 但请求 49152 MB | **悬留项（owner：部署执行者）**：部署前对照 FC 控制台核验卡型与 compute capability；cc ≥ 7.5 → `cuda13`（当前 Dockerfile），cc < 7.5 → 改 `cuda12`。在核验前不应把 P2 当作已决 |
| P3 | `rank` / `filter` 是否只写 `--output` 指向的文件 | **未实测；且"零拷贝"本身有条件**。上游 `rank.design_rows()` 在**请求的源表没有已记录行、但源目录里留了结构文件**时走 `score_structure_folder(campaign, ...)`（`upstream/bindcraft/rank.py:276-279`），后者把 `scored.csv` 写进**源 campaign 目录**（`upstream/bindcraft/score.py:89-94`，`scored_path = os.path.join(folder, SCORED_FILENAME)`）；`--rescore` 也会走这条路。相反，若源表已有记录行、只是某个请求指标未记录，`rerank_campaign` 走 `recompute_structure_metrics`（`rank.py:205`，逐行 `score_design` 更新内存），**不写源目录** | **条件成立时**（源表已有行 + 指标已记录/可派生）源目录只读，可保持零拷贝；**条件不成立时**（源表无行但有结构 / `--rescore`）上游会写 `scored.csv`。**Task 12 Step 6 的 smoke 用 `on=["i_pTM"]` 打 `table=accepted`——一个已记录的指标，因此即使通过也是假阴性**，不能据此宣布零拷贝安全。**该验证配方（Task 12 Step 6，curl 驱动：拍源目录文件集与 mtime 快照 → curl 跑 rank/filter → 比对）至今未运行**。若实测到源目录被改，`campaigns.resolve_campaign_dir` 需要加 `copy_to=<job_dir>/input/campaign` 分支 |

## 已知限制

分两类：(i) 本服务主动选择的 scope；(ii) 框架级行为，本服务只是暴露它。

**(i) 本服务的 scope 选择**

- **无 CPU 模式 / 卡型绑定**：见 §部署目标。`cuda13` extra 与卡型绑定，换卡需改 Dockerfile。
- **不支持多 target / 自定义 scaffold / score / resume / 实时进度**：见 §Endpoint 拓扑的
  "v0.0.1 明确不做"。
- **`binder_lengths` 无上界**：见 §显存预算的 ⚠️；超预算请求在数小时后 OOM，而不是请求时 422。
  是否加上界属待决。

**(ii) 框架级行为（本服务只是使用者）**

- **task endpoint 不做容量检查**：`max_concurrent_jobs` 只在框架 `JobRunner.submit`（submit/poll
  路径）里做准入（`framework/src/bioq_service/runner.py:145`）；`/api/tasks/*` 走
  `execute_task`（`framework/src/bioq_service/task_endpoint.py:76`），在请求线程里同步执行、
  不查任何计数。因此同一个 FC 实例上两个并发 task 会各跑一场 campaign 抢同一张卡。这正是
  `deploy/fc.yaml` 把 `instanceConcurrency` 钉成 `1`（配合 `sessionConcurrencyPerInstance: 1`）
  的原因：FC 层串行化是 task 路径唯一的"一实例一卡"保证。
- **`/healthz/detail` 的 `active_jobs` 只覆盖 submit/poll**：它读 `JobRunner.active_job_count`，
  框架里 task 路径没有活动计数，所以在飞的 `/api/tasks/*` job **无法**从这里观测。
  `active_jobs_scope` / `concurrency_note` 两个字段就是为了避免误读（`app.py:172-182`）。
- **继承的 `disk_limit_mb` 淘汰**：框架默认 8000 MB，语义是超限就删掉**整个**已结束 job 目录
  （`framework/src/bioq_service/jobs.py:294`）。一场 500-trajectory campaign 的 `output/` 轻易
  超过 8 GB，于是下一次 submit/execute_task 会静默删掉 `rank`/`filter` 承诺可 chain 的源
  campaign——`/api/tasks/rank` 甚至可能先删掉它正要读的目录再 404。本服务把
  `disk_limit_mb` 显式放宽到 1 TiB（`settings.py:34`，`deploy/fc.yaml` 也用
  `BINDCRAFT2_DISK_LIMIT_MB=1048576`）来消除这个副作用；它没有被禁用，只是阈值大到按 NAS
  容量不再触发。
- **`list_files()` 返回无上限的扁平清单**：`framework/src/bioq_service/downloads.py:34-41` 用
  `rglob("*")` 列出目录下**所有**文件并整包返回（`GET /api/jobs/<id>/files`）。一场 campaign
  的 `output/` 可能含成千上万个结构文件，响应会很大；agent 应按需读单个文件
  （`GET /api/jobs/<id>/file/<relpath>`）而不是整包拉取。
- **FC `DELETE` 一个正在跑的 job 不会取消它**：`DELETE /api/jobs/{id}` 只做
  `cleanup_job`——从 store 移除记录并 `rmtree` 目录（`framework/src/bioq_service/routes.py:205-211`、
  `jobs.py:277`）。子进程没有任何终止/取消注册表（`runner.py` 里没有 kill/terminate 路径），
  所以正在跑的 campaign 会继续占用 GPU 直到自己结束，只是产物被删了。**v0.0.1 无 wall-clock
  强杀**（见"v0.0.1 明确不做"）。
- **`attach_mcp` 在 mcp 2.x 下退化成一条 warning**：`framework/src/bioq_service/app.py:222-228`
  只捕获 `bioq_service.mcp_server` 的 `ImportError`；`mcp_server.py:47-53` 从
  `mcp.server.fastmcp` 导入 `FastMCP`，而 mcp 2.x 把它改名为 `MCPServer`（本服务实际环境装的
  是 mcp 2.2.0，`from mcp.server.fastmcp import FastMCP` 抛 `ModuleNotFoundError`）。结果是
  `/mcp` 不挂载、服务照常提供 HTTP API。要恢复 MCP 需要 framework 侧适配 mcp 2.x 或把依赖钉回
  `mcp<2`——不在本服务范围内。

## 风险 / 限制

- **运行时长天花板**：FC `timeout: 36000`（10 h）是硬上限。`max_trajectories` 是唯一主动
  兜底；超时后新实例启动、从磁盘恢复时任务变成 FAILED（`manifest.JobLifecycle.restart_semantics`），
  `1_Trajectories/`、`2_Refolded/` 的中间结果仍在 NAS 上（受 §已知限制的 disk 淘汰阈值约束），
  可人工挽救。v0.0.1 **无自动 resume**——框架每次 submit 建新 job_dir，BC2 原生的
  `.campaign_state.json` 续跑机制跨 job 失效。v0.0.2 候选方案：新增 `resume_from_uri`，
  把源 campaign 的 `output/` 作为 `project_folder` 复用。
- **上游 1.x 演进快**：`settings/core/reference.json` 的 key 会增删；服务只映射稳定子集，
  未知 key 无透传通道（不提供 `--set` 直通，避免绕过类型契约）→ 需要时发版跟进。
- **`campaign_uri` 零拷贝的副作用**：真正会写源目录的是 `score_structure_folder` 路径
  （见 §P3），不是"可能写 cache"这类泛泛担心。是否退化为"只读副本 + copy"取决于 P3 实测。
- **显存**：见 §显存预算。T4 16 GB 跑不动默认 preset；大 target / 大 binder 需要大卡或
  由服务侧在提交时拒绝。
- **冷启动**：JAX runtime + 模型编译。`JAX_COMPILATION_CACHE_DIR` 指向 NAS 可缓解；服务把它
  设成一个**共享**目录（不分卡型，见 §关键实现点），命中与否由 JAX 的缓存 key 决定。
  scale-to-zero 后首个请求仍要付 runtime 启动成本。
- **许可证**：见 §2。红线必须随 README 一起交付；对外暴露本服务前需先取得 UZH 商业许可。
- **命名**：服务与镜像名不得暗示与 UZH 的官方关联；README 只能做"基于 / 兼容 BindCraft2"
  的准确陈述。

## Sources

- 上游：[github.com/PacesaLab/BindCraft2](https://github.com/PacesaLab/BindCraft2) — pin
  **v1.0.1** = `5342aefa18dedad653f7a5f6dbee1e566ca24d8f`（default branch `main`）
- 许可证：*BindCraft2 Source-Available License (Hosting-Restricted)*,
  Copyright (c) 2026 Martin Pacesa, University of Zurich
- 上游文档：`README.md`（modality / property / input tiers）、`docs/installation.md`
  （权重、环境变量、容器、Slurm、显存公式）、`docs/outputs.md`（输出文件、度量、rank/filter）
- 上游代码：`bindcraft/model_weights.py`（`CAMPAIGN_MODELS` / 参数查找路径）、
  `bindcraft/protein_preparation.py::design_residue_count`（N 的定义）、
  `bindcraft/design_workers.py`（显存预算与 worker 计划）、`bindcraft/rank.py` /
  `bindcraft/score.py`（P3 的写路径）、`bindcraft/cli.py`（编译缓存）、
  `pyproject.toml`（`cuda12` / `cuda13` extras）
- 框架代码：`framework/src/bioq_service/{forms,cli,runner,task_endpoint,jobs,downloads,routes,manifest,app}.py`
- 参考实现：[boltzgen-server](../../services/boltzgen-server/)（权重外置 + 镜像瘦身）、
  [alphafold-server](../../services/alphafold-server/)（JAX/AF2 + NAS 参数 + 长 GPU 任务 +
  keepalive）、[seqkit-server](../../services/seqkit-server/)（本设计文档的结构样板）
