# RFdiffusion2 Server

基于 FastAPI 的 RFdiffusion2 HTTP 服务，构建在 [bioq-service-framework](../_framework/) 之上：
HTTP 层 / job 生命周期 / 错误处理 / 持久化 / 多实例一致性 / Agent 协议描述均由框架统一提供，服务自身只
负责把 RFdiffusion2 的 3 种典型用法（active_site / small_molecule_binder / 自定义）映射为
`rf_diffusion/run_inference.py` 的 Hydra override argv。

上游：[RosettaCommons/RFdiffusion2](https://github.com/RosettaCommons/RFdiffusion2)，
论文：Ahern et al. 2025 _"Atom level enzyme active site scaffolding using RFdiffusion2"_ (bioRxiv)。

> 与 [rfdiffusion-server](../rfdiffusion-server/) 的关系：v1 处理「残基级 motif + 蛋白 PPI binder」，
> v2 处理「原子级 motif（侧链锚定）+ 小分子配体」。设计酶活性中心、小分子 binder 用 v2；纯蛋白
> binder / motif scaffold / 对称 oligomer 用 v1。

## 架构

```
客户端 / Agent
  ↓ HTTP
┌────────────────────────────────────────────────────────────────┐
│  FastAPI + bioq-service-framework  (port 9000)             │
│                                                                │
│  服务专属（rfdiffusion2-server 注册）                          │
│    POST /api/generate/active_site             (酶活性位点)     │
│    POST /api/generate/small_molecule_binder   (小分子 binder)  │
│    POST /api/generate                         (自定义透传)     │
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
RFdiffusion2: rf_diffusion/run_inference.py <Hydra overrides...>
  ↓ 输出
NAS: <jobs_base_dir>/<job_id>/{input/, output/design_<N>.{pdb,trb}, logs/run.log, job.json}
```

## API 速览

每个 POST 端点接受 `multipart/form-data`：文件字段 + pydantic 参数。返回 `JobInfo`（含 `job_id`），
实际计算在后台异步执行。

| 端点 | 必填输入 | 用途 |
|---|---|---|
| `POST /api/generate/active_site` | PDB + contigs + ligand + contig_atoms | 围绕原子级 motif（指定侧链原子）+ 配体设计酶活性中心 scaffold |
| `POST /api/generate/small_molecule_binder` | PDB + contigs + ligand | 围绕小分子设计蛋白 binder；可选 RASA 包埋度条件 |
| `POST /api/generate` | contigs（+ 可选 PDB）| 自定义：任选 `--config-name`，透传任意 Hydra override |

### 公共参数（_GenerationCommon）

| 字段 | 默认 | 说明 |
|---|---|---|
| `num_designs` | 4 | 每次提交生成多少个 design |
| `diffuser_t` | 100 | flow-matching 步数；`aa.yaml` 默认 100 |
| `final_step` | 1 | 提前终止 trajectory（1 = 跑完）|
| `write_trajectory` | false | 是否保留完整 trajectory（默认关闭以省磁盘）|
| `deterministic` | false | RNG 固定 |
| `model` | (auto) | 显式 checkpoint：`rfd_140` / `rfd_173`；不填走 config 内置 |

### 端点 1: active_site (酶活性中心 scaffolding)

复现 `open_source_demo.json` 的 4 个 `active_site_*` case。

```bash
# active_site_unindexed_atomic (motif 位置由网络自选，contig_as_guidepost=true)
curl -X POST $URL/api/generate/active_site \
  -F input_pdb=@M0584_1ldm.pdb \
  -F 'contigs=46,A106-106,59,A166-166,2,A169-169,23,A193-193,46' \
  -F 'ligand=NAD,OXM' \
  -F 'contig_atoms={"A106":"NE,CD,CZ","A166":"OD1,CG","A169":"NH2,CZ","A193":"NE2,CD2,CE1"}' \
  -F contig_as_guidepost=true \
  -F num_designs=10
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `input_pdb` 或 `input_pdb_uri` | 是 | 含 motif + 配体的 PDB |
| `contigs` | 是 | Hydra contig 串，例：`46,A106-106,59,...`；裸数字 = 待设计的长度，`<chain><resnum>-<resnum>` = motif 锚 |
| `contig_atoms` | 是 | 每个 motif 残基锚定哪些侧链原子；JSON dict：`{"A106":"NE,CD,CZ"}` |
| `ligand` | 是 | 逗号分隔的配体残基名，例 `NAD,OXM` |
| `contig_as_guidepost` | 否 | `true`（默认）= 位置不定（guidepost），`false` = 位置固定（indexed）|
| `only_guidepost_positions` | 否 | guidepost 仅作用于某些位点（其余位点 indexed），例 `A106` |
| `partially_fixed_ligand` | 否 | 配体部分原子固定、其余 diffuse；JSON：`{"NAD":["O7N","C7N"]}` |
| `inpaint_seq` | 否 | 屏蔽序列的位置 |

四种 demo 对应参数：

| Demo case | contig_as_guidepost | only_guidepost_positions | partially_fixed_ligand |
|---|---|---|---|
| `active_site_indexed_atomic` | false | — | — |
| `active_site_unindexed_atomic` | true | — | — |
| `active_site_unindexed_atomic_some_indexed` | true | `A106` | — |
| `active_site_unindexed_atomic_partial_ligand` | true | — | `{NAD:[...], OXM:[...]}` |

### 端点 2: small_molecule_binder (小分子 binder)

复现 `small_molecule_binder_rasa_buried` demo。

```bash
# 围绕 PH2 配体设计 150aa binder，包埋（RASA=0）
curl -X POST $URL/api/generate/small_molecule_binder \
  -F input_pdb=@trimmed_ec2_M0151_NO_ORI_zero_com0.pdb \
  -F contigs=150 -F length=150-150 \
  -F ligand=PH2 \
  -F rasa_active=true -F rasa_target=0.0 \
  -F num_designs=10
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `input_pdb` 或 `input_pdb_uri` | 是 | 含小分子的 PDB |
| `contigs` | 是 | 单段长度 contig，例 `150` |
| `length` | 否 | 长度硬约束，例 `150-150` |
| `ligand` | 是 | 配体残基名（单个）|
| `rasa_active` | 否 | 是否开启 RASA-v2 条件（默认 true）|
| `rasa_target` | 否 | RASA 目标值 [0,1]；0 = 配体完全包埋 |

### 端点 3: 自定义 / 透传

任选 `config_name`（`rf_diffusion/config/inference/` 下的任意 .yaml），并通过 `extra_overrides` 透传
任意 Hydra 路径。适合做 partial diffusion、二级结构条件、PPI、对称等没单独包成 endpoint 的模式。

```bash
curl -X POST $URL/api/generate \
  -F input_pdb=@target.pdb \
  -F config_name=aa_ppi \
  -F input_pdb_required=true \
  -F 'contigs=A1-150/0 70-100' \
  -F num_designs=10 \
  -F 'extra_overrides={"ppi.hotspot_res":"[A59,A83,A91]"}'
```

`extra_overrides` 是 JSON dict，每条 `"key.path": value` 被拼成 `key.path=value` 加到 argv。
bool 会被转成 `true` / `false`。

可用 config 在 `opensource/RFdiffusion2/rf_diffusion/config/inference/` 下，常见的：

| Config | 用途 |
|---|---|
| `aa.yaml` (默认) | 标准 all-atom 推理 |
| `aa_ppi.yaml` | 蛋白-蛋白互作 |
| `aa_small.yaml` | 小蛋白 |
| `aa_tip_atoms_positioned.yaml` | tip-atom motif（位置已知）|
| `aa_tip_atoms_position_agnostic.yaml` | tip-atom motif（位置自由）|
| `unconditional.yaml` | 无条件设计 |
| `sym.yaml` | 对称（试验性）|

## 输入 PDB 来源

每个端点接受 5 种来源，二选一：

| 字段/scheme | 用途 |
|---|---|
| `input_pdb=@<file>` | multipart upload |
| `input_pdb_uri=job://<job_id>/<filename>` | 复用 NAS 上前一个 job 的输出文件（零拷贝）|
| `input_pdb_uri=file:///abs/path` | NAS 上的绝对路径（多个服务共享 mount）|
| `input_pdb_uri=oss://<bucket>/<key>` | 阿里云 OSS 对象 |
| `input_pdb_uri=https://...` | 任意 HTTP(S) URL（含 OSS 预签名）|

## 输出

```
<jobs_base_dir>/<job_id>/
├── input/<endpoint>.pdb       (上传 / 下载的 PDB 副本)
├── output/
│   ├── design_0.pdb           最终 backbone（含 motif 侧链）
│   ├── design_0.trb           pickle: contig / config / 残基映射
│   ├── design_1.pdb
│   ├── ...
│   └── traj/                  仅当 write_trajectory=true
├── logs/
│   └── run.log                完整 stdout/stderr
└── job.json                   JobInfo（status, error 详情）
```

`design_<N>.pdb` 中：motif 残基保留侧链坐标，diffused 残基只有 backbone。下游需配 LigandMPNN
（包含在镜像内的 `rf_diffusion/third_party_model_weights/ligand_mpnn/` 权重）做序列设计。

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `RFDIFFUSION2_ROOT` | `/opt/rfdiffusion2` | 上游源码 + 模型根目录 |
| `RFDIFFUSION2_MODELS_DIR` | `/opt/rfdiffusion2/rf_diffusion/model_weights` | RFD_*.pt 所在 |
| `RFDIFFUSION2_INFERENCE_SCRIPT` | `/opt/rfdiffusion2/rf_diffusion/run_inference.py` | Hydra 入口 |
| `RFDIFFUSION2_PYTHON` | `/opt/conda/envs/rfd2/bin/python` | conda env 内的 python |
| `RFDIFFUSION2_PYTHONPATH` | `/opt/rfdiffusion2` | 注入子进程的 PYTHONPATH |
| `RFDIFFUSION2_JOBS_BASE_DIR` | `/data/rfdiffusion2_jobs` | 任务持久化目录 |
| `RFDIFFUSION2_OSS_REGION` | `cn-hangzhou` | OSS URI 解析区域 |

OSS 凭证从标准环境变量读取（`OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET`）。

## 本地开发

```bash
# 拉源码 + 权重
cd opensource/RFdiffusion2 && python setup.py

# 创建 conda env（参考 envs/cuda124_env.yml）— 至少需要 16 GB GPU 显存
micromamba env create -n rfd2 -f opensource/RFdiffusion2/envs/cuda124_env.yml
micromamba run -n rfd2 pip install \
    git+https://github.com/RalphMao/PyTimer.git \
    git+https://github.com/baker-laboratory/ipd.git

# 启动服务
cd services/rfdiffusion2-server
RFDIFFUSION2_ROOT=$(pwd)/../../opensource/RFdiffusion2 \
RFDIFFUSION2_PYTHON=/path/to/rfd2/bin/python \
RFDIFFUSION2_JOBS_BASE_DIR=/tmp/rfd2_jobs \
micromamba run -n rfd2 uvicorn server.app:app --reload --port 9000
```

测试：

```bash
cd services/rfdiffusion2-server
uv run --with bioq-service-framework --with httpx --with pytest \
  --no-project python -m pytest tests/
```

## Vendor 与权重

RFdiffusion2 上游不是 pip-installable 包，官方安装方式是 conda env + 设 `PYTHONPATH` 指向源码（见 [installation doc](https://rosettacommons.github.io/RFdiffusion2/installation.html)）。本服务为了与 `opensource/` 解耦，把运行时所需源码 vendor 到 `upstream/`，权重放到 `weights/`，两者都 **gitignored**，由 `scripts/` 下的脚本从 `opensource/RFdiffusion2/` 派生：

- `upstream/rf_diffusion/` — Python 源码 + Hydra configs + benchmark/input/
- `upstream/envs/{cuda124_env.yml, requirements_cuda124.txt}` — conda env spec
- `weights/` — `.pt` checkpoint（每个 fresh checkout 都要重新填）

### 从 upstream 同步源码

```bash
./services/rfdiffusion2-server/scripts/vendor.sh
```

排除规则在脚本内：`test_data/` / `goldens/` / `dev/` / `exec/` / `*.pkl` / `*.pse` / `model_weights/` 等不进 vendor。Re-run 后 `git status` 看不到改动（gitignored），但 `du -sh upstream/` 可以确认大小。

### 填充权重

```bash
# 1. 先按 upstream 文档下权重到 opensource/
cd opensource/RFdiffusion2 && python setup.py && cd -

# 2. 再 rsync 到本服务
./services/rfdiffusion2-server/scripts/fetch_weights.sh
```

`docker build` 之前必须先跑这两个脚本，否则 Dockerfile 会因为 `upstream/` 或 `weights/` 不存在而 build 失败。

设计背景见 [engineering/decisions/2026-05-19-rfdiffusion2-server-vendor.md](../../engineering/decisions/2026-05-19-rfdiffusion2-server-vendor.md)。

## 构建 Docker 镜像

镜像从项目根目录构建：

```bash
# 1. 先在本机下载权重（~2.6 GB）
cd opensource/RFdiffusion2 && python setup.py

# 2. 构建（首次约 30 分钟：torch + dgl + pyg-lib + pyrosetta + 2.6GB 权重）
make build-rfdiffusion2-server

# 3. 推送到 harbor
make push-rfdiffusion2-server
```

镜像版本来自 `services/rfdiffusion2-server/VERSION`；新版本直接改这个文件即可。

预计大小：~18 GB（CUDA 12.4 devel 6.7 GB + conda env 8 GB + 模型权重 2.6 GB + 框架 + 源码）。
FC 上限是 15 GB **per layer**（总大小可以更大），所以分层要注意；模型权重单独一层即可。

## 阿里云 FC 部署

镜像规格基本同其他 GPU 服务：

| 项 | 值 |
|---|---|
| 实例类型 | GPU（A10 / L4 / L20）|
| 显存 | ≥ 24 GB（150aa 设计实测约 16 GB）|
| 内存 | ≥ 32 GB |
| 磁盘 | NAS 挂载到 `/data/rfdiffusion2_jobs` |
| 端口 | 9000 |
| 健康检查 | `GET /healthz` |
| Keep-alive | ≥ 15 分钟（长任务用）|

冷启动：首次加载 RFD_140.pt（1.3 GB）+ pyrosetta 初始化约 20-30 秒；之后保持热实例。

## 排查

- **`Error: cuda_runtime_api.h: No such file or directory`** — 必须用 `-devel` 基础镜像，且
  `CUDA_HOME=/usr/local/cuda` 已设置（防止 torch 自动选 conda 内 nvcc）。详见
  `engineering/guides/uv-dockerfile-patterns.md`。
- **`ModuleNotFoundError: ipd`** — 镜像内的 `pip install git+...baker-laboratory/ipd.git` 步失败；
  本地构建时检查能否访问 GitHub。
- **`pyrosetta` 找不到** — `conda.rosettacommons.org` 在国内访问较慢；如有学术 license 可换成
  本地下载的 .whl，或换用 PyRosetta-bundle。
- **JIT compile 失败** — RFdiffusion2 调用 cuequivariant / pyg_lib，可能在首次模型构造时触发
  CUDA 编译；确保 `-devel` 基础镜像 + `CUDA_HOME` 设置 + GPU 架构在 sm_70+。

## 与其他服务的关系

```
                       ┌─ rfdiffusion-server (v1) ─→ ProteinMPNN
                       │   contigs in PDB chains
客户端 / Agent  ─→  Router
                       │
                       └─ rfdiffusion2-server (v2) ─→ LigandMPNN
                           contigs with sidechain atoms + ligand
```

下游推荐：

- 序列设计：[proteinmpnn-server](../proteinmpnn-server/)（仅蛋白）或 LigandMPNN（含小分子，目前打包在
  RFdiffusion2 镜像内）。
- 结构验证：本仓库未独立提供 RF2/Chai-1 服务；可调上游 RFdiffusion2 的 pipeline.py 串联（依赖
  `rf_diffusion/exec/chai.sif`，FC 不支持 .sif）。
