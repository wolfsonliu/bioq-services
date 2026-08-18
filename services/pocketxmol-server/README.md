# pocketxmol-server

基于 FastAPI 的 [PocketXMol](https://github.com/pengxingang/PocketXMol) HTTP + CLI
双模服务——把 pocket-interacting foundation model 的**结构预测 / 分子设计 / 多肽设计**
能力包装成 FC GPU 服务。**构建在 [bioq-service-framework](../_framework/) 之上**。

**PocketXMol** (Peng et al., *Cell* 2026, doi:10.1016/j.cell.2026.01.003, MIT)
用**单个 ckpt** 统一建模原子级相互作用，覆盖多种 pocket-based 任务：小分子/多肽
对接、结构指导 de novo 分子设计（SBDD）、fragment linking / growing / PROTAC、
分子优化、de novo 线性/环肽设计、peptide inverse folding、side-chain packing。

镜像 base：`nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`；conda env
(Python 3.10 + rdkit + openbabel + biopython + lmdb) + pip torch 2.7 (cu128)
+ PyG 2.x wheels + pytorch-lightning。

## 架构

```
客户端 / Agent
  ↓ HTTP (multipart: protein.pdb + optional ligand/peptide + form fields)
┌────────────────────────────────────────────────────────────────┐
│  FastAPI + bioq-service-framework  (port 9000)             │
│                                                                │
│  服务专属（每个 endpoint 配套 /api/tasks/<name> 异步版）       │
│    POST /api/dock        (small mol / peptide docking)         │
│    POST /api/sbdd        (de novo SBDD)                        │
│    POST /api/linking     (linking / growing / PROTAC)          │
│    POST /api/optimize    (molecular optimization)              │
│    POST /api/pepdesign   (linear/cyclic/inv-fold/sc-pack)      │
│    POST /api/confidence  (tuned-ranker post-scoring)           │
│                                                                │
│  框架统一提供                                                  │
│    GET  /api/manifest                                          │
│    GET  /openapi.json                                          │
│    GET  /healthz, /healthz/detail   (含 3 个 ckpt 就位探针)    │
│    GET  /api/jobs/{id}/...                                     │
└────────────────────────────────────────────────────────────────┘
  ↓ subprocess (cwd=/opt/pocketxmol, 单卡串行)
server/configs.py → build_<task>_config → task_config.yml (5-block YAML)
  ↓
scripts/sample_use.py --config_task <yml> --config_model <yml> --outdir <dir>
scripts/believe_use_pdb.py --exp_name … --config confidence/<variant>.yml  (confidence)
  ↓ NAS
<jobs_base_dir>/<job_id>/{input/task_config.yml, output/<exp>_<ts>/…, logs/, job.json}
```

## 6 个 endpoint 速览

| Endpoint | 何时用 | 关键必填字段 |
|---|---|---|
| **`POST /api/dock`** | 已知配体（SDF/SMILES/PDB/pep_sequence）+ 蛋白口袋，求 N 个对接姿势 | `protein` + one of (`ligand` / `smiles` / `pep_sequence`) |
| **`POST /api/sbdd`** | 蛋白口袋 + `pocket_coord`，无配体先验，生成 N 个候选小分子 | `protein` + `pocket_coord` |
| **`POST /api/linking`** | 蛋白 + 输入分子（含 fragment 原子索引），生成中间连接/生长部分（PROTAC/growing/linking） | `protein` + `input_ligand` + `fragments` (JSON list-of-lists) |
| **`POST /api/optimize`** | 蛋白 + 输入分子，做 `init_step<1` 局部优化生成变体 | `protein` + `input_ligand` |
| **`POST /api/pepdesign`** | 多肽设计：`mode=denovo_linear/denovo_cyclic/inverse_fold/sc_pack` | `protein` + (`pep_length` \| `input_peptide`) |
| **`POST /api/confidence`** | 用 tuned-ranker 给已生成 job 的候选打分（post-processing） | `source_job_id` |

每个 endpoint 都有对应的 `/api/tasks/<name>` 异步任务版（FC async task mode 专用，
`X-Fc-Invocation-Type: Async` header 触发）。

## `POST /api/dock` — 分子对接

multipart/form-data：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `protein` / `protein_uri` | file .pdb / URI | ✓ | — | 蛋白 PDB |
| `ligand` / `ligand_uri` | file .sdf/.pdb / URI | ✱ | — | 输入配体（小分子 SDF 或多肽 PDB） |
| `smiles` | str | ✱ | — | SMILES 输入（小分子对接的替代） |
| `pep_sequence` | str (3-30 aa) | ✱ | — | 多肽序列（等价于 `pepseq_<seq>`；`is_pep=true`） |
| `ref_ligand` / `ref_ligand_uri` | file .sdf / URI | — | — | 口袋提取参考配体 |
| `num_samples` | int (1-200) | — | `10` | 生成对接姿势数 |
| `batch_size` | int (1-200) | — | `50` | GPU batch |
| `is_pep` | bool | — | `false` | 是否多肽（决定输出 `.pdb` vs `.sdf`） |
| `noise_mode` | `gaussian`\|`flexible` | — | `gaussian` | Gaussian 简单快；flexible 分 trans+rot+torsion 保内部几何 |
| `pocket_coord` | JSON `[x,y,z]` | — | null | 显式口袋中心；与 `ref_ligand` 二选一 |
| `pocket_radius` | float (5-25 Å) | — | `10.0` | 口袋提取半径 |
| `pocket_criterion` | `center_of_mass`\|`min` | — | `center_of_mass` | 口袋残基筛选 |
| `seed` | int | — | null | 随机种子 |

`✱` = 3 者恰一（file / smiles / pep_sequence）。

```bash
curl -X POST $URL/api/dock \
    -F protein=@8C7Y_TXV_protein.pdb \
    -F ligand=@8C7Y_TXV_ligand_start_conf.sdf \
    -F num_samples=10 \
    -F 'pocket_coord=[-8.257, 85.181, 19.050]' \
    -F pocket_radius=15
```

## `POST /api/sbdd` — de novo SBDD

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `protein` / `protein_uri` | file / URI | ✓ | — | 蛋白 PDB |
| `pocket_coord` | JSON `[x,y,z]` | ✓ | — | **必填** — de novo 无配体先验 |
| `mode` | `ar`\|`simple` | — | `ar` | AR 迭代精化质量高；simple 一步生成快 |
| `num_samples` / `batch_size` | int | — | 50 / 50 | — |
| `mol_size_mean` / `mol_size_std` | int | — | 28 / 2 | 目标原子数分布 |
| `pocket_radius` | float | — | 15.0 | — |

```bash
curl -X POST $URL/api/sbdd \
    -F protein=@2ar9_A.pdb \
    -F 'pocket_coord=[-8.16, 36.70, 38.77]' \
    -F pocket_radius=15 -F num_samples=50 -F mode=ar
```

## `POST /api/linking` — fragment linking / growing / PROTAC

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `protein` / `protein_uri` | file / URI | ✓ | — | 蛋白 PDB |
| `input_ligand` / `input_ligand_uri` | file .sdf / URI | ✓ | — | 含 fragment 的 SDF |
| `fragments` | JSON list-of-lists of int | ✓ | — | 每组是一个 fragment 的 0-based 原子索引；1 组=growing，2+ 组=linking / PROTAC |
| `part1_pert` | `fixed`\|`free`\|`small` | — | `fixed` | 位置约束：固定/自由/小扰动 |
| `mol_size_mean` / `mol_size_std` | int | — | 40 / 3 | 目标分子原子数（≥ fragment 原子和 + linker 尺寸） |
| `use_input_center` | bool | — | `true` | true=用输入分子中心；false=用口袋中心 |
| `pocket_radius` | float | — | 10.0 | — |
| `num_samples` / `batch_size` | int | — | 50 / 50 | — |

```bash
# Fragment growing (1 fragment group)
curl -X POST $URL/api/linking \
    -F protein=@2ar9_A.pdb -F input_ligand=@fragment.sdf \
    -F 'fragments=[[0,1,2,3,4,5,6]]' \
    -F mol_size_mean=28 -F num_samples=10

# PROTAC linker (warhead atoms in group 1, E3 ligand in group 2)
curl -X POST $URL/api/linking \
    -F protein=@target.pdb -F input_ligand=@warhead_e3.sdf \
    -F 'fragments=[[0,1,2,3,4,5,6], [23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]]' \
    -F mol_size_mean=60 -F part1_pert=fixed
```

## `POST /api/optimize` — 分子优化

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `protein` / `input_ligand` | file / URI | ✓ | — | 同 linking |
| `init_step` | float (0.05-0.99) | — | `0.5` | 越小越接近输入；越大越探索 |
| `num_steps` | int (10-200) | — | `50` | 去噪步数 |
| `mol_size_mean` / `mol_size_std` | int | — | 38 / 3 | — |

```bash
curl -X POST $URL/api/optimize \
    -F protein=@target.pdb -F input_ligand=@parent.sdf \
    -F init_step=0.3 -F num_samples=20
```

## `POST /api/pepdesign` — 多肽设计

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `protein` / `protein_uri` | file / URI | ✓ | — | 蛋白 PDB |
| `mode` | enum | — | `denovo_linear` | `denovo_linear`\|`denovo_cyclic`\|`inverse_fold`\|`sc_pack` |
| `pep_length` | int (5-30) | ✱ | null | de novo 必填；`inverse_fold`/`sc_pack` 忽略 |
| `input_peptide` / `input_peptide_uri` | file .pdb / URI | ✱ | — | `inverse_fold`/`sc_pack` 必填 |
| `ref_ligand` / `ref_ligand_uri` | file .sdf / URI | — | — | 口袋提取参考（可选） |
| `pocket_coord` | JSON `[x,y,z]` | — | null | 口袋中心（可选） |
| `pocket_radius` | float (10-30) | — | `20.0` | 多肽口袋要更大 |
| `fix_pos_res_bb`, `fix_pos_res_sc` | JSON list[int] | — | `[]` | 固定坐标的残基索引 |
| `fix_type_res_bb`, `fix_type_res_sc` | JSON list[int] | — | `[]` | 固定类型的残基索引 |
| `num_samples` / `batch_size` | int | — | 10 / 50 | — |

`✱` = 依 `mode` 而定。

```bash
# de novo 10-mer 线性肽
curl -X POST $URL/api/pepdesign \
    -F protein=@3bik_A.pdb \
    -F mode=denovo_linear -F pep_length=10 \
    -F ref_ligand=@3bik_A_pocket_coord.sdf \
    -F pocket_radius=20 -F num_samples=10

# Inverse folding：已知骨架，设计序列
curl -X POST $URL/api/pepdesign \
    -F protein=@target.pdb \
    -F input_peptide=@backbone.pdb \
    -F mode=inverse_fold -F num_samples=10
```

## `POST /api/confidence` — 置信度打分（后处理）

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `source_job_id` | str | ✓ | — | 上游生成 job 的 job_id |
| `variant` | `tuned_cfd`\|`flex_cfd` | — | `tuned_cfd` | tuned=通用；flex=柔性对接噪声评估 |
| `batch_size` | int | — | `50` | — |

链式调用：

```bash
# 1. 先跑一次 sbdd
JOB_ID=$(curl -X POST $URL/api/sbdd \
    -F protein=@target.pdb \
    -F 'pocket_coord=[0,0,0]' -F num_samples=50 \
    | jq -r .job_id)

# 2. 完成后打分
curl -X POST $URL/api/confidence \
    -F source_job_id=$JOB_ID -F variant=tuned_cfd
```

⚠️ `source_job_id` 引用的 job 必须仍在 NAS 上（未被回收），且 `output/*_SDF/` 目录存在。

## 输出目录

生成 endpoints (`dock` / `sbdd` / `linking` / `optimize` / `pepdesign`)：

```
<jobs_base_dir>/<job_id>/
├── input/
│   ├── protein.pdb                   # 保存的上传
│   ├── ligand.sdf                    # optional
│   ├── task_config.yml               # server 生成的 5-block YAML（可 replay）
│   └── model_config.yml              # ckpt 路径覆盖
├── output/
│   └── <exp>_<timestamp>/
│       ├── <exp>_<timestamp>_SDF/
│       │   ├── 0_inputs/{pocket_block.pdb, input_mol.sdf}
│       │   ├── 0.sdf, 1.sdf, …       # 小分子候选
│       │   └── 0.pdb, 0_mol.sdf, …   # 多肽候选（.pdb + .sdf 各一）
│       ├── SDF/                       # 采样轨迹（save_traj_prob>0 时）
│       ├── gen_info.csv              # 候选元信息 + cfd_traj / cfd_pos / cfd_node / cfd_edge 自置信度
│       └── log.txt
├── logs/run.log                       # server 层子进程输出
└── job.json                           # framework 落地
```

Confidence endpoint：

```
<jobs_base_dir>/<confidence_job_id>/
├── input/source_ref.txt              # 记录 source_job_id + variant
├── output/                            # 空目录（believe_use_pdb 写回 source 目录）
├── logs/run.log
└── job.json

# ⚠️ 实际 ranking 落到 source job：
<jobs_base_dir>/<source_job_id>/output/<exp>_<ts>/ranking/*.csv
```

## Weights（NAS 外置）

按 [权重外置化设计](../../engineering/decisions/2026-06-26-service-weights-externalization.md)，
3 个 ckpt 不打进镜像，走 NAS：

```
/data/models/pocketxmol/
├── pxm/
│   ├── checkpoints/pocketxmol.ckpt          # 主 foundation model
│   └── train_config/*.yml
├── tuned_ranker/
│   ├── checkpoints/tuned_ranker.ckpt        # /api/confidence variant=tuned_cfd
│   └── train_config/*.yml
├── flex_cfd/
│   ├── checkpoints/flex_cfd.ckpt            # /api/confidence variant=flex_cfd
│   └── train_config/*.yml
└── ccd/                                      # 可选（sdf2pdb_robust 用，v0.0.1 未 expose）
    ├── aa_fgs_db.pkl
    ├── atom_name_db.pkl
    └── smiles_aa.csv
```

### Pre-stage 命令

```bash
# 本地 stage 目录（inspection 用）：
./services/pocketxmol-server/scripts/fetch_weights.sh
# → services/pocketxmol-server/weights/  (~500 MB)

# 直接下到 NAS mount（FC 部署前必做）：
WEIGHTS_DST=/mnt/nas/data/models/pocketxmol \
    ./services/pocketxmol-server/scripts/fetch_weights.sh
```

### SIF / HPC apptainer

```bash
apptainer run --nv \
    --bind /scratch/models/pocketxmol:/data/models/pocketxmol \
    --bind /scratch/pocketxmol_jobs:/data/pocketxmol_jobs \
    pocketxmol-server.sif
```

### FC 部署前验证权重

部署完成后：

```bash
curl $URL/healthz/detail
# {
#   "weights_loaded": true,
#   "weights_missing": {},
#   ...
# }
```

任一 ckpt 缺失时：

```json
{
  "weights_loaded": false,
  "weights_missing": {
    "pxm_checkpoint": "/data/models/pocketxmol/pxm/checkpoints/pocketxmol.ckpt"
  }
}
```

## 配置 (`POCKETXMOL_*` env)

| Env | 默认 | 说明 |
|---|---|---|
| `POCKETXMOL_JOBS_BASE_DIR` | `/data/pocketxmol_jobs` | Job artifact 落盘目录 |
| `POCKETXMOL_ROOT` | `/opt/pocketxmol` | subprocess cwd（vendored upstream） |
| `POCKETXMOL_PYTHON` | `/opt/conda/envs/pocketxmol/bin/python` | Conda Python |
| `POCKETXMOL_SAMPLE_SCRIPT` | `/opt/pocketxmol/scripts/sample_use.py` | 生成脚本 |
| `POCKETXMOL_CONFIDENCE_SCRIPT` | `/opt/pocketxmol/scripts/believe_use_pdb.py` | 置信度脚本 |
| `POCKETXMOL_WEIGHTS_DIR` | `/data/models/pocketxmol` | NAS 权重根目录 |
| `POCKETXMOL_PXM_CHECKPOINT` | `.../pxm/checkpoints/pocketxmol.ckpt` | 主 ckpt |
| `POCKETXMOL_TUNED_CFD_CKPT` | `.../tuned_ranker/checkpoints/tuned_ranker.ckpt` | 置信度 tuned |
| `POCKETXMOL_FLEX_CFD_CKPT` | `.../flex_cfd/checkpoints/flex_cfd.ckpt` | 置信度 flex |
| `POCKETXMOL_MAX_CONCURRENT_JOBS` | `1` | 单卡串行 |
| `POCKETXMOL_TASK_ENDPOINTS_ENABLED` | `True` | `/api/tasks/*` 开关 |
| `POCKETXMOL_OSS_REGION` | `cn-hangzhou` | OSS URI 下载 region |

## 本地开发

```bash
# 1. Vendor upstream（一次性）
./services/pocketxmol-server/scripts/vendor.sh

# 2. （可选）本地下载权重到 stage 目录
./services/pocketxmol-server/scripts/fetch_weights.sh

# 3. 运行离线测试
uv run python -m pytest services/pocketxmol-server/tests/test_app.py -v
uv run python -m pytest services/pocketxmol-server/tests/test_cli.py -v

# 4. Lint
uvx ruff check services/pocketxmol-server/
```

## Docker 构建

```bash
# 项目根目录执行
./services/pocketxmol-server/scripts/vendor.sh
docker build --platform linux/amd64 -t pocketxmol-server \
    -f services/pocketxmol-server/Dockerfile .
```

## CLI 批处理模式（Slurm sbatch）

同一镜像可切换到 CLI 模式跑 sbatch 一次性任务，共 6 subcommand（dock / sbdd /
linking / optimize / pepdesign / confidence）：

```bash
# Docker CLI 模式
docker run --rm --gpus all -v /data:/data pocketxmol-server \
    .venv/bin/python -m server dock \
    --protein /data/8C7Y_TXV_protein.pdb \
    --ligand  /data/8C7Y_TXV_ligand.sdf \
    --output-dir /data/results/ \
    --params-json '{"num_samples": 10, "pocket_coord": [-8.257, 85.181, 19.050], "pocket_radius": 15}'

# Apptainer / SIF (sbatch)
apptainer exec --nv \
    --bind /scratch/models/pocketxmol:/data/models/pocketxmol \
    pocketxmol-server.sif \
    .venv/bin/python -m server pepdesign \
    --protein /scratch/target.pdb \
    --ref-ligand /scratch/ref.sdf \
    --output-dir /scratch/$SLURM_JOB_ID/ \
    --params-json '{"mode": "denovo_cyclic", "pep_length": 12, "pocket_radius": 20, "num_samples": 20}'
```

参见 [engineering/decisions/2026-05-29-cli-batch-mode.md](../../engineering/decisions/2026-05-29-cli-batch-mode.md)。

## FC 部署 checklist

1. `./services/pocketxmol-server/scripts/vendor.sh` — vendor upstream
2. `WEIGHTS_DST=/mnt/nas/data/models/pocketxmol ./services/pocketxmol-server/scripts/fetch_weights.sh` — 权重上 NAS
3. `docker build -t pocketxmol-server -f services/pocketxmol-server/Dockerfile .`
4. Push 到阿里云 ACR
5. 创建 FC 函数：
   - GPU 实例（T4 8GB 起步；`fc.gpu.tesla.1`）
   - 内存 16 GB / CPU 4 vCPU / 超时 1200s
   - 挂载 NAS：`/data → NAS`（job artifact 目录 + 权重共用）
   - 环境变量：`POCKETXMOL_PXM_CHECKPOINT`、`POCKETXMOL_TUNED_CFD_CKPT`、`POCKETXMOL_FLEX_CFD_CKPT` 三者已在 Dockerfile 里默认指到 NAS
   - **打开「异步任务模式」** — `/api/tasks/*` 依赖
   - session affinity: header `bioagent-session-id`
6. 部署完成后：
   ```bash
   curl $URL/healthz/detail  # → weights_loaded: true
   ```
7. 写入 `services/aliyun_fc_url.md` 一行：`pocketxmol-server: <URL>`
8. 跑 FC 集成测试：
   ```bash
   RUN_FC_TESTS=1 uv run python -m pytest -m fc services/pocketxmol-server/tests/test_fc.py -v
   RUN_FC_TESTS=1 uv run python -m pytest -m fc services/pocketxmol-server/tests/test_fc_task.py -v
   ```

## 相关设计与文档

- [pocketxmol-server 设计](../../engineering/decisions/2026-07-06-pocketxmol-server-design.md) — 完整设计文档
- [新增服务 cookbook](../../docs/adding-a-new-service/index.zh.md) — 骨架 + 规范
- [Service 权重 NAS 外置化设计](../../engineering/decisions/2026-06-26-service-weights-externalization.md)
- [FC 异步任务模式](../../engineering/decisions/2026-06-17-fc-async-task-mode.md)

## 引用

```bibtex
@article{Peng2026,
  title = {Unified modeling of 3D molecular generation via atomic interactions with PocketXMol},
  author = {Peng, Xingang and Guo, Ruihan and Guo, Fenglin and Wang, Ziyi and Sun, Jiayu and Guan, Jiaqi and Jia, Yinjun and Xu, Yan and Huang, Yanwen and Zhang, Muhan and Peng, Jian and Wang, Xinquan and Han, Chuanhui and Wang, Zihua and Ma, Jianzhu},
  journal = {Cell},
  year = {2026},
  doi = {10.1016/j.cell.2026.01.003},
}
```

License: upstream MIT.  See `upstream/LICENSE` after running `vendor.sh`.
