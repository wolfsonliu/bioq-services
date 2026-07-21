# bindflow-server

基于 FastAPI 的 [BindFlow](https://github.com/ale94mleon/BindFlow) HTTP + CLI
包装 —— snakemake 编排的 GROMACS ABFE / MMPBSA 全流程。**HPC-primary**：
主入口是 `apptainer exec bindflow-server.sif python -m server {fep|mmpbsa}`；
HTTP 端点保留用于本地 dev / smoke / 未来 K8s 部署，**不部署 FC**（工作负载
经常超出 24 h 上限）。构建在 [bioq-service-framework](../_framework/) 之上。

BindFlow：给定蛋白 + 已知结合姿势的配体，用 Boresch-restrained FEP 或
MM(P/G)BSA 计算严格的结合自由能。**Upstream 是 GPL-3.0**（fork 自 Biggin Lab
的 ABFE_workflow）。

## 与 boltz-server 亲和力头的定位差异

| | **bindflow-server** | boltz-server 亲和力 |
|---|---|---|
| 方法 | MD-based FEP / MMPBSA（Boresch 限制 + λ 窗口） | ML 亲和力 head |
| 时长 | 几小时 – 几天 | 秒 – 分钟 |
| 严格性 | 物理化学一线 | 高吞吐筛选 |
| 适用 | shortlist top-N 精细排序 | 大规模筛选（几千候选） |

## 架构

```
客户端 / Agent
  ↓ HTTP multipart / URI  OR  CLI subcommand
┌───────────────────────────────────────────────────────────────────┐
│  FastAPI + bioq-service-framework  (port 9000, HTTP mode)     │
│    POST /api/calculate/fep                                        │
│    POST /api/calculate/mmpbsa                                     │
│    GET  /api/manifest, /healthz, /healthz/detail, /openapi.json   │
│    GET  /api/jobs/{id}/...                                        │
│                                                                   │
│  CLI batch mode (HPC-primary)                                     │
│    python -m server fep    ...                                    │
│    python -m server mmpbsa ...                                    │
└───────────────────────────────────────────────────────────────────┘
  ↓ subprocess (server/inference.py → bindflow.runners.calculate)
snakemake DAG → pdbfixer + parmed → TOFF → gmx grompp/mdrun × N λ × M replica
  ↓ alchemlyb / pymbar / gmx_MMPBSA
NAS: <jobs_base_dir>/<job_id>/{input/, output/(fep|mmxbsa)_partial_results.csv + per-ligand dirs}
```

## API 速览

### `POST /api/calculate/fep`

Multipart form (recommended for small ligand sets):

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `protein` | UploadFile (.pdb) | ✓（或 `protein_uri`） | — | 蛋白 |
| `ligands` | List[UploadFile] (.sdf/.mol) | ✓（或 `ligands_zip_uri`） | — | ≥1 配体 |
| `cofactor` / `cofactor_uri` | 可选 | ✗ | — | 辅因子 |
| `membrane` / `membrane_uri` | 可选 (.pdb, 需 CRYST1) | ✗ | — | 膜 |
| `custom_ff_zip` / `custom_ff_uri` | 可选 zip | ✗ | — | 自定义 `*.ff` 目录 |
| `topology_zip` / `topology_uri` | 可选 zip | ✗ | — | 每 ligand 一份 .top+.gro |
| `water_model` | enum | ✗ | `amber/tip3p` | 见 models.py `WaterModel` |
| `hmr_factor` | float\|null | ✗ | `2.5` | 若已 HMR 则设 null |
| `replicas` | int | ✗ | `3` | 独立重复次数 |
| `threads`, `num_jobs` | int | ✗ | `12`, `10` | 并发控制 |
| `nwindows_ligand_vdw`/`_coul`, `nwindows_complex_vdw`/`_coul`/`_bonded` | int | ✗ | 11, 11, 21, 11, 11 | λ 窗口数 |
| `scheduler` | `frontend`\|`slurm` | ✗ | `frontend` | opt-in Slurm 需要 apptainer bind |
| `global_config_yaml` | str (YAML) | ✗ | — | Power-user 逃生舱：`cluster/mdrun/mdp/extra_directives` 覆盖 |

**响应**：`JobInfo`（`status=pending`）；用 `GET /api/jobs/<id>` 轮询到
`completed` / `failed`；用 `/api/jobs/<id>/files` 列出产物、`/file/<name>`
下载单文件、`/download` 拉 zip。

### `POST /api/calculate/mmpbsa`

同 FEP，但改为 MMPBSA-specific 字段：

| 字段 | 默认 | 说明 |
|---|---|---|
| `samples` | 20 | 每 replica 采样帧数 |
| `mmpbsa_yaml` | — | 覆盖 BindFlow `global_config.mmpbsa` 块 |

FEP 的 `nwindows_*` 字段在 MMPBSA endpoint 不接受（pydantic 422）。

### 例子

```bash
# FEP 快速跑（3 配体，3 replicas）：
curl -X POST http://localhost:9000/api/calculate/fep \
    -F protein=@receptor.pdb \
    -F ligands=@lig_a.sdf -F ligands=@lig_b.sdf -F ligands=@lig_c.sdf \
    -F water_model=amber/tip3p -F replicas=3 \
    -F threads=8 -F num_jobs=6

# MMPBSA 走 job:// URI（前一步生成的对接姿势）：
curl -X POST http://localhost:9000/api/calculate/mmpbsa \
    -F protein_uri=job://prev-dock-job/receptor.pdb \
    -F ligands_zip_uri=job://prev-dock-job/ligands.zip \
    -F samples=40 -F replicas=3
```

## CLI 批处理模式（HPC 主入口）

同一镜像 `python -m server ...` 一次性同步跑：

```bash
apptainer exec --nv --bind /scratch:/scratch \
    /opt/sif/bindflow-server.sif \
    python -m server fep \
    --protein $INPUT_DIR/receptor.pdb \
    --ligands-dir $INPUT_DIR/ligands \
    --output-dir /scratch/$SLURM_JOB_ID/ \
    --replicas 3 --threads 16 --num-jobs 8 \
    --water-model amber/tip3p --hmr-factor 2.5
```

复杂参数走 `--params-json`：

```bash
apptainer exec bindflow-server.sif python -m server mmpbsa \
    --protein rec.pdb --ligands-dir ligands/ \
    --output-dir out/ \
    --params-json '{"samples": 20, "replicas": 3, "solv_ion_conc": 0.15}'
```

`--json` 让 CLI 输出机器可读的完成信息（便于 sbatch epilog）。

### sbatch 模板

```bash
#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16 --mem=64G
#SBATCH --time=48:00:00

apptainer exec --nv \
    --bind $SCRATCH:/scratch \
    /opt/sif/bindflow-server.sif \
    python -m server fep \
    --protein $INPUT_DIR/protein.pdb \
    --ligands-dir $INPUT_DIR/ligands \
    --output-dir /scratch/$SLURM_JOB_ID/ \
    --replicas 3 --threads 16 --num-jobs 8
```

## 配置

`env_prefix=BINDFLOW_`：

| Env var | 默认 | 说明 |
|---|---|---|
| `BINDFLOW_JOBS_BASE_DIR` | `/data/bindflow_jobs` | Job root |
| `BINDFLOW_ROOT` | `/opt/bindflow` | BindFlow install root |
| `BINDFLOW_PYTHON` | `/opt/conda/envs/bindflow/bin/python` | Conda env Python |
| `BINDFLOW_INFERENCE_SCRIPT` | `/opt/bindflow/server/inference.py` | Wrapper 路径 |
| `BINDFLOW_WEIGHTS_DIR` | `/data/models/bindflow` | 无 NN 权重；字段保留一致性 |
| `BINDFLOW_MAX_CONCURRENT_JOBS` | 1 | 一次一个 workflow |
| `BINDFLOW_TASK_ENDPOINTS_ENABLED` | false | 不注册 `/api/tasks/*`（HPC-primary） |
| `BINDFLOW_SUBPROCESS_TIMEOUT_S` | 604800 (7d) | 子进程硬顶 |
| `BINDFLOW_OSS_REGION` | cn-hangzhou | OSS URI |

## 本地开发

```bash
# 一次性：vendor 上游到 upstream/
./services/bindflow-server/scripts/vendor.sh

# Ruff
uvx ruff check services/bindflow-server/

# 单元测试（HTTP + CLI）
uv run python -m pytest services/bindflow-server/tests/ -v

# Docker build（约 25-45 分钟；下载 conda + 装 bioconda gromacs + pip deps）
docker build --platform linux/amd64 -t bindflow-server \
    -f services/bindflow-server/Dockerfile .

# HTTP 快速起服务
docker run --rm -p 9000:9000 bindflow-server &
curl http://localhost:9000/api/manifest | jq .endpoints
curl http://localhost:9000/healthz/detail | jq
kill %1

# CLI 帮助
docker run --rm bindflow-server \
    /opt/conda/envs/bindflow/bin/python -m server --help
docker run --rm bindflow-server \
    /opt/conda/envs/bindflow/bin/python -m server fep --help
```

## 打包 SIF（HPC）

```bash
# 1) push image 到 harbor
make push-bindflow-server

# 2) 在 HPC 转 SIF
apptainer pull bindflow-server.sif docker://harbor.ruosheng.bio/aliyun_fc/bindflow-server:v0.0.1

# 3) 用户 sbatch 用（见上文模板）
```

## 已知限制

- **长跑**：单次 FEP 项目常常 20+ 小时；HTTP submit/poll 模式在开发机上跑
  只推荐 MMPBSA 或极小规模 FEP。生产走 CLI + sbatch。
- **espaloma FF 未包含**：v0.0.1 只支持 openff / gaff。见设计文档 §11.3。
- **GROMACS 走 bioconda / conda-forge (`>=2023,<2026`, CPU-only)**：具体版本
  由 solver 从两个 channel 里选（bioconda 不发所有 patch 版本，锁死 patch
  pin 会 unresolvable）；未来上 GPU 需要 v0.0.2 变体镜像或换 source-build
  路径。
- **`global_config_yaml` 是可信客户端专用**：可注入 shell 命令
  (`extra_directives.dependencies`)；不要暴露给不受信任的调用方。见 §11.5。

## 相关文档

- [bindflow-server 设计文档](../../engineering/decisions/2026-07-06-bindflow-server-design.md)
- [Service 框架抽象设计](../../engineering/decisions/2026-05-12-service-framework-design.md)
- [CLI 批处理模式设计](../../engineering/decisions/2026-05-29-cli-batch-mode.md)
- [新增服务 cookbook](../../engineering/guides/adding-a-new-service.md)
- [apptainer/singularity 兼容性](../../engineering/guides/apptainer-compatibility.md)
- 上游：[ale94mleon/BindFlow](https://github.com/ale94mleon/BindFlow)（GPL-3.0）
- gmx_MMPBSA：[Valdes-Tresanco-MS/gmx_MMPBSA](https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA) @ pinned commit
