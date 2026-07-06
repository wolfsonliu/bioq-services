# qligfep-server

Q + qligfep FEP/LIE 全流程 HTTP + CLI 包装。9 endpoint 覆盖 ligand
参数化（OpenFF）→ protein prep → FEP/LIE setup → 单 λ window MD run
→ DDG 后处理。

**HPC-primary**：主要用法是 `apptainer exec --nv qligfep-server.sif
.venv/bin/python -m server <endpoint>`，通过用户自己的 sbatch array
并发跑 λ window。HTTP 端点保留用于本地 dev / smoke，`/api/tasks/*` 关闭。

设计文档：[engineering/decisions/2026-07-06-qligfep-server-design.md](../../engineering/decisions/2026-07-06-qligfep-server-design.md)
实施计划：[engineering/decisions/2026-07-06-qligfep-server-plan.md](../../engineering/decisions/2026-07-06-qligfep-server-plan.md)

## Endpoint

| Path | 用途 | 典型耗时 | HTTP 实用 |
|---|---|---|---|
| `POST /api/ligprep` | OpenFF ligand → Q lib/prm/pdb | 5-60 s | ✅ |
| `POST /api/protprep` | Spherical boundary protein prep | 1-5 s | ✅ |
| `POST /api/cog` | Center-of-geometry helper | < 1 s | ✅ |
| `POST /api/setup-ligfep` | QligFEP dual-topology setup | 5-30 s | ✅ |
| `POST /api/setup-resfep` | QresFEP residue mutation setup | 5-30 s | ✅ |
| `POST /api/setup-lie` | QLIE setup | 5-30 s | ✅ |
| `POST /api/run-fep` | 单 λ window MD (qprep + qdyn/qdynp) | 30 min - 2 h | ⚠️ CLI only |
| `POST /api/analyze-fep` | DDG (Zwanzig/OS/BAR) 后处理 | 10 s - 2 min | ✅ |
| `POST /api/analyze-lie` | LIE 后处理 | 10 s - 2 min | ✅ |

## Vendor + Build

```bash
# 1. Vendor upstream code
./services/qligfep-server/scripts/vendor.sh
# 首次跑会打印 Q6 resolved SHA，拷回 vendor.sh 的 Q6_SHA= 后再跑一次固化。

# 2. Build docker image (20-40 min，主要在 Q6 编译)
make build-qligfep-server

# 3. Convert to SIF for HPC (可选)
make sif-qligfep-server
```

## HPC sbatch 模板

```bash
#!/bin/bash
#SBATCH --job-name=qligfep-w${SLURM_ARRAY_TASK_ID}
#SBATCH --time=02:00:00
#SBATCH --array=0-50%8       # 51 windows, 最多 8 并发
#SBATCH --ntasks=4           # MPI ranks for qdynp

apptainer exec \
    --bind /scratch/${USER}/qligfep_jobs:/data/qligfep_jobs \
    qligfep-server.sif \
    .venv/bin/python -m server run-fep \
        --setup-dir /scratch/${USER}/inputs/setup_leg_protein \
        --window-idx ${SLURM_ARRAY_TASK_ID} \
        --leg protein \
        --device mpi --nprocs 4 \
        --output-dir /scratch/${USER}/qligfep_jobs/win_${SLURM_ARRAY_TASK_ID}
```

## Force fields

内置（vendor 的 `upstream/qligfep/FF/`）：`OPLS2005`, `OPLS2015`,
`OPLSAAM`, `AMBER14sb`, `CHARMM36`。

Ligand 参数化只支持 **OpenFF**（v0.0.1）——`ffld_server` / `cgenff` /
PyMOL Prep Wizard 需商用授权，未集成。已有 residue lib/prm 可直接沿用；
自定义 ligand 若不能用 OpenFF 表达，需 offline 生成 lib/prm/pdb 后传给
setup-* 端点。

## v0.0.1 已知限制

1. **无 CUDA 支持**：vendored 的 Q6（esguerra/Q6 @ 202d90c）本身没有 CUDA
   构建目标，镜像只 build `qdyn`（默认 gfortran）+ `qdynp`（MPI）。
   `device=gpu` 会返回 `rc=3` 明确错误。设计文档 §2 目标 6 写的
   "MPI + CUDA" 计划实际收敛到 "MPI only"。若 upstream 添加 CUDA 或换到
   带 CUDA 的 fork，未来版本可恢复。
2. **`settings.task_endpoints_enabled=False` 是 hard-coded 设计约定**：
   不要打开除非同时决定接 FC——run-fep 单窗口 30 min - 2 h 超过 FC 30 s
   HTTP gateway 上限，任务模式也不适配。
3. **Ligand 含金属或罕见杂原子时 OpenFF 会失败**——错误信息会指出具体化学
   问题；此类 ligand 需 offline 生成 lib/prm/pdb。
4. **CHARMM 参数化不通**：`generate_charmm.py` 要 cgenff，我们不装。
   只支持沿用已 vendor 的 CHARMM36 residue lib/prm。

## Tests

```bash
# Offline HTTP + CLI smoke (mock subprocess) — 应全绿
uv run python -m pytest services/qligfep-server/tests/ -v \
    --ignore=services/qligfep-server/tests/test_ligprep_openff.py

# Real OpenFF ligprep (opt-in, 需 openff-toolkit)
uv run python -m pytest -m slow services/qligfep-server/tests/test_ligprep_openff.py -v
```

FC 集成测试：**不做**（本服务不部署 FC）。
