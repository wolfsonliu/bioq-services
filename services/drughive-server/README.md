# drughive-server

> ⚠ **LICENSE WARNING** — Upstream [DrugHIVE](https://github.com/jssweller/DrugHIVE)
> is released under [**USC-RL v2.0**](https://github.com/jssweller/DrugHIVE/blob/main/LICENSE),
> a **non-commercial academic research** license.  Commercial use requires
> separate authorization from USC Stevens Center for Innovation
> (`info@stevens.usc.edu`).  This service is deployed for **internal
> research only** — do NOT expose the base URL publicly and do NOT integrate
> into commercial SaaS offerings without USC clearance.
> See [engineering/decisions/2026-07-02-drughive-server-design.md](../../engineering/decisions/2026-07-02-drughive-server-design.md) §Risks §1.

FastAPI HTTP wrapper for [**DrugHIVE**](https://github.com/jssweller/DrugHIVE)
(Weller & Rohs, *JCIM* 2024, [10.1021/acs.jcim.4c01193](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01193))
— a hierarchical VAE for **structure-based drug design (SBDD)**.  Given a
**protein pocket** + **reference ligand**, generate N new ligand candidates;
optionally optimize a scoring criterion (QVina2 affinity / QED / SA / ALogP)
over multiple genetic cycles.

Built on [bioq-service-framework](../_framework/).

Base image: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`;
conda env (Python 3.10 + PyTorch 1.12.1 + cu117 + RDKit + OpenBabel +
QVina2 via `bioconda::qvina`).

## Architecture

```
Client / Agent
  ↓ HTTP (multipart: target.pdb + ligand.sdf [+ substruct.sdf | pdbqt])
┌────────────────────────────────────────────────────────────────┐
│  FastAPI + bioq-service-framework  (port 9000)             │
│                                                                │
│  Service-specific                                              │
│    POST /api/generate           (de novo, MolGenerator)        │
│    POST /api/generate_spatial   (scaffold hop, MolGeneratorSp.)│
│    POST /api/optimize           (multi-cycle QVina2)           │
│    POST /api/tasks/<same-name>  (FC async task mode)           │
│                                                                │
│  Framework                                                     │
│    GET  /api/manifest                                          │
│    GET  /openapi.json                                          │
│    GET  /healthz, /healthz/detail   (weights + qvina2 probe)   │
│    GET  /api/jobs/{id}/...                                     │
└────────────────────────────────────────────────────────────────┘
  ↓ subprocess (cwd=/opt/drughive, single-GPU serial)
generate_molecules.py <cfg.yml>  OR  generate_optimize.py <cfg.yml>
  ↓
NAS: <jobs_base>/<job_id>/{input/{config.yml, target.pdb, ligand.sdf},
                           output/mols_gen_*.sdf | mols_pred_*.sdf | ...,
                           logs/run.log, job.json}
```

## API overview

### `POST /api/generate` — de novo ligand generation

Uses `MolGenerator` (upstream `generate_molecules.py` without
`substruct_modify_*` YAML keys).

multipart/form-data:

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `target` / `target_uri` | file (`.pdb`) / URI | ✓ | — | Protein pocket |
| `ligand` / `ligand_uri` | file (`.sdf`) / URI | ✓ | — | Reference ligand (drives posterior mix) |
| `n_samples` | int (1–5000) | — | `10` | Number of candidates |
| `pdb_id` | str | — | `"target"` | Tag used in output filenames |
| `zbetas` | list[float] len 4 or scalar | — | `[0, 0, 0, 0]` | Prior↔posterior interpolation per resolution.  Scalar is broadcast |
| `temps` | list[float] len 4 or scalar | — | `[0.5, 0.5, 0.5, 0.5]` | Sampling temperature per resolution |
| `random_rotate` / `random_translate` | bool | — | `true` / `false` | Data augmentation |
| `ffopt_mols` | bool | — | `true` | RDKit FF post-processing |
| `mol_filter` | JSON | — | `{}` | `ring_sizes`, `ring_system_max`, `ring_loops_max`, `dbl_bond_pairs`, `n_atoms_min` |

Example:

```bash
curl -X POST $URL/api/generate \
    -F target=@5d3h_pocket.pdb \
    -F ligand=@5d3h_ligand.sdf \
    -F n_samples=10 \
    -F pdb_id=5d3h
# → { "job_id": "...", "status": "pending", ... }

# poll
curl $URL/api/jobs/<job_id>
# completed → download
curl $URL/api/jobs/<job_id>/files
# → ["output/mols_gen_5d3h.sdf", ...]
curl -O $URL/api/jobs/<job_id>/file/output/mols_gen_5d3h.sdf
```

### `POST /api/generate_spatial` — scaffold hopping

Uses `MolGeneratorSpatial` (upstream routes on presence of
`substruct_modify_path` or `substruct_modify_pattern` in YAML).  Provide
**exactly one** of:

- `substruct_modify` / `substruct_modify_uri` — SDF file of the fragment
  to preserve, OR
- `substruct_modify_pattern` — SMILES / SMARTS string

All other fields same as `/api/generate`.  Default `zbetas` is
`[0.3, 0.3, 0.3, 0.3]` (posterior-leaning; makes sense when preserving
part of the reference).

```bash
# option 1: SMARTS pattern
curl -X POST $URL/api/generate_spatial \
    -F target=@4w9f_pocket.pdb \
    -F ligand=@4w9f_ligand.sdf \
    -F substruct_modify_pattern='[CH2]C1:C:C:C:C:C:1' \
    -F n_samples=10 -F pdb_id=4w9f

# option 2: SDF fragment
curl -X POST $URL/api/generate_spatial \
    -F target=@4w9f_pocket.pdb \
    -F ligand=@4w9f_ligand.sdf \
    -F substruct_modify=@fragment.sdf \
    -F n_samples=10 -F pdb_id=4w9f
```

### `POST /api/optimize` — multi-cycle QVina2 optimization

Uses `generate_optimize.py` — genetic-algorithm style: initial population
→ dock with QVina2 → select best parents → resample around them → repeat.

Required for `key_opt=affinity_qvina`: `target_pdbqt` (or `_uri`) — PDBQT
formatted receptor for QVina docking.

Selected fields:

| Field | Default | Notes |
|---|---|---|
| `key_opt` | `affinity_qvina` | `affinity_qvina` \| `qed` \| `alogp` \| `sa` |
| `opt_increase` | `false` | `true` = maximize (QED); `false` = minimize (affinity kcal/mol) |
| `n_cycles` | `8` (1-20) | Optimization cycles |
| `n_samples_initial` | `1000` (10-10000) | Initial population size |
| `n_samples` | `20` (per parent) | Children per parent per cycle |
| `n_best_parents` | `20` | Parents kept each cycle |
| `zbetas` | `[0.3]` (broadcast to n_cycles) | Per-cycle prior/posterior mix.  For 8 cycles, paper uses `[0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1]` |
| `zbetas_initial`, `temps_initial` | `0.3`, `1.0` | For initial population |
| `cluster_parents` | `true` | Cluster parent pool by Tanimoto similarity |
| `protonate` | `true` | obabel protonation before QVina |

**Runtime warning**: default params run 8 × 1000 × 20 = 160 000 molecules × QVina docking each — 4-8 hours per call.
**Always use `/api/tasks/optimize` (async task mode)** for real runs; the
sync endpoint will hit HTTP gateway timeouts under FC deployment.

Test-friendly quick run:

```bash
curl -X POST $URL/api/tasks/optimize \
    -H "X-Fc-Invocation-Type: Async" \
    -H "X-Bioagent-Job-Id: my-opt-run-001" \
    -F target=@5d3h_pocket.pdb \
    -F ligand=@5d3h_ligand.sdf \
    -F target_pdbqt=@5d3h_pocket.pdbqt \
    -F key_opt=affinity_qvina \
    -F n_cycles=2 \
    -F n_samples_initial=20 \
    -F n_samples=4 \
    -F n_best_parents=2
# → 202 Accepted; poll GET /api/jobs/my-opt-run-001
```

### `POST /api/tasks/{name}` — FC async task mode

Same input schema as sync variants above; HTTP returns 202 immediately and
FC binds the request lifecycle to the compute process (no 30 s HTTP
gateway recycle risk, platform-level dedup on `X-Fc-Async-Task-Id`).
FC console must have **异步任务模式 / async task mode** enabled — see
`engineering/decisions/2026-06-17-fc-async-task-mode.md`.

Headers:

- `X-Fc-Invocation-Type: Async` — required to trigger async mode
- `X-Bioagent-Job-Id: <task-id>` — client-supplied stable ID (recommended)
- `X-Fc-Async-Task-Id: <task-id>` — FC platform ID (usually same as job-id)

## URI schemes

Every file input accepts either a multipart upload OR a `<field>_uri`
field.  Supported schemes:

| Scheme | Use case |
|---|---|
| multipart upload | Default — small files (<128 KiB async event cap) |
| `oss://<bucket>/<key>` | Alibaba OSS object |
| `file:///data/...` | NAS-mounted file (FC / SIF) |
| `job://<job_id>/<file>` | Chain from previous job's `output/` |
| `http(s)://...` | Public URL |

Typical DrugHIVE inputs are small (pocket PDB ≤ 100 KB, ligand SDF ≤ 10 KB,
PDBQT ≤ 100 KB) so multipart upload is fine — URI fallback is only needed
for the rare case of large multi-chain complexes.

## Weights (external — NAS)

Weights are **not** baked into the image.  They live on NAS at
`/data/models/drughive/checkpoints/`:

```
/data/models/drughive/
└── checkpoints/
    └── drughive_model_ch9.ckpt      # single checkpoint from Zenodo
```

Pretrained model: [Zenodo 10.5281/zenodo.12668687](https://doi.org/10.5281/zenodo.12668687)
— `drughive_model_ch9.ckpt`, `model_id = c9_pdbzinc` (trained on PDBbind + ZINC).

### Pre-stage weights to NAS (once per NAS)

```bash
# option A: stage locally, then rsync to NAS
./services/drughive-server/scripts/fetch_weights.sh
rsync -a services/drughive-server/weights/ \
    <NAS-mount>:/data/models/drughive/checkpoints/

# option B: download directly to NAS
WEIGHTS_DST=<NAS-mount>/data/models/drughive/checkpoints \
    ./services/drughive-server/scripts/fetch_weights.sh
```

### FC deployment verify

```bash
curl https://fc-drughive-XXX.cn-hangzhou-vpc.fcapp.run/healthz/detail
# Expected:
# {
#   "status": "ok",
#   "weights_dir": "/data/models/drughive/checkpoints",
#   "weights_loaded": true,
#   "weights_missing": {},
#   "qvina2_available": true,
#   "qvina2_path": "/opt/conda/envs/drughive/bin/qvina2",
#   ...
# }
```

### SIF / HPC (Apptainer)

```bash
apptainer run --nv \
    --bind /scratch/models/drughive:/data/models/drughive \
    --bind /scratch/jobs:/data/drughive_jobs \
    drughive-server.sif
```

## Configuration (env vars)

| Env var | Default | Notes |
|---|---|---|
| `DRUGHIVE_JOBS_BASE_DIR` | `/data/drughive_jobs` | NAS-mounted jobs directory |
| `DRUGHIVE_ROOT` | `/opt/drughive` | subprocess cwd (upstream reads relative `data/pains_filter/PAINS.sieve`) |
| `DRUGHIVE_PYTHON` | `/opt/conda/envs/drughive/bin/python` | Conda-env Python |
| `DRUGHIVE_GENERATE_SCRIPT` | `/opt/drughive/generate_molecules.py` | Upstream ligand generation entry |
| `DRUGHIVE_OPTIMIZE_SCRIPT` | `/opt/drughive/generate_optimize.py` | Upstream optimization entry |
| `DRUGHIVE_WEIGHTS_DIR` | `/data/models/drughive/checkpoints` | NAS mount |
| `DRUGHIVE_CHECKPOINT_FILENAME` | `drughive_model_ch9.ckpt` | Single Zenodo ckpt |
| `DRUGHIVE_MODEL_ID` | `c9_pdbzinc` | Passed to upstream YAML |
| `DRUGHIVE_DOCKING_CMD` | `qvina2.1` | Vendored QuickVina 2 binary from QVina github (Apache-2.0); `qvina2` symlink also available |
| `DRUGHIVE_MAX_CONCURRENT_JOBS` | `1` | Single-GPU serial |
| `DRUGHIVE_SESSION_HEADER_NAME` | `bioagent-session-id` | FC session affinity |
| `DRUGHIVE_TASK_ENDPOINTS_ENABLED` | `true` | Toggle `/api/tasks/*` |

## Local development

```bash
# 1a. Vendor DrugHIVE upstream (once, or when bumping SHA)
./services/drughive-server/scripts/vendor.sh

# 1b. Vendor QuickVina 2 static binary (once, Apache-2.0)
./services/drughive-server/scripts/vendor_qvina.sh

# 2. Offline unit tests — no Docker, no GPU, no weights
uv run python -m pytest services/drughive-server/tests/test_app.py -v
uv run python -m pytest services/drughive-server/tests/test_cli.py -v

# 3. Lint
uvx ruff check services/drughive-server/
```

## Docker

```bash
# Build (from bioagent project root, after vendor.sh)
make build-drughive-server
# equivalent to:
docker build --platform linux/amd64 -t drughive-server \
    -f services/drughive-server/Dockerfile .

# HTTP mode
docker run --rm --gpus all -p 9000:9000 \
    -v /scratch/models/drughive:/data/models/drughive:ro \
    -v /scratch/jobs:/data/drughive_jobs \
    drughive-server

# CLI batch mode (Slurm / sbatch)
docker run --rm --gpus all \
    -v /scratch/models/drughive:/data/models/drughive:ro \
    -v /scratch/inputs:/data/inputs:ro \
    -v /scratch/results:/data/results \
    drughive-server \
    /opt/conda/envs/drughive/bin/python -m server generate \
        --target /data/inputs/pocket.pdb \
        --ligand /data/inputs/lig.sdf \
        --output-dir /data/results/ \
        --params-json '{"n_samples": 10, "pdb_id": "5d3h"}'
```

## Alibaba Cloud FC deployment

| Setting | Value |
|---|---|
| Image | `harbor.ruosheng.bio/aliyun_fc/drughive-server:v0.0.1` |
| Instance type | `fc.gpu.tesla.1` (T4 8 GB is enough; Ada / Blackwell also work) |
| Timeout | `43200` s (12h — `/api/optimize` default params can run 4-8h) |
| Memory / CPU | 16 GB / 8 vCPU (QVina2 is CPU-heavy) |
| Async task mode | **Enabled** (max concurrent 5-10) |
| NAS (jobs) | `/fc → /data` |
| NAS (weights) | `/fc → /data/models/drughive` (read-only) |
| Session affinity | Enabled (`bioagent-session-id`) |
| Keepalive URL | Empty (async mode makes it unnecessary) |
| PreStop hook | Enabled |

Push + update:

```bash
echo v0.0.2 > services/drughive-server/VERSION   # bump on release
make push-drughive-server                         # build + tag + push to harbor
# then update the FC function's image via console
```

## Testing FC deployment

```bash
# smoke (offline-mocked test_fc.py smoke + inference sanity)
RUN_FC_TESTS=1 uv run python -m pytest -m fc \
    services/drughive-server/tests/test_fc.py -v

# full async task suite
RUN_FC_TESTS=1 uv run python -m pytest -m fc \
    services/drughive-server/tests/test_fc_task.py -v
```

## References

- **Paper**: Weller & Rohs, "Structure-Based Drug Design with a Deep Hierarchical
  Generative Model", *J. Chem. Inf. Model.* 2024, [10.1021/acs.jcim.4c01193](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01193)
- **Upstream repo**: https://github.com/jssweller/DrugHIVE (USC-RL v2.0, pinned SHA `d965edf6e6770bc15c38860e0f7e773bdf28975b`)
- **Weights**: [Zenodo 10.5281/zenodo.12668687](https://doi.org/10.5281/zenodo.12668687)
- **QVina2**: https://qvina.github.io (packaged via bioconda `qvina 1.2.2`)
- **Design doc**: [engineering/decisions/2026-07-02-drughive-server-design.md](../../engineering/decisions/2026-07-02-drughive-server-design.md)
- **Implementation plan**: [engineering/decisions/2026-07-02-drughive-server-plan.md](../../engineering/decisions/2026-07-02-drughive-server-plan.md)
