# diffdock-server

FastAPI HTTP wrapper for [**DiffDock**](https://github.com/gcorso/DiffDock)
— Corso et al., ICLR 2023 (arXiv:2210.01776); DiffDock-L Corso 2024
(arXiv:2402.18396) — the state-of-the-art diffusion generative model for
**small-molecule blind docking**.  Given a **protein structure** (PDB or
sequence) + a **ligand** (SDF/MOL2 or SMILES), sample N candidate poses
on the product manifold `T(3) × SO(3) × SO(2)^m` and rank them by an
independently trained confidence model.

Upstream license: **MIT**.  Built on [bioagent-service-framework](../_framework/).

Base image: `nvidia/cuda:11.7.1-devel-ubuntu22.04` (builder) →
`nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` (runtime); conda env
(Python 3.9 + PyTorch 1.13.1 + cu117 + PyG 2.2 + e3nn 0.5.1 + fair-esm
[esmfold] 2.0 + openfold from git + rdkit 2022.03).

Related: [engineering/decisions/2026-07-06-diffdock-server-design.md](../../engineering/decisions/2026-07-06-diffdock-server-design.md),
[wiki/foundational-tools/diffdock.md](../../wiki/foundational-tools/diffdock.md).

## Architecture

```
Client / Agent
  ↓ HTTP (multipart: protein.pdb [| protein_sequence]
                    + ligand.sdf [| ligand_description SMILES])
┌────────────────────────────────────────────────────────────────┐
│  FastAPI + bioagent-service-framework  (port 9000)             │
│                                                                │
│  Service-specific                                              │
│    POST /api/dock           (single-complex docking)           │
│    POST /api/tasks/dock     (FC async task mode)               │
│                                                                │
│  Framework                                                     │
│    GET  /api/manifest                                          │
│    GET  /openapi.json                                          │
│    GET  /healthz, /healthz/detail  (weights + LUT probe)       │
│    GET  /api/jobs/{id}/...                                     │
└────────────────────────────────────────────────────────────────┘
  ↓ subprocess (cwd=/opt/diffdock; SO(3)/torus LUT CWD-relative)
run_inference.py → upstream inference.main()
  ↓
NAS: <jobs_base>/<job_id>/{input/{protein.pdb, ligand.sdf},
                           output/<complex_name>/{rank1.sdf,
                              rank<r>_confidence<c>.sdf,
                              confidence_scores.json,
                              [<name>_esmfold.pdb if sequence input]},
                           logs/run.log, job.json}
```

## API overview

### `POST /api/dock` — single-complex blind docking

multipart/form-data:

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `protein` / `protein_uri` / `protein_sequence` | file `.pdb` / URI / str (20–1500 aa) | ✓ (exactly one) | — | Protein structure |
| `ligand` / `ligand_uri` / `ligand_description` | file `.sdf/.mol2` / URI / SMILES str | ✓ (exactly one) | — | Ligand |
| `complex_name` | str, `[A-Za-z0-9_-]+` | — | `"complex_0"` | Output subdirectory |
| `samples_per_complex` | int (1–100) | — | `10` | Reverse diffusion samples (paper peak = 40) |
| `inference_steps` | int (10–40) | — | `20` | Total denoise steps |
| `actual_steps` | int (≤ inference_steps) | — | `19` | Actual steps to run |
| `batch_size` | int (1–20) | — | `10` | GPU batch |
| `no_final_step_noise` | bool | — | `true` | (upstream store_true; requested `false` is ignored — see design doc §Risks §4) |
| `save_visualisation` | bool | — | `false` | Dump reverse-diffusion trajectory PDB per rank |
| `seed` | int | — | (framework fills) | RNG seed |

Response: `JobInfo` (framework).  Poll `/api/jobs/<id>` until completed.

Three worked examples:

```bash
# 1. PDB + SDF file upload (fastest path)
curl -X POST $URL/api/dock \
    -F protein=@target.pdb \
    -F ligand=@ligand.sdf \
    -F complex_name=1a0q \
    -F samples_per_complex=10

# 2. PDB + SMILES ligand
curl -X POST $URL/api/dock \
    -F protein=@target.pdb \
    -F 'ligand_description=COc1ccc(C#N)cc1' \
    -F complex_name=abl1_inhibitor

# 3. protein_sequence + SMILES (ESMFold pipeline)
curl -X POST $URL/api/dock \
    -F 'protein_sequence=MKW...' \
    -F 'ligand_description=CCOc(cc1)ccc1NC(=O)C' \
    -F complex_name=novel_target
```

### `POST /api/tasks/dock` — FC async task mode

Same body as `/api/dock` plus headers:

```
X-Fc-Invocation-Type: Async
X-Bioagent-Job-Id: <caller-generated-id>
X-Fc-Async-Task-Id: <same as X-Bioagent-Job-Id>
```

Returns `202` immediately.  Client polls `/api/jobs/<X-Bioagent-Job-Id>`
for the final JobInfo.  Preferred over sync `/api/dock` for FC jobs
longer than ~2 min (avoids HTTP gateway 30 s timeout; better GPU
scheduling).  See
[engineering/decisions/2026-06-17-fc-async-task-mode.md](../../engineering/decisions/2026-06-17-fc-async-task-mode.md).

### `GET /healthz/detail` — probe

Beyond framework fields, this service adds:

- `weights_loaded` — score + confidence + ESM-2 checkpoints present
- `weights_missing` — dict of missing paths (empty when OK)
- `esmfold_available` — soft check; only needed for `protein_sequence` input
- `so3_cache_ok` / `torus_cache_ok` — SO(3) / torus LUT `.npy` files present
  under `/opt/diffdock/` (pre-computed at build time; missing = first
  request will hang ~2 min while regenerating)

## Output layout

```
<job_dir>/output/<complex_name>/
├── rank1.sdf                              ← top-1 pose (no confidence suffix)
├── rank1_confidence<c>.sdf                ← rank-1 with confidence in filename
├── rank<r>_confidence<c>.sdf              ← rank 2 to samples_per_complex
├── confidence_scores.json                 ← [{rank, confidence, sdf}] ordered
└── <complex_name>_esmfold.pdb             ← only when protein_sequence input
```

`confidence_scores.json` is produced by the wrapper post-processor by
parsing the confidence values embedded in each ranked SDF filename.

## Weights

Weights are **not baked into the image** — mounted from NAS at
`/data/models/diffdock/`:

```
/data/models/diffdock/
├── score_model/best_ema_inference_epoch_model.pt                      (~1 GB)
├── score_model/model_parameters.yml
├── confidence_model/best_model_epoch75.pt                            (~250 MB)
├── confidence_model/model_parameters.yml
└── esm_cache/hub/checkpoints/
    ├── esm2_t33_650M_UR50D.pt                                          (~2.5 GB)
    ├── esm2_t33_650M_UR50D-contact-regression.pt                       (~4 MB)
    └── esmfold_3B_v1.pt                                                (~5 GB, optional)
```

Stage weights with:

```bash
WEIGHTS_DST=/mnt/nas/data/models/diffdock \
    ./services/diffdock-server/scripts/fetch_weights.sh

# Skip 5 GB ESMFold if you never use protein_sequence input:
DIFFDOCK_SKIP_ESMFOLD=1 ./services/diffdock-server/scripts/fetch_weights.sh
```

## Build & deploy

```bash
# 1. Vendor upstream (once, or to bump the pinned SHA)
./services/diffdock-server/scripts/vendor.sh

# 2. Stage weights to NAS (or local weights/ dir for inspection)
./services/diffdock-server/scripts/fetch_weights.sh

# 3. Build image (from project root)
docker build --platform linux/amd64 -t diffdock-server \
    -f services/diffdock-server/Dockerfile .
# Or via Makefile:
make images/diffdock-server
```

FC deployment target (see design doc §Deployment):

| Item | Recommended |
|---|---|
| GPU instance | `fc.gpu.ampere.1` (A10 24GB) |
| Timeout | 3600 s |
| Memory | 32 GB |
| CPU | 8 vCPU |
| NAS mount | `/fc → /data` |
| Async task mode | enabled |
| Session affinity | enabled (`bioagent-session-id` header) |

## Testing

```bash
# Offline (no FC / no GPU)
uv run python -m pytest services/diffdock-server/tests/test_app.py \
    services/diffdock-server/tests/test_cli.py -v

# FC integration (opt-in, requires deployed base URL in aliyun_fc_url.md)
RUN_FC_TESTS=1 uv run python -m pytest -m fc \
    services/diffdock-server/tests/test_fc.py -v

RUN_FC_TESTS=1 uv run python -m pytest -m fc \
    services/diffdock-server/tests/test_fc_task.py -v
```

## Boundaries with sibling services

| Task | Service | Notes |
|---|---|---|
| Small-molecule blind docking (rigid protein) | **diffdock-server** (this) | Structure known or foldable; needs SMILES/SDF |
| Rigid protein-protein docking | [diffdock-pp-server](../diffdock-pp-server/) | Both inputs are proteins |
| Sequence → structure + docking co-folding | [boltz-server](../boltz-server/) / esmfold-driven pipelines | Best when starting from sequence only + no ligand structure |
| ADMET / property prediction | [openadmet-server](../openadmet-server/) | Downstream: score DiffDock's poses |
| Structure-based drug **design** | [drughive-server](../drughive-server/) | Generative — DiffDock only docks pre-existing ligands |

## References

- Corso, Stärk, Jing, Barzilay, Jaakkola. *"DiffDock: Diffusion Steps, Twists,
  and Turns for Molecular Docking"*. ICLR 2023. arXiv:2210.01776
- Corso, Deng, Polizzi, Barzilay, Jaakkola. *"Deep Confident Steps to New
  Pockets: Strategies for Docking Generalization"* (DiffDock-L). ICLR 2024.
  arXiv:2402.18396
- Wiki: [foundational-tools/diffdock.md](../../wiki/foundational-tools/diffdock.md)
