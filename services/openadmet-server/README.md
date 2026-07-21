# openadmet-server

FastAPI HTTP wrapper for [**OpenADMET Models**](https://github.com/OpenADMET/openadmet-models)
(MIT / OMSF) — a general-purpose ADMET (Absorption / Distribution /
Metabolism / Excretion / Toxicity) ML modeling toolbox.  **Built on
[bioq-service-framework](../_framework/).**

Not a single-model service — OpenADMET is a **framework**.  The server
exposes a **model registry** (`GET /api/models`) listing NAS-pre-staged
anvil-trained model directories; clients select one or more `model_name`
values and submit SMILES for prediction.

Base image: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`;
conda env (Python 3.12 + PyTorch-GPU + chemprop + molfeat + rdkit + all
of `openadmet-models-gpu.yaml`).

## v0.0.1 scope

- ✅ `POST /api/predict` — inference with 1+ NAS-registered models
- ✅ `POST /api/compare` — post-hoc model comparison (Mode A = model_dirs; Mode B = stats JSONs)
- ✅ `GET /api/models` — enumerate NAS-registered model directories
- ✅ Both endpoints have `/api/tasks/<name>` async task variants (FC async task mode)
- ❌ **No `anvil` training endpoint** — deferred to v0.0.2 or a separate `openadmet-trainer-server` (training runs hours + needs bigger inputs).

## Architecture

```
Client / Agent
  ↓ HTTP (multipart: input_smiles | input_csv | input_sdf + model_names[])
┌──────────────────────────────────────────────────────────────────────┐
│  FastAPI + bioq-service-framework  (port 9000)                   │
│                                                                      │
│  Service-specific                                                    │
│    POST /api/predict          POST /api/tasks/predict                │
│    POST /api/compare          POST /api/tasks/compare                │
│    GET  /api/models          (NAS-model registry)                    │
│                                                                      │
│  Framework provided                                                  │
│    GET  /api/manifest        GET  /openapi.json                      │
│    GET  /healthz             GET  /healthz/detail (w/ registry probe)│
│    GET  /api/jobs/{id}/...                                           │
└──────────────────────────────────────────────────────────────────────┘
  ↓ subprocess (GPU, single-card)
openadmet predict --input-path ... --input-col ... --model-dir ... --accelerator gpu
  ↓ output
NAS: <jobs_base_dir>/<job_id>/{input/, output/predictions.csv, logs/run.log, job.json}

NAS: /data/models/openadmet/
     ├── models/<name>/{model.pth, model.json, recipe_components/...}
     └── foundations/.chemprop/chemeleon_mp.pt  ← HOME override hits this
```

## API cheat-sheet

### `POST /api/predict`

Multipart form fields (see `models.py::PredictRequest` for full schema):

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `input_smiles` | str | one of 5 input sources | — | Comma/whitespace-separated, up to 200 mols |
| `input_csv` | file | ← | — | CSV with a recognized SMILES column |
| `input_csv_uri` | str | ← | — | `oss://` / `file://` / `http(s)://` / `job://` |
| `input_sdf` | file | ← | — | SDF; upstream reads natively |
| `input_sdf_uri` | str | ← | — | URI variant |
| `input_col` | str | — | *auto* | Auto-derived from first model's `data.yaml` |
| `model_names` | list[str] | ✓ | — | 1-20 names from `GET /api/models` |
| `accelerator` | enum | — | `gpu` | `cpu` / `gpu` / `auto` / `mps` / `tpu` / `ipu` |
| `aq_fxns` | list[enum] | — | `[]` | `ucb` / `ei` / `pi` (each at most once) |
| `beta`, `best_y`, `xi` | list[float] | — | `[]` | Aligned to aq_fxns |
| `debug` | bool | — | `false` | Upstream verbose logging |

**Inline SMILES against a single model**:

```bash
curl -X POST $URL/api/predict \
    -F 'input_smiles=CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl,CC(=O)OC1=CC=CC=C1C(=O)O' \
    -F 'model_names=herg-chemeleon-baseline'
```

**CSV upload against multiple models** (mixed input_col groups auto-handled):

```bash
curl -X POST $URL/api/predict \
    -F 'input_csv=@my_compounds.csv' \
    -F 'model_names=herg-chemeleon-baseline' \
    -F 'model_names=cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1'
```

**Large batch via OSS URI + UCB acquisition scoring**:

```bash
curl -X POST $URL/api/predict \
    -F 'input_csv_uri=oss://my-bucket/screening/lib.csv' \
    -F 'model_names=cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1' \
    -F 'aq_fxns=ucb' -F 'beta=1.5'
```

Output columns follow upstream:
- `OADMET_PRED_<tag>_<taskname>` — point predictions
- `OADMET_STD_<tag>_<taskname>` — ensemble std (NaN if not ensembled)
- `OADMET_UCB_<tag>_<taskname>` / `OADMET_EI_<tag>_<taskname>` / `OADMET_PI_<tag>_<taskname>` — acquisition scores (if requested)

### `POST /api/compare`

Two mutually exclusive modes:

**Mode A** — from NAS model_dirs:

```bash
curl -X POST $URL/api/compare \
    -F 'model_names=cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1' \
    -F 'model_names=cyp1a2-cyp2d6-cyp3a4-cyp3c9-chemeleon-baseline' \
    -F 'label_types=biotarget' -F 'label_types=biotarget' \
    -F 'mt_id=CYP3A4'
```

**Mode B** — from pre-computed stats JSON:

```bash
curl -X POST $URL/api/compare \
    -F 'model_stats_files=@stats_a.json' \
    -F 'model_stats_files=@stats_b.json' \
    -F 'labels=modelA' -F 'labels=modelB' \
    -F 'task_names=pchembl_value_mean' \
    -F 'task_names=pchembl_value_mean' \
    -F 'report=true'
```

Output: `comparison_stats.json` + `plots/*.png` + optional `report.pdf`.

### `GET /api/models`

```bash
curl $URL/api/models | jq .
```

Returns `{count: N, models: [{name, path, input_col, target_cols, biotargets, tag, model_type, feat_type, ...}]}`.

### Task-mode endpoints

FC async task mode — HTTP returns 202 immediately, FC binds request lifetime to compute:

```bash
curl -X POST $URL/api/tasks/predict \
    -H 'X-Fc-Invocation-Type: Async' \
    -H 'X-Bioagent-Job-Id: my-predict-2026-07-05-01' \
    -H 'X-Fc-Async-Task-Id: my-predict-2026-07-05-01' \
    -F 'input_smiles=CCO,c1ccccc1' \
    -F 'model_names=herg-chemeleon-baseline'
```

## Configuration

`pydantic-settings`, `OPENADMET_` prefix.

| Env var | Default | Notes |
|---|---|---|
| `OPENADMET_JOBS_BASE_DIR` | `/data/openadmet_jobs` | NAS job root |
| `OPENADMET_ROOT` | `/opt/openadmet/upstream` | Upstream source root (subprocess cwd) |
| `OPENADMET_PYTHON` | `/opt/conda/envs/openadmet-models/bin/python` | Conda env python |
| `OPENADMET_WEIGHTS_DIR` | `/data/models/openadmet` | NAS model + foundation cache root |
| `OPENADMET_MAX_CONCURRENT_JOBS` | `1` | GPU single-card; FC session affinity handles horizontal scale |
| `OPENADMET_OSS_REGION` | `cn-hangzhou` | OSS URI download region |
| `OPENADMET_SESSION_HEADER_NAME` | `bioagent-session-id` | FC session affinity header |
| `HOME` | `/data/models/openadmet/foundations` | **Set by Dockerfile** so upstream CheMeleon cache hits NAS |

## Weights & data (NAS-mounted, NOT baked)

Expected layout (produced by `scripts/fetch_weights.sh` + rsync to NAS):

```
/data/models/openadmet/
├── foundations/
│   └── .chemprop/
│       └── chemeleon_mp.pt                           ← Zenodo record 15460715
└── models/                                            ← 6 HF pre-staged models
    ├── herg-chemeleon-baseline/                      ← hERG binding pIC50
    ├── cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1/     ← Multitask CYP LogAC50 (v1)
    ├── cyp1a2-cyp2d6-cyp3a4-cyp3c9-chemeleon-baseline/  ← Multitask CYP (baseline)
    ├── microsomal-clearance-chemeleon-v1/            ← Liver microsomal CLint (H/R/M)
    ├── permeability-logd-ppb-chemeleon-baseline/     ← logD + Caco-2 + PPB
    └── pxr-chemeleon-baseline/                       ← PXR (CYP3A4 induction)
```

Each `models/<name>/` contains `{model.pth, model.json, recipe_components/{metadata,data,procedure,eval}.yaml, anvil_recipe.yaml, ...}` (flattened from HF's nested `<name>/anvil_training/`).

Total ~330 MB on NAS (models ~280 MB + foundation ~50 MB).

**Why `HOME` override?**  Upstream `openadmet/models/architecture/chemprop.py::_download_chemeleon` caches at `Path.home() / ".chemprop"`.  FC can't reach Zenodo reliably + `$HOME` is ephemeral per instance.  Dockerfile sets `HOME=/data/models/openadmet/foundations` so `Path.home() / ".chemprop" / "chemeleon_mp.pt"` resolves to the NAS-cached copy.

### Pre-stage (one-time)

```bash
# 1. Vendor upstream openadmet-models + 4 pip-git deps at pinned SHA
./services/openadmet-server/scripts/vendor.sh

# 2. Fetch weights: 6 HF models + CheMeleon foundation from Zenodo.
#    (Prefers opensource/openadmet-models/hc/ as source when present.)
./services/openadmet-server/scripts/fetch_weights.sh
# → services/openadmet-server/weights/

# 3. Upload to NAS
rsync -av services/openadmet-server/weights/ \
    <NAS-mount>:/data/models/openadmet/
```

Or fetch straight to NAS:

```bash
WEIGHTS_DST=/mnt/nas/data/models/openadmet \
    ./services/openadmet-server/scripts/fetch_weights.sh
```

### Verify FC deployment

```bash
curl https://fc-openadmet-XXX.cn-hangzhou-vpc.fcapp.run/healthz/detail
# expect: {"weights_loaded":true, "chemeleon_foundation_present":true,
#          "models_count": 6, "models_available":["herg-chemeleon-baseline", ...]}
```

### SIF / HPC (Apptainer)

```bash
apptainer run --nv \
    --bind /scratch/models/openadmet:/data/models/openadmet \
    openadmet-server.sif \
    python -m server predict \
        --params-json '{"input_smiles":"CCO","model_names":["herg-chemeleon-baseline"]}' \
        --output-dir results/
```

See [weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md).

## Docker build & run

```bash
# 1. Vendor upstream + pip-git deps (one-time; rerun to bump SHAs)
./services/openadmet-server/scripts/vendor.sh

# 2. Build (image ~4-6 GB, no baked weights)
make build-openadmet-server

# 3. Local run (mount pre-staged weights read-only + jobs r/w)
docker run -p 9000:9000 --gpus all \
    -v $(pwd)/services/openadmet-server/weights:/data/models/openadmet:ro \
    -v /tmp/openadmet_jobs:/data/openadmet_jobs \
    openadmet-server
```

Build context must be the project root (Dockerfile also `COPY services/_framework`).

## CLI batch mode

```bash
# Predict from an SDF
docker run --rm --gpus all \
    -v /data:/data \
    -v /path/to/models:/data/models/openadmet:ro \
    openadmet-server \
    /opt/conda/envs/openadmet-models/bin/python -m server predict \
        --input-sdf /data/inputs/library.sdf \
        --params-json '{"model_names":["herg-chemeleon-baseline"], "accelerator":"gpu"}' \
        --output-dir /data/results/

# Compare two models
docker run --rm \
    -v /data:/data \
    -v /path/to/models:/data/models/openadmet:ro \
    openadmet-server \
    /opt/conda/envs/openadmet-models/bin/python -m server compare \
        --params-json '{
            "model_names":["m1","m2"],
            "label_types":["biotarget","biotarget"],
            "mt_id":"CYP3A4"
        }' \
        --output-dir /data/compare/
```

Slurm sbatch template: see [apptainer-compatibility.md](../../engineering/guides/apptainer-compatibility.md).

## Offline tests

```bash
uv run python -m pytest services/openadmet-server/tests/test_app.py -v
uv run python -m pytest services/openadmet-server/tests/test_cli.py -v

# FC integration (post-deployment)
RUN_FC_TESTS=1 uv run python -m pytest -m fc \
    services/openadmet-server/tests/test_fc.py -v

RUN_FC_TESTS=1 uv run python -m pytest -m fc \
    services/openadmet-server/tests/test_fc_task.py -v
```

## Alibaba Cloud Function Compute deployment

| Config | Recommended |
|---|---|
| Function type | **GPU** (T4 / A10 24 GB) |
| CPU | 8 vCPU |
| Memory | 32 GB |
| GPU memory | ≥ 16 GB |
| Listen port | `9000` |
| Function timeout | 1800 s (predict) / 3600 s (compare + report) |
| NAS mount | `/fc → /data` (contains `models/openadmet/`) |
| Async task mode | **Enabled** |
| Session affinity | Enabled (`bioagent-session-id` header) |

## Related docs

- [openadmet-server design](../../engineering/decisions/2026-07-05-openadmet-server-design.md)
- [Service framework abstraction](../../engineering/decisions/2026-05-12-service-framework-design.md)
- [Service weights externalization](../../engineering/decisions/2026-06-26-service-weights-externalization.md)
- [Adding a new service (cookbook)](../../engineering/guides/adding-a-new-service.md)
- Upstream: [OpenADMET/openadmet-models](https://github.com/OpenADMET/openadmet-models) (MIT)
- CheMeleon foundation: [Zenodo record 15460715](https://zenodo.org/records/15460715)
- Pre-staged models: [huggingface.co/openadmet](https://huggingface.co/openadmet)
