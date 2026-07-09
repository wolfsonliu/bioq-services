# reinvent-server

REINVENT4 (AstraZeneca / MolecularAI) de novo molecule generation, RL, TL, and
scoring wrapped as a dual-mode service. 5 structured endpoints — one per REINVENT
run mode — each with an FC async-task twin. The server builds a REINVENT TOML from
the request (`config_builder.py`), then runs the vendored `reinvent config.toml`
CLI; upstream source is **0-modified**.

**Dual-mode deployment:**
- **FC GPU async task mode** — quick `sampling` / `scoring` / `enumeration`
  (`/api/tasks/*`, enabled by default).
- **HPC SIF sbatch** — long-running `transfer-learning` / `staged-learning`, which
  support checkpoint chaining (a stage's `.chkpt` feeds the next run's `agent_file`).

设计文档：[engineering/decisions/2026-07-08-reinvent-server-design.md](../../engineering/decisions/2026-07-08-reinvent-server-design.md)

## Endpoints

Each sync endpoint below has an FC async twin at `/api/tasks/<name>`
(`task_endpoints_enabled=True` by default).

| Path | run_type | Key inputs | Output | Notes |
|---|---|---|---|---|
| `POST /api/sampling` | `sampling` | `generator`, `model_file`, `num_smiles`; opt. `smiles_file` | `sampling.csv` | Seed `smiles_file` needed for libinvent/linkinvent/mol2mol/pepinvent, not reinvent |
| `POST /api/scoring` | `scoring` | `scoring` (JSON), `smiles_file` (req) | `score_results.csv` | `scoring` is a JSON-encoded form field |
| `POST /api/enumeration` | `enumeration` | `scoring` (JSON), `peptide_smiles` (req), `amino_acid_library` (req) | `peptide_enumeration.csv` | Peptide enumeration (pepinvent) |
| `POST /api/transfer-learning` | `transfer_learning` | `smiles_file` (req); opt. `validation_smiles_file`, `input_model_upload` | `*.model` | Fine-tune a prior; long — prefer HPC |
| `POST /api/staged-learning` | `staged_learning` | `stages` (JSON list, req); opt. `smiles_file`/`prior_upload`/`agent_upload`, `diversity_filter`/`inception`/`learning_strategy` (JSON) | `*_1.csv` + `*.chkpt` | RL / curriculum; long — prefer HPC |

Complex fields (`scoring`, `stages`, `diversity_filter`, `inception`,
`learning_strategy`, `pairs`) arrive as JSON strings over multipart form and are
decoded server-side.

## Generators + prior registry

`generator` ∈ `reinvent` / `libinvent` / `linkinvent` / `mol2mol` / `pepinvent`.
`model_file` (sampling/TL) accepts a **registry dot-key** or a path relative to
`prior_base`. `None` → the generator's default prior.

| Dot-key | Prior file | Default for |
|---|---|---|
| `.reinvent` | `reinvent.prior` | `reinvent` |
| `.libinvent` | `libinvent.prior` | `libinvent` |
| `.linkinvent` | `linkinvent.prior` | `linkinvent` |
| `.m2m_high` | `mol2mol_high_similarity.prior` | |
| `.m2m_medium` | `mol2mol_medium_similarity.prior` | `mol2mol` |
| `.m2m_mmp` | `mol2mol_mmp.prior` | |
| `.m2m_scaffold` | `mol2mol_scaffold.prior` | |
| `.m2m_scaffold_generic` | `mol2mol_scaffold_generic.prior` | |
| `.pepinvent` | `pepinvent.prior` | `pepinvent` |

Priors are **not baked** into the image. The 9 files ship from Zenodo
(DOI 10.5281/zenodo.15641296) and load at runtime from NAS via
`REINVENT_PRIOR_BASE=/data/models/reinvent` (FC NAS mount / SIF `--bind`).

## Vendor + Build

```bash
# 1. Vendor upstream REINVENT4 at the pinned SHA (no network at build time).
./services/reinvent-server/scripts/vendor.sh

# 2. Fetch the 9 priors from Zenodo → stage dir (default services/.../weights/),
#    then rsync to NAS / HPC scratch. NOT baked into the image.
./services/reinvent-server/scripts/fetch_weights.sh
# Or straight to NAS:
WEIGHTS_DST=/mnt/nas/data/models/reinvent \
    ./services/reinvent-server/scripts/fetch_weights.sh

# 3. Build docker image (uv venv, 2-stage; torch 2.12 cu126 + reinvent[chemprop2] + iSIM).
docker build --platform linux/amd64 -t reinvent-server \
    -f services/reinvent-server/Dockerfile .
# Or via Makefile:  make build-reinvent-server

# 4. Convert to SIF for HPC (optional).
make sif-reinvent-server
```

## HTTP usage

```bash
URL=http://localhost:9000

# De novo sampling from the Reinvent prior (no seed).
curl -X POST $URL/api/sampling -F generator=reinvent -F num_smiles=100

# Scoring — scoring is a JSON-encoded form field.
curl -X POST $URL/api/scoring \
    -F 'scoring={"type":"geometric_mean","component":[]}' \
    -F smiles_file=@compounds.smi

# Staged learning (RL) — stages is a JSON-encoded list field.
curl -X POST $URL/api/staged-learning -F generator=reinvent \
    -F 'learning_strategy={"type":"dap","sigma":128,"rate":0.0001}' \
    -F 'stages=[{"chkpt_name":"s1.chkpt","max_score":0.6,"min_steps":25,"max_steps":100,"scoring":{"type":"geometric_mean","component":[]}}]'
```

Jobs return a `JobInfo`; poll `GET /api/jobs/<id>`. Under FC async mode, POST to
the `/api/tasks/<name>` twin with the `X-Fc-Async-Task-Id` header instead.

## CLI / HPC sbatch

CLI batch mode runs one endpoint synchronously (`python -m server <subcommand>`),
suited to Slurm. Subcommands: `sampling scoring enumeration transfer-learning
staged-learning`.

```bash
#!/bin/bash
#SBATCH --job-name=reinvent-rl
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

apptainer exec --nv \
    --bind /scratch/${USER}/models/reinvent:/data/models/reinvent \
    --bind /scratch/${USER}/reinvent_jobs:/data/reinvent_jobs \
    reinvent-server.sif \
    python -m server staged-learning \
        --params-json rl.json \
        --agent /scratch/${USER}/reinvent_jobs/prev/output/s1.chkpt \
        --output-dir /scratch/${USER}/reinvent_jobs/run2/output
```

`rl.json` is the `StagedLearningRequest` payload (including the `stages` list).
**Checkpoint chaining for long RL**: pass a previous run's stage `.chkpt` as
`--agent` (the run's `agent_file`) to resume from where the last stage left off;
`transfer-learning` similarly accepts `--input-model` for an explicit start point.

## Configuration (`REINVENT_` env prefix)

| Env var | Default | Purpose |
|---|---|---|
| `REINVENT_PRIOR_BASE` | `/data/models/reinvent` | Prior/model dir (NAS mount / SIF `--bind`); also read by upstream's prior_registry |
| `REINVENT_JOBS_BASE_DIR` | `/data/reinvent_jobs` | Per-job working / output root |
| `REINVENT_DEVICE` | `cuda:0` | Torch device; per-request `device` overrides |
| `REINVENT_MAX_CONCURRENT_JOBS` | `2` | Async runner concurrency cap |
| `REINVENT_TASK_ENDPOINTS_ENABLED` | `True` | Register `/api/tasks/*` FC async twins |

`GET /healthz/detail` reports `prior_base`, `priors_loaded` / `priors_missing`,
`cuda_available`, and active/max job counts.

## v0.0.1 known limits

1. **No OpenEye scoring components** — commercial license required; not integrated.
2. **chemprop2 is the only extra ML scoring backend** on top of REINVENT's
   built-ins (rdkit / pumas / mmpdb / apted / iSIM, etc.).
3. **Scoring config is passthrough** — the `scoring` / `stages[].scoring` JSON is
   forwarded verbatim into the TOML and validated only at REINVENT runtime, not by
   this service.
4. **Long RL / TL should use HPC, not FC** — FC async tasks have a max duration;
   `transfer-learning` / `staged-learning` on large datasets belong on sbatch.

## Tests

```bash
# Offline HTTP + CLI + config-builder suite (mocked subprocess).
uv run python -m pytest services/reinvent-server/tests/ -v
```
