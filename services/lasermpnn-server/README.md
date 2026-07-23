# lasermpnn-server

HTTP + CLI wrapper around [LASErMPNN](https://github.com/polizzilab/LASErMPNN)
(Polizzi Lab, *Nature* 2026) — **small-molecule-conditioned protein sequence
design**. Given a protein structure with a bound ligand (small molecule,
cofactor, metal), LASErMPNN redesigns the protein sequence and packs side chains
*conditioned on the ligand environment*, with a separate temperature knob for the
ligand first shell. It is the sibling of LigandMPNN, trained on **protonated**
structures.

```
client ── HTTP ─▶ FastAPI (server.app) ── bioq-service-framework ──▶ subprocess
                    /api/design                                       python -m LASErMPNN.run_batch_inference
                    /api/design_ligandmpnn                            python -m LASErMPNN.run_batch_inference_ligandmpnn
                    /api/tasks/*  (FC async)                              │
                                                                         ▼
                                              NAS: /data/models/lasermpnn (weights)
                                                   /data/lasermpnn_jobs/<job_id>/output/
```

> ⚠️ **Input ligands must carry hydrogens** in the correct protonation state.
> LASErMPNN was trained on protonated structures; unprotonated ligands give
> unexpected results. This service does **not** protonate inputs — see the NISE
> repo's `protonate_and_add_conect_records.py`.

## Endpoints

| Endpoint | Upstream script | Purpose |
|---|---|---|
| `POST /api/design` | `run_batch_inference` | LASErMPNN batch design; `model_variant` picks one of 3 checkpoints |
| `POST /api/design_ligandmpnn` | `run_batch_inference_ligandmpnn` | Retrained LigandMPNN variant (paper comparison) |
| `POST /api/tasks/design`, `/api/tasks/design_ligandmpnn` | — | Same, as FC async tasks |

Job lifecycle (`/healthz`, `/api/jobs/*`, `/api/manifest`, `/openapi.json`) comes
from the framework. `GET /healthz/detail` reports NAS weight presence.

### Examples

```bash
# Ligand-conditioned design, 5 designs, output FASTA + PDBs
curl -X POST $URL/api/design \
    -F pdb=@4jnj-1_prot.pdb \
    -F designs_per_input=5 \
    -F sequence_temp=0.3 \
    -F model_variant=nothing_heldout

# Conservative binding site: lower first-shell temperature + ALA/GLY budget
curl -X POST $URL/api/design \
    -F pdb=@complex.pdb \
    -F designs_per_input=10 \
    -F sequence_temp=0.5 -F first_shell_sequence_temp=0.1 \
    -F constrain_ala_gly=true -F ala_budget=4

# Retrained LigandMPNN variant
curl -X POST $URL/api/design_ligandmpnn \
    -F pdb=@4jnj-1_prot.pdb -F designs_per_input=5
```

Output lands in `<job_id>/output/**/design_<i>.pdb` (redesigned sequence + packed
side chains) plus `designs.fasta` (headers carry a log10-probability score) when
`output_fasta=true`.

## Configuration

All env vars use the `LASERMPNN_` prefix (pydantic-settings; no `os.getenv`).

| Env var | Default | Description |
|---|---|---|
| `LASERMPNN_JOBS_BASE_DIR` | `/data/lasermpnn_jobs` | NAS job root |
| `LASERMPNN_ROOT` | `/opt/lasermpnn` | subprocess cwd (parent of the `LASErMPNN` package) |
| `LASERMPNN_WEIGHTS_DIR` | `/data/models/lasermpnn` | NAS weights mount |
| `LASERMPNN_DEVICE` | `cuda:0` | PyTorch device (`cpu` for GPU-less smoke tests) |
| `LASERMPNN_MAX_CONCURRENT_JOBS` | `1` | Single-GPU serial |
| `LASERMPNN_OSS_OUTPUT_MOUNT` | `/mnt/oss` | gateway output-sink mount |

## Weights

Not baked into the image — mounted from NAS at `/data/models/lasermpnn/`:

```
/data/models/lasermpnn/
├── laser_weights_0p1A_nothing_heldout.pt                  # default (model_variant=nothing_heldout)
├── laser_weights_0p1A_noise_ligandmpnn_split.pt           # model_variant=ligandmpnn_split + /api/design_ligandmpnn
├── soluble_weights_no_heldout_drop_clusters_optstep_65000.pt  # model_variant=soluble
└── pretrained_ligand_encoder_weights.pt                   # ligand encoder
```

Pre-stage from the upstream repo (weights are committed there, ~260 MB total):

```bash
# stage locally, then rsync to NAS:
./services/lasermpnn-server/scripts/fetch_weights.sh
rsync -a services/lasermpnn-server/weights/ <nas>:/data/models/lasermpnn/

# or download straight to NAS:
WEIGHTS_DST=/mnt/nas/data/models/lasermpnn \
    ./services/lasermpnn-server/scripts/fetch_weights.sh
```

FC: mount the NAS dir at `/data/models/lasermpnn` and verify:

```bash
curl -s $URL/healthz/detail | jq '{weights_loaded, weights_missing}'
```

SIF / HPC: `apptainer run --nv --bind /scratch/models/lasermpnn:/data/models/lasermpnn ...`.

The small `files/*.pt` reference tensors (ideal coords / bond geometry) ride
along with the vendored upstream tree — no separate staging needed.

## Local development

```bash
cd services/lasermpnn-server
uv run --group dev python -m pytest tests/test_app.py tests/test_cli.py -q
uvx ruff check .
```

## Docker build

```bash
# 1. vendor upstream at the pinned SHA (host-side; no in-build git clone)
./services/lasermpnn-server/scripts/vendor.sh
# 2. (optional, for local SIF) stage weights
./services/lasermpnn-server/scripts/fetch_weights.sh
# 3. build
make build-lasermpnn-server
# or:
docker build --platform linux/amd64 -t lasermpnn-server \
    -f services/lasermpnn-server/Dockerfile .

# CPU smoke (no GPU): override device
docker run --rm -e LASERMPNN_DEVICE=cpu lasermpnn-server \
    .venv/bin/python -m server --help
```

### CLI batch mode (Slurm sbatch)

```bash
apptainer exec --nv lasermpnn-server.sif \
    .venv/bin/python -m server design \
    --pdb /data/complex.pdb \
    --designs-per-input 5 --sequence-temp 0.3 \
    --output-dir /scratch/$SLURM_JOB_ID/
```

## FC deployment

GPU instance; enable async task mode; clear the keepalive URL; mount NAS
(`/data/models/lasermpnn`, `/data/lasermpnn_jobs`) and the OSS bucket at
`/mnt/oss`. See [engineering/guides/adding-a-new-service/deploy.md](../../engineering/guides/adding-a-new-service/deploy.md).
