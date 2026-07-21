# diamond-server

FastAPI + CLI wrapper around **DIAMOND** v2.2.1
([bbuchfink/diamond](https://github.com/bbuchfink/diamond)) — a fast **CPU**
protein / translated-DNA aligner and protein clusterer. Design rationale:
[engineering/decisions/2026-07-14-diamond-server-design.md](../../engineering/decisions/2026-07-14-diamond-server-design.md).

DIAMOND is **CPU-only** (no GPU). For GPU MSA use
[mmseqs2-server](../mmseqs2-server/).

## Endpoints

HTTP (each has an `/api/tasks/<name>` FC async-task-mode variant):

| Endpoint | Input | Output |
|---|---|---|
| `POST /api/blastp` | `query` FASTA + (`db_uri` .dmnd \| `subject` FASTA) | `output/<name>.tsv` |
| `POST /api/blastx` | `query` DNA FASTA + (`db_uri` \| `subject`) | `output/<name>.tsv` |
| `POST /api/cluster` | `sequences` FASTA | `output/<name>.clusters.tsv` |
| `POST /api/msa` | `query` FASTA + (`db_uri` \| default `DIAMOND_MSA_DB`) | `output/<name>.a3m` |

Framework endpoints (`/healthz`, `/healthz/detail`, `/api/jobs/*`,
`/api/manifest`, `/openapi.json`) come from `bioq_service.create_app`.

`makedb` is **CLI/SIF-only** — building a database is an offline step; the HTTP
service consumes prebuilt `.dmnd` files (or builds a tiny one inline from an
uploaded `subject`).

### Input references

- `blastp` / `blastx`: provide **exactly one** of `db_uri` (a prebuilt `.dmnd`,
  via `job://` / `file://` / `oss://` / `http(s)://`) **or** a `subject`
  protein FASTA (uploaded or `subject_uri`), which is built into a `.dmnd`
  inline.
- `msa`: provide `db_uri` (a `.dmnd`) or configure `DIAMOND_MSA_DB` (a filename
  under `DIAMOND_DB_DIR` on NAS).

## Configuration (`DIAMOND_` env prefix)

| Env | Default | Meaning |
|---|---|---|
| `DIAMOND_JOBS_BASE_DIR` | `/data/diamond_jobs` | Per-job working dirs (NAS) |
| `DIAMOND_BINARY` | `/usr/local/bin/diamond` | DIAMOND binary path |
| `DIAMOND_DB_DIR` | `/data/models/diamond` | Reference `.dmnd` dir (NAS mount) |
| `DIAMOND_MSA_DB` | (unset) | Default `/api/msa` reference DB (relative to `DIAMOND_DB_DIR`) |
| `DIAMOND_THREADS` | `8` | CPU threads per invocation (`-p`) |
| `DIAMOND_DEFAULT_SENSITIVITY` | (unset) | Sensitivity when a request omits it |
| `DIAMOND_MAX_CONCURRENT_JOBS` | `2` | Concurrent job cap |

## Build & deploy

```bash
# 1. Vendor the prebuilt binary (0-network docker build afterwards)
./services/diamond-server/scripts/vendor.sh

# 2. Build the image (from project root)
docker build --platform linux/amd64 -t diamond-server -f services/diamond-server/Dockerfile .
```

Deploy target: FC **CPU** instance (e.g. 8 vCPU / 16–32 GB). Mount NAS at
`/data/models/diamond` (reference DBs) and `/data/diamond_jobs`; mount the
data-plane OSS bucket at `/mnt/oss` for gateway output-sink downloads.

## CLI / SIF (Slurm sbatch)

The same image runs every command as a one-shot batch job (no FastAPI):

```bash
# Build a database (CLI-only)
apptainer exec diamond-server.sif .venv/bin/python -m server makedb \
    --sequences ref.faa --name ref --output-dir /scratch/$SLURM_JOB_ID/

# Protein search against a prebuilt DB
apptainer exec diamond-server.sif .venv/bin/python -m server blastp \
    --query q.faa --db /scratch/.../output/ref.dmnd \
    --sensitivity very-sensitive --output-dir /scratch/$SLURM_JOB_ID/

# Or build the DB inline from a subject FASTA
apptainer exec diamond-server.sif .venv/bin/python -m server blastp \
    --query q.faa --subject subj.faa --output-dir out/

# Cluster / MSA
apptainer exec diamond-server.sif .venv/bin/python -m server cluster \
    --sequences lib.faa --algorithm cluster --approx-id 90 --output-dir out/
apptainer exec diamond-server.sif .venv/bin/python -m server msa \
    --query q.faa --db /data/models/diamond/uniref50.dmnd --output-dir out/
```

Bind NAS into the container when needed:
`apptainer exec --bind /scratch/models/diamond:/data/models/diamond ...`.

## Tests

```bash
# Offline (no DIAMOND binary needed)
uv run python -m pytest services/diamond-server/tests/test_app.py \
    services/diamond-server/tests/test_cli.py \
    services/diamond-server/tests/test_tools.py -v

# FC integration (opt-in; needs a deployed service)
RUN_FC_TESTS=1 uv run python -m pytest -m fc services/diamond-server/tests/test_fc.py -v
```

## Notes / limitations

- `/api/msa` produces a **fast, coarse** homolog a3m (blastp hits → query-anchored
  alignment); depth scales with `max_target_seqs`. It is not a substitute for
  ColabFold/HHblits-grade MSAs (use mmseqs2-server when sensitivity matters).
- Building large reference `.dmnd` databases (tens of GB) is an offline ops step
  (`makedb` on HPC), not part of this service.
- **Not in v0.0.1**: makedb over HTTP; realign/recluster/reassign/view/getseq;
  blastn (DNA-DNA); taxonomy output; DAA output; paired multimer MSA.
