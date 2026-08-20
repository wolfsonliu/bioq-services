# seqkit-server

[SeqKit](https://github.com/shenwei356/seqkit) (Shen & Zou 2016, *PLOS ONE*
11(10):e0163962) wrapped as a bioq service: deterministic FASTA/FASTQ stats and
reverse-complement over the shared service framework. CPU-only — a single
static Go binary, no GPU, no model weights. Design doc:
[`docs/specs/2026-08-20-seqkit-server-design.md`](../../docs/specs/2026-08-20-seqkit-server-design.md).

```
client / agent ──HTTP──▶ FastAPI + bioq framework ──subprocess──▶ seqkit (static binary)
                              │                                        │
                              └── /api/jobs/* lifecycle ◀── output/ ───┘
```

## Endpoints

| Endpoint | Mode | Purpose |
|---|---|---|
| `POST /api/stats` | submit/poll | `seqkit stats --tabular [--all]` → `output/stats.tsv` |
| `POST /api/tasks/stats` | task (blocking) | same, FC async task mode |
| `POST /api/revcomp` | submit/poll | `seqkit seq -r -p` → `output/revcomp.fasta` |
| `POST /api/tasks/revcomp` | task (blocking) | same, FC async task mode |

```bash
# stats — full column set
curl -X POST $URL/api/stats -F input_fasta=@reads.fasta -F all_stats=true

# revcomp — explicit DNA alphabet
curl -X POST $URL/api/revcomp -F input_fasta=@reads.fasta -F seq_type=dna

# URI inputs (job:// / oss:// / file:// / http(s)://) instead of an upload
curl -X POST $URL/api/stats -F input_fasta_uri=job://<job_id>/output/revcomp.fasta
```

CLI batch mode (same image, Slurm/sbatch):

```bash
docker run --rm -v /data:/data seqkit-server \
    .venv/bin/python -m server stats \
    --input-fasta /data/reads.fasta --output-dir /data/results/
```

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `SEQKIT_JOBS_BASE_DIR` | `/data/seqkit_jobs` | job-dir root |
| `SEQKIT_BIN` | `/opt/seqkit/bin/seqkit` | seqkit binary path |
| `SEQKIT_THREADS` | `4` | `-j/--threads` |
| `SEQKIT_MAX_CONCURRENT_JOBS` | `2` | concurrent jobs per instance |
| `SEQKIT_OSS_OUTPUT_MOUNT` | `/mnt/oss` | gateway output-sink mount |

## Weights

**None.** SeqKit is a static binary with no model weights, so there is no NAS
weights externalization and no `weights_dir`. `/healthz/detail` reports whether
the binary exists and runs (`ready`) instead.

## Local development

```bash
./scripts/vendor.sh                          # pin + verify the seqkit binary
cd services/seqkit-server
uv run --group dev python -m pytest tests/test_app.py tests/test_cli.py -q
```

## Docker build

From the repo root:

```bash
make build-seqkit-server        # Makefile auto-discovers the Dockerfile
docker run --rm seqkit-server .venv/bin/python -m server --help
```

## FC deployment

- CPU instance (no GPU); small spec suffices (2 vCPU / 4 GB class).
- Enable 异步任务模式 in the console (task endpoints are the `bioq run` entry).
- No NAS weights mount needed; mount the data-plane OSS bucket at `/mnt/oss`
  when called through the gateway (`oss_mount: true` in `services.yaml`).
