# mmseqs2-server

FastAPI wrapper around the GPU-enabled `mmseqs` binary, exposing the
[ColabFold MSA HTTP protocol](https://github.com/sokrypton/ColabFold/blob/main/colabfold/mmseqs/search.py)
so any client that talks to `https://api.colabfold.com` — notably
[`boltz-server`](../boltz-server/)'s `--msa_server_url` — can be redirected to
an internal endpoint. Built on
[bioagent-service-framework](../_framework/).

```
client (boltz, ...)  ──▶  FastAPI (this service)
                              │
                              ├─ POST /ticket/msa                 ColabFold protocol — single-chain MSA
                              ├─ POST /ticket/pair                ColabFold protocol — multi-chain pairing
                              ├─ GET  /ticket/<id>                ColabFold protocol — status
                              ├─ GET  /result/download/<id>       ColabFold protocol — tar.gz stream
                              ├─ POST /api/tasks/msa              FC async task — single-chain MSA (blocking)
                              ├─ POST /api/tasks/pair             FC async task — multi-chain pairing (blocking)
                              └─ GET  /api/jobs/<id>              framework lifecycle (JobInfo, files, log, ...)
                                     │
                                     ▼
                              vendored ColabFold orchestrator
                                     │
                                     ▼
                              mmseqs (GPU) × 15+ subprocess steps
                                     │
                                     ▼
                              NAS: /data/mmseqs2_jobs/<job_id>/output/
```

Same image also runs in **CLI batch mode** for sbatch / one-shot use:

```bash
apptainer exec --nv mmseqs2-server.sif \
    .venv/bin/python -m server msa \
        --input-fasta /data/query.fasta \
        --mode env \
        --output-dir /scratch/$SLURM_JOB_ID/

apptainer exec --nv mmseqs2-server.sif \
    .venv/bin/python -m server pair \
        --input-fasta /data/multimer.fasta \
        --mode pairgreedy \
        --output-dir /scratch/$SLURM_JOB_ID/
```

The two HTTP paths and the CLI batch mode share the same
`colabfold_search_argv` builder — see
[`engineering/decisions/2026-05-29-cli-batch-mode.md`](../../engineering/decisions/2026-05-29-cli-batch-mode.md)
for the dual-mode rationale and
[`engineering/decisions/2026-06-17-fc-async-task-mode.md`](../../engineering/decisions/2026-06-17-fc-async-task-mode.md)
for why the `/api/tasks/*` surface exists alongside `/ticket/*`.

## HTTP usage

ColabFold protocol (submit + poll + download) — backwards-compatible with any
client that targets `api.colabfold.com`:

```bash
# Submit
curl -X POST "$URL/ticket/msa" \
    -F 'q=>q1\nMKQHKAMIVALIVICITAVVAAL...' \
    -F mode=env
# → {"id": "<job_id>", "status": "PENDING"}

# Poll
curl "$URL/ticket/<job_id>"
# → {"id": "<job_id>", "status": "COMPLETE"}

# Download
curl -o msa.tar.gz "$URL/result/download/<job_id>"
```

FC async task mode (one request = one atomic task; FC manages queueing /
dedup at the platform layer):

```bash
TASK_ID="mmseqs-$(date +%s)"
curl -X POST "$URL/api/tasks/msa" \
    -H "X-Fc-Invocation-Type: Async" \
    -H "X-Bioagent-Job-Id: $TASK_ID" \
    -H "X-Fc-Async-Task-Id: $TASK_ID" \
    -F 'q=>q1\nMKQHKAMIVALIVICITAVVAAL...' \
    -F mode=env
# → HTTP 202 (function runs synchronously inside the FC instance)

# Poll JobInfo via the framework's lifecycle endpoints (or, in
# production, FCDispatcher.get_status — HTTP polling is rate-limit-prone).
curl "$URL/api/jobs/$TASK_ID"
```

Errors on `/ticket/*` follow the ColabFold convention (HTTP 200 + `{"status":
"ERROR"}` — the upstream client treats 4xx/5xx as fatal). Errors on
`/api/tasks/*` use standard HTTP 4xx/5xx because the callers there speak
JobInfo, not the ColabFold protocol.

## Build

Prerequisite: vendor the MMseqs2 GPU binary tarball on the host (once; re-run
to bump the pinned release):

```bash
./services/mmseqs2-server/scripts/vendor.sh
# → services/mmseqs2-server/upstream/mmseqs-linux-gpu.tar.gz  (gitignored)
```

Then:

```bash
make build-mmseqs2-server
# or:
docker build --platform linux/amd64 -t mmseqs2-server \
    -f services/mmseqs2-server/Dockerfile .
```

`docker build` no longer touches the network for the mmseqs binary — the
`vendor.sh` step retries GitHub Releases with backoff, so transient CN
connection resets are absorbed there instead of failing a 3-GB image build.
To pin a different release: `MMSEQS_VERSION=<tag> ./scripts/vendor.sh`. Only
tags that publish a `mmseqs-linux-gpu.tar.gz` asset are valid (`18-8cc5c` is
the current default; earlier tags like `15-6f452` do NOT have a GPU artifact).

## FC deployment

| Env var | Default | Notes |
|---|---|---|
| `MMSEQS2_JOBS_BASE_DIR` | `/data/mmseqs2_jobs` | NAS path for job state + output a3m / templates |
| `MMSEQS2_DB_DIR` | `/data/models/mmseqs2` | NAS path containing the pre-built ColabFold DBs (follows the `/data/models/<svc>/` externalization convention) |
| `MMSEQS2_DEFAULT_DB` | `uniref30_subset_4090_gpu` | UniRef30 GPU DB name (relative to `MMSEQS2_DB_DIR`) |
| `MMSEQS2_ENV_DB` | `colabfold_envdb_gpu` | ColabFoldDB env DB; unset to disable env mode |
| `MMSEQS2_GPU_ENABLED` | `true` | Force CPU fallback by setting to `false` |
| `MMSEQS2_THREADS` | `4` | CPU threads per mmseqs invocation |
| `MMSEQS2_SESSION_HEADER_NAME` | `bioagent-session-id` | FC session affinity (set via FC console too) |
| `MMSEQS2_MMSEQS_BINARY` | `/opt/mmseqs-gpu/bin/mmseqs` | Override only if the install path changes |

Framework env vars (`SERVICE_DISK_LIMIT_MB`, `SERVICE_ERROR_TAIL_CHARS`, ...)
behave as documented in `services/_framework/README.md`.

DB preparation runs offline on an HPC node; see
[`scripts/prepare_databases.sh`](scripts/) (added in a later stage).

## Related

- [Design](../../engineering/decisions/2026-06-23-mmseqs2-server-design.md)
- [Implementation plan](../../engineering/decisions/2026-06-23-mmseqs2-server-plan.md)
- [ColabFold MSA pipeline reference (Mirdita et al. 2022)](https://www.nature.com/articles/s41592-022-01488-1)
- [adding-a-new-service guide](../../engineering/guides/adding-a-new-service.md)
