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
                              ├─ POST /ticket/msa                 single-chain MSA
                              ├─ POST /ticket/pair                multi-chain pairing
                              ├─ GET  /ticket/<id>                status
                              └─ GET  /result/download/<id>       tar.gz stream
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

Status: scaffold only — see
[design doc](../../engineering/decisions/2026-06-23-mmseqs2-server-design.md)
and
[implementation plan](../../engineering/decisions/2026-06-23-mmseqs2-server-plan.md)
for the full protocol surface and the Stage 1-4 task breakdown.

## FC deployment

| Env var | Default | Notes |
|---|---|---|
| `MMSEQS2_JOBS_BASE_DIR` | `/data/mmseqs2_jobs` | NAS path for job state + output a3m / templates |
| `MMSEQS2_DATA_DIR` | `/data/mmseqs2` | NAS path containing the pre-built DBs |
| `MMSEQS2_DEFAULT_DB` | `uniref30_subset_4090_gpu` | UniRef30 GPU DB name (relative to `MMSEQS2_DATA_DIR`) |
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
