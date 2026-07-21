# dockq-server

FastAPI wrapper for [DockQ](https://github.com/bjornwallner/DockQ) — protein /
NA / small-molecule docking quality scoring. Built on
[bioq-service-framework](../_framework/).

```
client ──▶ FastAPI (this service)
              │
              ├─ /api/score        single (model, native) pair
              └─ /api/score_batch  1 native + N candidate models
                     │
                     ▼
              DockQ subprocess(es) (CPU-only, ~1–10 s per pair)
                     │
                     ▼
              NAS: /data/dockq_jobs/<job_id>/output/
```

CPU-only, image ~350 MB (no CUDA). Designed for ranking RFantibody / RFdiffusion
output against a known reference complex.

## Endpoints

### `POST /api/score`

Single (model, native) DockQ scoring. Returns the raw DockQ JSON as
`output/<name>.json`.

```bash
curl -X POST $URL/api/score \
  -F model=@design.pdb \
  -F native=@reference.pdb \
  -F mapping=HLA:BCX \
  -F name=ab_ag
# → {"job_id":"...","status":"pending"}
# poll /api/jobs/<job_id> until "completed", then download:
curl $URL/api/jobs/<job_id>/file/ab_ag.json
```

### `POST /api/score_batch`

Score N candidate models against 1 reference native. Per-model JSON +
`scores.csv` sorted summary.

```bash
curl -X POST $URL/api/score_batch \
  -F native=@reference.pdb \
  -F models=@design_001.pdb \
  -F models=@design_002.pdb \
  -F models=@design_003.pdb \
  -F sort_by=DockQ
# output/scores.csv          → sorted summary
# output/per_model/<m>.json  → raw DockQ JSON per model
# output/failed.csv          → models that errored (if any)
```

Common parameters (both endpoints):

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | str | `run` | output basename |
| `mapping` | str | — | DockQ `--mapping MODELCHAINS:NATIVECHAINS`, e.g. `HLA:BCX` or `:HL` |
| `small_molecule` | bool | `false` | required for HEM / cofactor / ligand inputs |
| `capri_peptide` | bool | `false` | peptide-protein mode (DockQ flags it as unreliable) |
| `no_align` | bool | `false` | trust residue numbering directly (skip sequence alignment) |
| `allowed_mismatches` | int | 0 | aligned-mismatch tolerance |
| `optDockQF1` | bool | `false` | optimize chain mapping for DockQ_F1 |
| `n_cpu` | int | — | override `DOCKQ_DEFAULT_N_CPU` for this call |

Batch-only: `sort_by` (default `DockQ`) — column name in `scores.csv` for descending sort.

### Input URIs

Both endpoints accept any of `model_uri` / `native_uri` (and per-file in
single mode) instead of multipart upload:

- `job://<id>/<file>` — re-use an output from a prior job on the same NAS
- `file:///abs/path` — direct NAS path (cross-service shared mount)
- `oss://<bucket>/<key>` — Alibaba Cloud OSS (needs `OSS_ACCESS_KEY_ID/_SECRET`)
- `http(s)://...` — generic URL, including OSS pre-signed URLs

## Configuration

All env-driven via `pydantic-settings`; `DOCKQ_` prefix.

| Env var | Default | Notes |
|---|---|---|
| `DOCKQ_JOBS_BASE_DIR` | `/data/dockq_jobs` | NAS path for job state + outputs |
| `DOCKQ_ROOT` | `/opt/dockq` | DockQ source tree (informational; binary resolved via PATH) |
| `DOCKQ_BINARY` | `DockQ` | Absolute or PATH-resolved path to the DockQ entrypoint |
| `DOCKQ_DEFAULT_N_CPU` | `4` | DockQ `--n_cpu` default; per-call `n_cpu` overrides |
| `DOCKQ_MAX_BATCH_SIZE` | `200` | Hard cap on N models per `/api/score_batch` |
| `DOCKQ_OSS_REGION` | `cn-hangzhou` | for `oss://` URIs |

Framework env vars (`SERVICE_DISK_LIMIT_MB`, `SERVICE_ERROR_TAIL_CHARS`, ...)
behave as documented in `services/_framework/README.md`.

## Local development

```bash
# Set up the editable install
cd services/dockq-server
uv venv .venv --python python3.11
uv pip install --python .venv/bin/python "numpy<2.0" biopython networkx parallelbar cython
uv pip install --python .venv/bin/python -e ../../opensource/DockQ
uv pip install --python .venv/bin/python "../_framework[mcp]" httpx alibabacloud-oss-v2 pytest

# Run the offline tests (no real DockQ subprocess)
DOCKQ_BINARY=/bin/true pytest tests/

# Start the server
DOCKQ_JOBS_BASE_DIR=/tmp/dockq-jobs .venv/bin/python -m uvicorn server.app:app \
    --host 0.0.0.0 --port 9000

# Sanity-check manifest in another shell
curl -s localhost:9000/api/manifest | jq .endpoints
```

## Docker build

```bash
make build-dockq-server                       # local image
make push-dockq-server                        # build + tag + push to harbor
make push-dockq-server TAG=v0.0.2             # override tag
```

Image base: `ghcr.io/astral-sh/uv:python3.11-bookworm-slim` (uv pre-installed,
glibc — important for numpy/biopython manylinux wheels). Final image ~350 MB.

## FC deployment

1. `make push-dockq-server` (writes `harbor.ruosheng.bio/aliyun_fc/dockq-server:vX.Y.Z`)
2. Update the FC console: image tag → new `vX.Y.Z`
3. Register the URL in [`services/aliyun_fc_url.md`](../aliyun_fc_url.md):
   ```
   dockq-server: https://fc-dockq-XXXXXXXXXX.cn-hangzhou.fcapp.run
   ```
4. Verify:
   ```bash
   pytest -m fc services/dockq-server/tests/test_fc.py
   ```

CPU-only service — no GPU instance class needed; 4 vCPU / 4 GB memory is
plenty for batches up to ~100 candidates.

## Related

- [`opensource/DockQ`](../../opensource/DockQ/) — upstream source
- [adding-a-new-service guide](../../engineering/guides/adding-a-new-service.md)
- [testing-fc-services guide](../../engineering/guides/testing-fc-services.md)
- [DockQ v2 paper](https://academic.oup.com/bioinformatics/article/40/10/btae586/7796530) — Mirabello & Wallner 2024
