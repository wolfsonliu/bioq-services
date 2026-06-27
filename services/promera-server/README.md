# promera-server

FastAPI wrapper for [Promera](https://github.com/bjing2016/promera) —
pairformer + EDM-diffusion model for protein complex structure prediction
(cofold) and de novo binder design (minibinder / VHH). Built on
[bioagent-service-framework](../_framework/).

```
client ──▶ FastAPI (this service)
              │
              ├─ /api/cofold              complex 3D structure prediction
              ├─ /api/design              de novo binder design
              ├─ /api/tasks/cofold        same cofold, FC async task mode
              └─ /api/tasks/design        same design, FC async task mode
                     │
                     ▼
              python -m promera ... (single GPU)
                     │
                     ▼
              NAS: /data/promera_jobs/<job_id>/output/
                       cofold/cofold_seed<i>_samp<j>.cif      (cofold)
                       target/sample<b>/backbone.cif          (design)
```

Two operational modes for each endpoint:

- **submit/poll** (`/api/cofold`, `/api/design`) — returns immediately with
  a `job_id`; client polls `/api/jobs/<job_id>` until terminal status.
  Subprocess runs in the FastAPI process's `ThreadPoolExecutor`.
- **async task** (`/api/tasks/cofold`, `/api/tasks/design`) — designed for
  FC Async Task Mode (`X-Fc-Invocation-Type: Async`). HTTP request returns
  202 immediately but the server handler blocks for the entire subprocess
  lifetime so the FC instance won't recycle mid-run. JobInfo is persisted
  to NAS at each state transition; client polls `/api/jobs/<task_id>` the
  same as the submit/poll mode. See
  [decision doc](../../engineering/decisions/2026-06-17-fc-async-task-mode.md).

## Endpoints

### `POST /api/cofold` (and `POST /api/tasks/cofold`)

Predict the 3D structure of a complex described by a tinyprot JSON schema.

```bash
curl -X POST $URL/api/cofold \
  -F input_schema=@ubiquitin.json \
  -F num_seeds=1 \
  -F diffusion_samples=5
# → {"job_id":"...","status":"pending"}
# poll /api/jobs/<job_id> until "completed", then download:
curl $URL/api/jobs/<job_id>/file/cofold/cofold_seed0_samp0.cif
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `input_schema` / `input_schema_uri` | upload / URI | — | tinyprot JSON schema (see below) |
| `num_seeds` | int | `1` | independent diffusion seeds |
| `diffusion_samples` | int | `5` | samples per seed (1-25) |
| `diffusion_steps` | int | `200` | EDM denoising steps (10-1000) |
| `recycling_steps` | int | `4` | pairformer recycling (1-20) |
| `save_trajectory` | bool | `false` | dump `*_traj.cif` per sample |
| `save_full_confidence` | bool | `false` | dump `*_conf.npz` (full matrices) |
| `save_distogram` | bool | `false` | dump `*_distogram.npy` |

Outputs (under `output/cofold/`):

- `cofold_seed<i>_samp<j>.cif` — predicted structure (mmCIF)
- `cofold_seed<i>_samp<j>_conf.json` — confidence summary (plddt, ptm,
  iptm, chain scores)
- Optional: `*_conf.npz`, `*_distogram.npy`, `*_traj.cif`

The output sub-directory name is the stem of the input JSON filename;
the server saves uploads as `cofold.json` so outputs land in
`output/cofold/`.

### `POST /api/design` (and `POST /api/tasks/design`)

De novo binder design against a target schema.

```bash
curl -X POST $URL/api/design \
  -F target_schema=@IL7Ra.json \
  -F design_type=minibinder \
  -F num_backbones=10 \
  -F binder_length_min=60 \
  -F binder_length_max=80 \
  -F inverse_folder_type=solublempnn
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `target_schema` / `target_schema_uri` | upload / URI | — | tinyprot JSON describing the target |
| `target_template` | upload | — | optional mmCIF template for structural conditioning |
| `design_type` | enum | `minibinder` | `minibinder` / `vhh` |
| `num_backbones` | int | `10` | diffusion backbones (1-10000) |
| `diffusion_steps` | int | `200` | EDM denoising steps |
| `recycling_steps` | int | `4` | |
| `binder_chain` | str | `B` | chain id for the designed binder |
| `binder_length_min` / `_max` | int / int | `40` / `120` | minibinder backbone length range |
| `epitope_chain` | str | `A` | target chain id |
| `epitope_residues` | str | `""` | comma-separated 1-indexed positions, e.g. `"32,36,37"` |
| `target_chains` | str | `""` | restrict to subset of target chains |
| `inverse_folder_type` | enum | `solublempnn` | `proteinmpnn` / `solublempnn` / `ligandmpnn` / `none` |
| `inverse_folder_num_seqs` | int | `1` | sequences per backbone (1-100) |
| `target_template_chain` | str | `A` | chain id within `target_template` |
| `target_template_subsample_frac` | float | `1.0` | fraction of resolved positions to keep |
| `save_full_confidence` | bool | `false` | |

VHH mode (`design_type=vhh`): caplacizumab framework with variable CDR
lengths (H1: 5-7, H2: 7-12, H3: 9-15).

Outputs (under `output/target/sample<b>/`):

- `backbone.cif` — diffusion backbone
- `sample<b>_design<j>.fasta` — redesigned binder sequence(s)
- `refolds/sample<b>_design<j>_refold<r>.cif` — refolded structures
- `refolds/*_confidence.json` — iptm, scRMSD, scDockQ per refold

### Input schema (tinyprot JSON)

Each top-level key is a chain id; value has `type` and `sequence`:

```json
{
  "A": {"type": "protein", "sequence": "MQIFVKTLT..."},
  "B": {"type": "ligand", "smiles": "N[C@@H]..."}
}
```

### Input URIs

Wherever a URI is allowed (`input_schema_uri`, `target_schema_uri`,
plus URI variants on common endpoints):

- `job://<id>/<file>` — reuse output from a prior job on the same NAS
- `file:///abs/path` — direct NAS path (shared mount)
- `oss://<bucket>/<key>` — Alibaba Cloud OSS (`OSS_ACCESS_KEY_ID/_SECRET`)
- `http(s)://...` — generic URL, including OSS pre-signed URLs

## Configuration

`pydantic-settings` with `PROMERA_` prefix.

| Env var | Default | Notes |
|---|---|---|
| `PROMERA_JOBS_BASE_DIR` | `/data/promera_jobs` | NAS path for job state + outputs |
| `PROMERA_ROOT` | `/opt/promera` | Promera install root (subprocess cwd) |
| `PROMERA_PYTHON` | `/opt/promera/.venv/bin/python` | venv interpreter |
| `PROMERA_WEIGHTS` | `/data/models/promera/promera/promera_2606.ckpt` | Promera ckpt (v0.0.4 起 NAS 挂载，外置) |
| `PROMERA_LIGANDMPNN_DIR` | `/opt/promera/LigandMPNN` | LigandMPNN 源码 + 权重 symlink |
| `PROMERA_TEMPLATES_DIR` | `/opt/promera/promera_src/examples/templates` | upstream-provided templates |
| `PROMERA_MAX_CONCURRENT_JOBS` | `1` | single-GPU FC instances → serial |
| `PROMERA_OSS_REGION` | `cn-hangzhou` | for `oss://` URIs |
| `TINYPROT_CACHE` | `/data/models/promera/tinyprot` | LMDB caches on NAS (v0.0.4 起外置) |

Framework env vars (`SERVICE_DISK_LIMIT_MB`, `SERVICE_ERROR_TAIL_CHARS`,
`SERVICE_TASK_ENDPOINTS_ENABLED`, ...) behave as documented in
`services/_framework/README.md`.

## Local development

```bash
cd services/promera-server
uv venv .venv --python python3.12
uv pip install --python .venv/bin/python torch numpy
uv pip install --python .venv/bin/python ../../opensource/tinyprot
uv pip install --python .venv/bin/python -e ../../opensource/promera
uv pip install --python .venv/bin/python "../_framework[mcp]" httpx alibabacloud-oss-v2 pyyaml pytest

# Offline tests (no GPU; subprocess stubbed via /bin/true)
PROMERA_PYTHON=/bin/true PROMERA_JOBS_BASE_DIR=/tmp/promera-jobs \
    .venv/bin/python -m pytest tests/test_app.py tests/test_cli.py -v

# Start the server (needs GPU + pre-staged weights + tinyprot LMDB)
PROMERA_JOBS_BASE_DIR=/tmp/promera-jobs \
    .venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 9000
```

## Docker build

```bash
make build SERVICE=promera-server                       # local image
make push SERVICE=promera-server                        # build + tag + push to harbor
make push SERVICE=promera-server TAG=v0.0.3             # override tag
```

Image base: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`. Final image
includes:

- Python 3.12 + uv venv with torch 2.9 / triton 3.5 (CUDA 12 wheels)
- Promera source (editable install) + LigandMPNN runtime + tinyprot
- Promera checkpoint (`promera_2606.ckpt`)
- LigandMPNN model params (`*_v_48_020.pt`, `ligandmpnn_v_32_010_25.pt`, ...)
- **tinyprot LMDB caches** (`ccd.lmdb/`, `taxonomy.lmdb/`) — required at
  module import time by `tinyprot.msa`
- **`build-essential`** — triton's NVIDIA backend JITs a driver-glue `.so`
  on first kernel run via host `gcc`; runtime-flavor CUDA bases don't ship
  it. See [promera-tinyprot-cache.md](../../engineering/guides/promera-tinyprot-cache.md).

## Weights

v0.0.4 起 3 个权重源（~5 GB 总）**不再 baked 到镜像**，从 NAS 加载。
LigandMPNN model_params 路径在镜像里是 symlink → NAS。

期望布局：

```
/data/models/promera/
├── promera/
│   └── promera_2606.ckpt           ← ~2 GB
├── ligandmpnn/                     ← LigandMPNN model_params
│   └── *.pt                        ← 6 small files
└── tinyprot/                       ← TINYPROT_CACHE
    ├── ccd.lmdb/data.mdb
    └── taxonomy.lmdb/data.mdb
```

⚠️ `tinyprot.msa` 在 **import** 时打开 `taxonomy.lmdb`，缺权重时服务会在
启动阶段 crash 而不是在第一次推理时——务必确保 NAS 已挂。

### Pre-staging（一次性）

```bash
# 默认下载到本地 stage 目录
./services/promera-server/scripts/fetch_weights.sh
# → services/promera-server/weights/{promera,ligandmpnn,tinyprot}/

# 上传到 NAS
rsync -av services/promera-server/weights/ \
    <NAS-mount>:/data/models/promera/
```

或直接下到 NAS：

```bash
WEIGHTS_DST=/mnt/nas/data/models/promera \
    ./services/promera-server/scripts/fetch_weights.sh
```

### FC

NAS 自动挂载到 `/data/models/promera/`。验证：

```bash
curl https://fc-promera-adkrlhmlcq.cn-hangzhou-vpc.fcapp.run/healthz/detail
# 期望：{"status":"ok","weights_loaded":true,"weights_missing":{}}
```

### SIF / HPC

```bash
apptainer run --nv \
    --bind /scratch/models/promera:/data/models/promera \
    promera-server.sif python -m server cofold ...
```

详见 [weights externalization design](../../engineering/decisions/2026-06-26-service-weights-externalization.md)。

## FC deployment

1. `make push SERVICE=promera-server` (writes
   `harbor.ruosheng.bio/aliyun_fc/promera-server:vX.Y.Z`)
2. Update the FC console: image tag → new `vX.Y.Z`
3. **Enable async task mode** in the FC console (required for
   `/api/tasks/*` endpoints to return HTTP 202 rather than executing
   synchronously)
4. Recommended resources:

   | 项 | 推荐值 |
   |---|---|
   | GPU 配置 | `fc.gpu.ada.1` (24 GB) for typical complexes |
   | 内存 | `32768` MB |
   | 磁盘 | `20480` MB |
   | 超时 | `3600` s (design can exceed 30 min with `num_backbones>10`) |
   | NAS | `/fc → /data` (shared with other bioagent services) |
   | 外网 | optional (only needed if pre-staged caches need refresh) |

5. Register the URL in [`services/aliyun_fc_url.md`](../aliyun_fc_url.md):

   ```
   promera-server: https://fc-promera-XXXXXXXXXX.cn-hangzhou-vpc.fcapp.run
   ```

6. Verify deployment (see Testing below).

## Testing

Three layers:

| File | Mode | What it covers |
|---|---|---|
| `tests/test_app.py` | offline | request models, argv builders, FastAPI route handlers (subprocess stubbed) |
| `tests/test_cli.py` | offline | CLI batch-mode entrypoints (`python -m server cofold/design`) |
| `tests/test_fc.py` | FC (opt-in) | submit/poll mode against deployed `/api/cofold` + `/api/design` |
| `tests/test_fc_task.py` | FC (opt-in) | async task mode against `/api/tasks/cofold` + `/api/tasks/design` — submit returns 202, JobInfo lifecycle, identity propagation, FC platform-layer dedup |

### Offline (no GPU, ~3 s)

```bash
uv run python -m pytest \
    services/promera-server/tests/test_app.py \
    services/promera-server/tests/test_cli.py -v
```

### FC end-to-end (opt-in, ~5-30 min)

```bash
RUN_FC_TESTS=1 uv run python -m pytest -m fc \
    services/promera-server/tests/test_fc.py \
    services/promera-server/tests/test_fc_task.py -v -s
```

`test_fc_task.py` covers async task mode in 7 sections (22 tests):

1. **TestAsyncSubmit** — both `/api/tasks/*` return HTTP 202 + OpenAPI registration
2. **TestAsyncCofoldRunsToCompletion** — JobInfo lifecycle, duration, outputs, params echo
3. **TestAsyncCofoldOutputs** — files listing has `.cif`, single-file download, zip download
4. **TestAsyncDesignRunsToCompletion** — same for design
5. **TestAsyncDesignOutputs** — `backbone.cif` present + downloadable
6. **TestAsyncTaskIdentity** — `X-Bioagent-Job-Id` ↔ `JobInfo.job_id` propagation
7. **TestAsyncDuplicateDedup** — same `X-Fc-Async-Task-Id` resubmit → 409 (FC platform-layer dedup) or 202 + framework-layer dedup

### Inspecting real outputs locally

Both `test_fc.py` and `test_fc_task.py` download the JobInfo, subprocess
log, raw zip, and extracted output files to a per-session timestamped
directory, **before** the completion assertion (so artifacts survive
failed subprocesses):

```
services/promera-server/tests/fc_outputs/run-<UTC-timestamp>/
├── cofold/
│   ├── jobinfo.json            ← full JobInfo (incl. error_summary on failure)
│   ├── subprocess.log          ← /api/jobs/<id>/log content
│   ├── <task_id>.zip           ← raw output bundle
│   └── extracted/              ← unzipped tree
│       └── cofold/
│           ├── cofold_seed0_samp0.cif
│           └── cofold_seed0_samp0_conf.json
└── design/
    └── ...
```

The directory is gitignored (`services/*/tests/fc_outputs/`). Use it to:

- Visually inspect predicted CIFs in PyMOL / ChimeraX
- Diff confidence scores across runs
- Forensically debug a failed run (`jobinfo.json` has full
  `error_summary` + `error_tail`)

## Troubleshooting

- **`Taxonomy database not found at /root/.cache/tinyprot/taxonomy.lmdb`** —
  the tinyprot LMDB caches weren't COPYed into the image. Run
  `scripts/fetch_weights.sh` to populate `services/promera-server/weights/tinyprot/`
  and rebuild. See [promera-tinyprot-cache.md](../../engineering/guides/promera-tinyprot-cache.md).
- **`Failed to find C compiler. Please specify via CC environment variable`** —
  base image is `cuda:*-runtime` which lacks host gcc; triton can't compile
  its NVIDIA driver launcher. Add `build-essential` to the Dockerfile's
  apt-get line. cofold doesn't trigger this; design does.
- **Async submit returns 200 instead of 202** — FC console hasn't enabled
  async task mode for this function. Toggle it on or use the submit/poll
  endpoints (`/api/cofold`, `/api/design`) instead.
- **`/api/jobs/<id>` returns 404 right after async submit** — transient FC
  routing race; the test helpers retry with backoff. In production, prefer
  `FCDispatcher.get_status` (FC GetAsyncTask API) over raw HTTP polling
  (see [project memory on FC HTTP polling](../../engineering/guides/)).
- **Same `X-Fc-Async-Task-Id` returns 409** — FC platform-layer dedup
  rejects the duplicate before the function is invoked. Treat 409 as
  "already accepted, poll the existing task" — see decision doc.
- **`cofold` outputs landed in `output/input/`** — old layout from before
  v0.0.2; the schema was saved as `input.json` so promera derived `name =
  "input"`. v0.0.2+ saves as `cofold.json` → `output/cofold/`.

## Related

- [`opensource/promera`](../../opensource/promera/) — upstream source
- [`opensource/tinyprot`](../../opensource/tinyprot/) — featurization + MSA library
- [adding-a-new-service guide](../../engineering/guides/adding-a-new-service.md)
- [testing-fc-services guide](../../engineering/guides/testing-fc-services.md)
- [FC async task mode decision](../../engineering/decisions/2026-06-17-fc-async-task-mode.md)
- [promera tinyprot/triton runtime traps](../../engineering/guides/promera-tinyprot-cache.md)
