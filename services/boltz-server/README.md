# boltz-server

FastAPI wrapper for [Boltz-2](https://github.com/jwohlwend/boltz) —
AlphaFold3-class biomolecular foundation model for complex structure +
binding-affinity prediction. Built on
[bioagent-service-framework](../_framework/).

```
client ──▶ FastAPI (this service)
              │
              ├─ /api/predict_structure   complex 3D structure (no affinity)
              └─ /api/predict_affinity    structure + log10 IC50 + binder probability
                     │
                     ▼
              boltz predict <yaml> --model boltz2 (single GPU)
                     │
                     ▼
              NAS: /data/boltz_jobs/<job_id>/output/predictions/input/
```

Boltz-2 only (Boltz-1 intentionally out of scope; see
[design doc](../../engineering/decisions/2026-05-19-boltz-server-design.md)).
Image ~11.1 GB with weights + `mols/` CCD data baked in.

## Endpoints

### `POST /api/predict_structure`

Predict a complex's 3D structure. Accepts a structured `sequences` /
`constraints` / `templates` description, or a raw Boltz YAML escape hatch.

```bash
curl -X POST $URL/api/predict_structure \
  -F name=demo \
  -F msa_mode=auto \
  -F 'sequences=[{"type":"protein","id":"A","sequence":"MVTPEGN..."}]'
# → {"job_id":"...","status":"pending"}
# poll /api/jobs/<job_id> until "completed", then download:
curl $URL/api/jobs/<job_id>/file/predictions/input/input_model_0.cif
```

### `POST /api/predict_affinity`

Same plus a binding-affinity head for one ligand chain.

```bash
curl -X POST $URL/api/predict_affinity \
  -F name=binding \
  -F binder_id=B \
  -F msa_mode=auto \
  -F 'sequences=[
        {"type":"protein","id":"A","sequence":"MVTPEGN..."},
        {"type":"ligand","id":"B","smiles":"N[C@@H](Cc1ccc(O)cc1)C(=O)O"}
      ]'
# output/predictions/input/affinity_input.json contains:
#   affinity_pred_value         — log10 IC50 (lower = stronger binding)
#   affinity_probability_binary — binder vs decoy probability (0-1)
```

Common parameters (both endpoints):

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | str | `run` | output basename |
| `sequences` | JSON list | — | required (unless `raw_yaml` given); see SequenceEntry below |
| `constraints` | JSON list | `[]` | bond / pocket / contact constraints |
| `templates` | JSON list | `[]` | structural templates (CIF or PDB) |
| `raw_yaml` / `raw_yaml_uri` | str / URI | — | escape hatch: full Boltz YAML, mutually exclusive with `sequences` |
| `msa_mode` | enum | `auto` | `auto` / `provided` / `empty` |
| `msa_server_url` | str | `https://api.colabfold.com` | only used in `auto` mode |
| `msa_pairing_strategy` | enum | `greedy` | `greedy` / `complete` |
| `seed` | int | — | reproducibility |
| `recycling_steps` | int | `3` | AF3-style: `10` |
| `sampling_steps` | int | `200` | |
| `diffusion_samples` | int | `1` | AF3-style: `25` (much slower) |
| `step_scale` | float | — | Boltz-2 default 1.5; lower → higher diversity |
| `output_format` | enum | `mmcif` | `mmcif` / `pdb` |
| `use_potentials` | bool | `false` | physical-plausibility potentials (~1.5× cost) |
| `write_full_pae` | bool | `false` | dump pae npz per sample |
| `write_full_pde` | bool | `false` | dump pde npz per sample |
| `no_kernels` | bool | `false` | disable cuequivariance (older GPUs) |

Affinity-only fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `binder_id` | str | — | required; must match a `type=ligand` SequenceEntry id |
| `affinity_mw_correction` | bool | `false` | molecular-weight correction on affinity head |
| `sampling_steps_affinity` | int | `200` | |
| `diffusion_samples_affinity` | int | `5` | ensemble size for affinity prediction |

### SequenceEntry schema

```json
{
  "type": "protein|dna|rna|ligand",
  "id": "A" | ["A", "B"],
  "sequence": "MKT...",                 // protein/dna/rna
  "smiles": "CCO",                      // ligand (xor ccd)
  "ccd": "ATP",                         // ligand (xor smiles)
  "msa_uri": "empty" | "<file>.a3m" | "job://..." | "file://..." | "oss://...",
  "cyclic": false,
  "modifications": [{"position": 5, "ccd": "SEP"}]
}
```

Per-chain MSA control: each protein `SequenceEntry.msa_uri` overrides
`msa_mode` for that chain. `"empty"` skips MSA; a bare filename matches a
multipart-uploaded `msa_files` entry by stem (e.g. `msa_uri: "A.a3m"` matches
`-F msa_files=@A.a3m`). URI schemes are resolved before YAML construction.

### Input URIs

Both endpoints accept these schemes wherever a URI is allowed (`msa_uri`,
`cif_uri` / `pdb_uri` on TemplateEntry, `raw_yaml_uri`):

- `job://<id>/<file>` — re-use an output from a prior job on the same NAS
- `file:///abs/path` — direct NAS path (cross-service shared mount)
- `oss://<bucket>/<key>` — Alibaba Cloud OSS (needs `OSS_ACCESS_KEY_ID/_SECRET`)
- `http(s)://...` — generic URL, including OSS pre-signed URLs

Multipart upload alternatives: `msa_files` (one per chain, filename stem =
chain id), `template_files` (one per template), `raw_yaml_upload`.

## Configuration

All env-driven via `pydantic-settings`; `BOLTZ_` prefix.

| Env var | Default | Notes |
|---|---|---|
| `BOLTZ_JOBS_BASE_DIR` | `/data/boltz_jobs` | NAS path for job state + outputs |
| `BOLTZ_ROOT` | `/opt/boltz` | Boltz install root (subprocess cwd) |
| `BOLTZ_BINARY` | `/opt/boltz/.venv/bin/boltz` | `boltz` CLI entrypoint (click group) |
| `BOLTZ_CACHE_DIR` | `/opt/boltz/weights` | Pre-staged weights + `mols/` CCD data (copied from `opensource/boltz/weights/`) |
| `BOLTZ_MAX_CONCURRENT_JOBS` | `1` | single-GPU FC instances → serial |
| `BOLTZ_OSS_REGION` | `cn-hangzhou` | for `oss://` URIs |

Framework env vars (`SERVICE_DISK_LIMIT_MB`, `SERVICE_ERROR_TAIL_CHARS`, ...)
behave as documented in `services/_framework/README.md`.

## Local development

```bash
cd services/boltz-server
uv venv .venv --python python3.10
uv pip install --python .venv/bin/python "torch>=2.2" "numpy<2.0"
uv pip install --python .venv/bin/python -e ../../opensource/boltz".[cuda]"
uv pip install --python .venv/bin/python "../_framework[mcp]" httpx alibabacloud-oss-v2 pyyaml pytest

# Offline tests (no GPU, no real `boltz` binary needed — stubbed)
BOLTZ_BINARY=/bin/true BOLTZ_JOBS_BASE_DIR=/tmp/boltz-jobs \
    .venv/bin/python -m pytest tests/

# Start the server (needs a GPU + pre-downloaded weights)
BOLTZ_CACHE_DIR=$HOME/.boltz BOLTZ_JOBS_BASE_DIR=/tmp/boltz-jobs \
    .venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 9000

# Sanity-check manifest in another shell
curl -s localhost:9000/api/manifest | jq .endpoints
```

To pre-download weights for local dev (one-time, ~6 GB):

```bash
.venv/bin/python -c "from boltz.main import download_boltz2; \
    from pathlib import Path; download_boltz2(Path.home() / '.boltz')"
rm $HOME/.boltz/mols.tar  # save 1.5 GB after extraction
```

## Docker build

```bash
make build-boltz-server                       # local image
make push-boltz-server                        # build + tag + push to harbor
make push-boltz-server TAG=v0.0.2             # override tag
```

Image base: `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`. Modern PyTorch
ships its CUDA / cuDNN runtime libs as PyPI `nvidia-*` sub-packages, so
`nvidia/cuda` base images would just duplicate those libs without speedup —
slim base is ~3 GB smaller. Final image ~11.1 GB (code stack ~5 GB + Boltz-2
weights/cache ~6 GB).

The Dockerfile assumes Boltz-2 weights are pre-staged at
`opensource/boltz/weights/` (`boltz2_conf.ckpt`, `boltz2_aff.ckpt`, extracted
`mols/`) and copies them in as the runtime cache — no network download during
`docker build`. To pre-stage, run `download_boltz2` once into that directory:

```bash
.venv/bin/python -c "from pathlib import Path; \
    from boltz.main import download_boltz2; \
    download_boltz2(Path('opensource/boltz/weights'))"
rm -f opensource/boltz/weights/mols.tar  # keep image small; extracted mols/ is enough
```

The Dockerfile `COPY`s each Boltz-2 file explicitly
(`boltz2_conf.ckpt`, `boltz2_aff.ckpt`, `mols/`) instead of the whole
`weights/` tree, so Boltz-1 artifacts (`boltz1_conf.ckpt`, `ccd.pkl`) never
enter the image even if you ran `download_boltz1` into the same directory.
The Boltz-2 codepath loads CCD via `mols/` (`load_canonicals()`), not
`ccd.pkl`.

`BOLTZ_CACHE_DIR=/opt/boltz/weights` at runtime, so `boltz predict --cache`
points at the baked-in weights. If `mols.tar` is missing the build touches an
empty marker so `download_boltz2` (called by every `boltz predict`) skips its
existence check and doesn't re-download 1.5 GB on first request.

A build-time `import torch, cuequivariance_torch, boltz` smoke check fails
fast if PyPI wheel versions drift apart.

## FC deployment

1. `make push-boltz-server` (writes `harbor.ruosheng.bio/aliyun_fc/boltz-server:vX.Y.Z`)
2. Update the FC console: image tag → new `vX.Y.Z`
3. Recommended resources:

   | 项 | 推荐值 |
   |---|---|
   | GPU 配置 | `fc.gpu.ada.1` (24 GB) — fits most ≤ 1000-residue complexes |
   | 内存 | `32768` MB |
   | 磁盘 | `20480` MB |
   | 超时 | `3600` s (large complexes with AF3-style settings can hit 30+ min) |
   | NAS | `/fc → /data` (shared with other bioagent services) |
   | 外网 | required if any caller uses `msa_mode=auto` (ColabFold MSA server) |

4. Register the URL in [`services/aliyun_fc_url.md`](../aliyun_fc_url.md):
   ```
   boltz-server: https://fc-boltz-XXXXXXXXXX.cn-hangzhou.fcapp.run
   ```
5. Verify:
   ```bash
   pytest -m fc services/boltz-server/tests/test_fc.py
   ```

## Troubleshooting

- **`CUDA error: forward compatibility was attempted on non-supported HW`** —
  the FC instance's NVIDIA driver is older than PyTorch's bundled CUDA
  expects. Roll back to `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` as the
  base image (see decisions doc for the alternate Dockerfile).
- **MSA server times out** — the ColabFold server can be unreachable from
  inside FC. Switch the request to `msa_mode=provided` with a pre-computed
  `.a3m` per protein chain.
- **`cuequivariance` errors on older GPUs** — pass `no_kernels: true` in the
  request to forward `--no_kernels` to the boltz CLI.

## Related

- [`opensource/boltz`](../../opensource/boltz/) — upstream source
- [adding-a-new-service guide](../../engineering/guides/adding-a-new-service.md)
- [testing-fc-services guide](../../engineering/guides/testing-fc-services.md)
- [boltz-server design](../../engineering/decisions/2026-05-19-boltz-server-design.md)
- [Boltz-2 paper (Passaro et al. 2025)](https://doi.org/10.1101/2025.06.14.659707)
