# plip-server

HTTP + CLI wrapper around [PLIP](https://github.com/pharmai/plip) (Protein-Ligand
Interaction Profiler, GPL-2.0) — rule-based detection of non-covalent
protein-ligand (and protein-protein / nucleic-acid) interactions.

```
client ──HTTP──▶ FastAPI (server.app) ──▶ SubprocessRunner ──▶ python -m plip.plipcmd
                     │                                              │
             bioagent_service framework                    output/<name>.xml + .txt
             (jobs / manifest / MCP)                       (+ optional .pse / .png)
```

PLIP is **CPU-only** and **rule-based** — no GPU, no model weights. Given one PDB
complex (protein + ligand already present), it reports the 8 interaction types it
detects (hydrophobic, hydrogen bond, water bridge, π-stacking, π-cation, salt
bridge, halogen bond, metal complex). It does **not** score, predict affinity, or
prepare structures. See
[design doc](../../engineering/decisions/2026-07-15-plip-server-design.md).

## Endpoints

| Endpoint | Description |
|---|---|
| `POST /api/profile` | Profile interactions in one PDB complex → XML/TXT (+ optional PyMOL `.pse` / `.png`). |
| `POST /api/tasks/profile` | Same, FC async task mode (blocks until done). |
| `GET /healthz/detail` | Extended health: upstream source + openbabel/pymol importability + PLIP version. |
| `python -m server profile ...` | CLI batch mode (SIF / sbatch). |

### `POST /api/profile`

```bash
# Default ligand-interaction profile (XML + TXT)
curl -X POST $URL/api/profile \
  -F input_pdb=@complex.pdb \
  -F name=complex

# Protein-peptide (inter-chain) mode + PyMOL session
curl -X POST $URL/api/profile \
  -F input_pdb=@complex.pdb \
  -F mode=peptide -F peptide_chains=I \
  -F pymol_session=true -F name=complex

# Re-use a docked pose from a prior job (job:// / oss:// / file:// / http(s)://)
curl -X POST $URL/api/profile \
  -F input_pdb_uri=job://<dock_job_id>/output/pose_1.pdb \
  -F name=pose1
```

Request fields: `mode` (`default`/`peptide`/`intra`/`dnareceptor`),
`peptide_chains`, `intra_chain`, `report_formats` (`["xml","txt"]`),
`pymol_session`, `render_images`, `name`, `breakcomposite`, `altlocation`,
`nofix`, `keepmod`, `nohydro`, `model`, `maxthreads`. See `GET /api/manifest` for
the full schema.

Outputs land in `output/`: `<name>.xml` (structured report — primary
machine-readable output), `<name>.txt` (human-readable), `*.pse` / `*.png`
(when visualization enabled), `<name>_protonated.pdb` (PLIP's fixed structure).

## Configuration

All env vars use the `PLIP_` prefix (pydantic-settings, no `os.getenv`).

| Env var | Default | Description |
|---|---|---|
| `PLIP_JOBS_BASE_DIR` | `/data/plip_jobs` | Job root (NAS on FC). |
| `PLIP_UPSTREAM_DIR` | `/opt/plip/upstream` | Vendored PLIP source root (PYTHONPATH). |
| `PLIP_PYTHON` | `/opt/plip/.venv/bin/python` | Interpreter used to launch `plip.plipcmd`. |
| `PLIP_THREADS` | `4` | Default `--maxthreads`. |
| `PLIP_MAX_CONCURRENT_JOBS` | `2` | Concurrent job cap (CPU tool). |
| `PLIP_PYTHONPATH` | (derived) | Override PYTHONPATH prefix injected into the subprocess. |

## Weights

**None.** PLIP is a rule-based tool with no model weights, so there is no NAS
weights externalization and no `weights_dir`. `/healthz/detail` reports whether
the vendored upstream source is present and whether `openbabel` (required) and
`pymol` (optional, only for `-y`/`-p`) import successfully.

## Development

```bash
# Offline unit tests (no real PLIP needed — PLIP_PYTHON=/bin/true)
uv run python -m pytest services/plip-server/tests/test_app.py -v
uv run python -m pytest services/plip-server/tests/test_cli.py -v

# Vendor upstream source (required before docker build)
./services/plip-server/scripts/vendor.sh
ls services/plip-server/upstream/plip | head

# Build (from project root)
make build-plip-server
# or:
docker build --platform linux/amd64 -t plip-server -f services/plip-server/Dockerfile .

# Manifest / task-route sanity
docker run --rm -p 9000:9000 plip-server &
curl http://localhost:9000/api/manifest | jq .endpoints
curl http://localhost:9000/openapi.json | jq '.paths | keys | .[]' | grep /api/tasks/

# FC integration (after deploy)
pytest -m fc services/plip-server/tests/test_fc.py
pytest -m fc services/plip-server/tests/test_fc_task.py
```

### CLI / SIF batch mode

```bash
apptainer exec plip-server.sif \
  /opt/plip/.venv/bin/python -m server profile \
  --input-pdb /data/complex.pdb --name complex \
  --output-dir /scratch/$SLURM_JOB_ID/
```

## Deployment (FC)

- CPU instance (2–4 vCPU / 4–8 GB); no GPU.
- Enable **async task mode** in the console + clear the keepalive URL.
- Mount NAS at `PLIP_JOBS_BASE_DIR` (no weights mount needed).
- For gateway access: mount the data-plane OSS bucket `bioagent-inputs` at
  `/mnt/oss` (RW); `PLIP_OSS_OUTPUT_MOUNT=/mnt/oss` is set in the Dockerfile.
  Large PDBs should be passed via `input_pdb_uri` (`oss://...`), not multipart,
  to stay under the 128 KiB async payload cap.
