# haddock3-server

HTTP + CLI service wrapping [HADDOCK3](https://github.com/haddocking/haddock3)
(BonvinLab, *JCIM* 2025) — an integrative biomolecular **docking** platform.
Built on [bioq-service-framework](../_framework/).

```
client ──HTTP──▶ FastAPI (server.app) ──▶ JobRunner ──▶ subprocess
                   │                                      │
                   │                        inference.py wrapper
                   │                                      │
                   │             haddock3 / haddock3-score / haddock3-restraints
                   │                                      │
                   ▼                                      ▼
             /api/manifest                    NAS: CNS engine (CNS_EXEC)
                                                    <job>/output/ results
```

Design doc: [engineering/decisions/2026-07-15-haddock3-server-design.md](../../engineering/decisions/2026-07-15-haddock3-server-design.md).

## Endpoints

Each has a matching `/api/tasks/<name>` (FC async task mode).

| Endpoint | CNS? | What |
|---|---|---|
| `POST /api/dock` | yes | General workflow runner: `molecules` PDBs + `config` (workflow body) + optional `tbl` |
| `POST /api/dock/protein-protein` | yes | Curated two-body docking (`mol1`,`mol2`,`ambig`,`reference` + knobs) |
| `POST /api/score` | yes | Standalone HADDOCK scoring of a `complex` PDB |
| `POST /api/restraints/restrain-bodies` | **no** | Body restraints `.tbl` from a multi-chain PDB |
| `POST /api/restraints/active-passive-to-ambig` | **no** | Ambig `.tbl` from two `.actpass` files |

```bash
# CNS-free restraints (works without a license)
curl -X POST $URL/api/restraints/restrain-bodies -F structure=@complex.pdb

# Curated docking (needs CNS staged on NAS)
curl -X POST $URL/api/dock/protein-protein \
  -F mol1=@receptor.pdb -F mol2=@ligand.pdb -F ambig=@ambig.tbl \
  -F sampling=200 -F top_models=4

# Standalone scoring (needs CNS)
curl -X POST $URL/api/score -F complex=@complex.pdb -F full=true

# General runner — config is the workflow body only (no run_dir/molecules/mode/ncores);
# reference uploads by bare filename.
curl -X POST $URL/api/dock \
  -F molecules=@a.pdb -F molecules=@b.pdb -F tbl=@air.tbl \
  -F 'config=[topoaa]

[rigidbody]
ambig_fname = "air.tbl"
sampling = 200

[caprieval]
'
```

Poll `GET /api/jobs/<id>` until `completed`, then `GET /api/jobs/<id>/files` /
`/api/jobs/<id>/download`. Learn the full protocol from `GET /api/manifest`.

## Configuration (`HADDOCK3_*` env)

| Env | Default | Notes |
|---|---|---|
| `HADDOCK3_JOBS_BASE_DIR` | `/data/haddock3_jobs` | Per-job dirs |
| `HADDOCK3_ROOT` | `/opt/haddock3` | Service root |
| `HADDOCK3_PYTHON` | `/opt/haddock3/.venv/bin/python` | venv interpreter |
| `HADDOCK3_INFERENCE_SCRIPT` | `/opt/haddock3/server/inference.py` | wrapper |
| `HADDOCK3_CNS_EXEC` / `CNS_EXEC` | `/data/models/haddock3/cns/cns` | externalised CNS binary |
| `HADDOCK3_WEIGHTS_DIR` | `/data/models/haddock3` | CNS parent (no NN weights) |
| `HADDOCK3_DEFAULT_NCORES` | `8` | `mode='local'` cores |
| `HADDOCK3_MAX_CONCURRENT_JOBS` | `2` | CPU concurrency |

## CNS (required for docking + scoring)

HADDOCK3's compute engine is [CNS](http://cns-online.org/v1.3/), which is
**license-gated** (academic users request it free; only education emails). The
repo ships only the HADDOCK *patches* (`upstream/varia/cns1.3/`), **not CNS
itself**. Restraints endpoints do not use CNS and work without it.

NAS layout the service expects:

```
/data/models/haddock3/
└── cns/
    └── cns          # your compiled + HADDOCK-patched CNS 1.3 executable
```

1. Obtain + compile CNS 1.3 patched with `upstream/varia/cns1.3/[bis]*` per the
   upstream guide (`upstream/docs/pages/CNS.md`). Requires gcc/gfortran/csh.
2. Stage it (no download step — license-gated):
   ```bash
   # local stage -> services/haddock3-server/weights/cns/cns
   CNS_SRC=/path/to/cns_solve-XXXX.exe ./scripts/stage_cns.sh
   # or straight to NAS
   CNS_SRC=/path/to/cns.exe WEIGHTS_DST=/mnt/nas/data/models/haddock3 ./scripts/stage_cns.sh
   ```
3. Verify after deploy: `curl $URL/healthz/detail` → `"cns_available": true`.
   The runtime image ships `libgfortran5` for the CNS fortran binary.

SIF / HPC: `apptainer run --bind /scratch/models/haddock3:/data/models/haddock3 ...`.

> Docking is **CPU-heavy** — an FC 8-vCPU instance is slow (see
> [dockq FC CPU throttling](../../engineering/guides/dockq-fc-cpu-throttling.md)).
> Prefer a large-vCPU instance / ECS, or the CLI batch mode on Slurm. Keep
> `sampling` small for smoke tests.

## Development

```bash
# offline unit tests (no haddock3 / CNS needed)
uv run python -m pytest services/haddock3-server/tests/test_app.py \
    services/haddock3-server/tests/test_cli.py \
    services/haddock3-server/tests/test_configs.py -v

# vendor upstream (once; re-run to bump the pinned SHA)
./services/haddock3-server/scripts/vendor.sh

# build
make build-haddock3-server

# CLI batch mode (Slurm / sbatch)
apptainer exec haddock3-server.sif python -m server dock-protein-protein \
    --mol1 a.pdb --mol2 b.pdb --ambig air.tbl --output-dir out/ --sampling 200

# FC integration tests (after deploy); restraints run without CNS,
# dock/score self-skip until CNS is staged.
RUN_FC_TESTS=1 uv run python -m pytest -m fc services/haddock3-server/tests/test_fc.py -v
RUN_FC_TESTS=1 uv run python -m pytest -m fc services/haddock3-server/tests/test_fc_task.py -v
```

## FC deployment

CPU function (no GPU). Enable async task mode; mount NAS `/data/models/haddock3`
(CNS) + `/data/haddock3_jobs`; for gateway use, mount the data-plane OSS bucket
at `/mnt/oss`. Add the `haddock3-server` entry to
[services/services.yaml](../services.yaml) after the first deploy.
