# lightdock-server

HTTP + CLI service wrapping [LightDock](https://github.com/lightdock/lightdock)
— a **protein-protein / protein-peptide / protein-DNA** docking framework based
on Glowworm Swarm Optimization (GSO). CPU-only, no NN weights.

Built on `bioq-service-framework`. Design rationale:
[engineering/decisions/2026-07-15-lightdock-server-design.md](../../engineering/decisions/2026-07-15-lightdock-server-design.md).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/dock` | Full LightDock docking protocol; submit/poll job |
| POST | `/api/tasks/dock` | Same, as an FC async task (blocks until done) |
| GET | `/healthz` / `/healthz/detail` | Liveness + LightDock version / scoring functions |
| GET | `/api/manifest` / `/openapi.json` | Service description |
| GET | `/api/jobs/{id}` … `/download` | Job lifecycle / result retrieval |

`/api/dock` runs the full pipeline internally:
`setup → GSO run → conformation generation → per-swarm clustering → ranking → top-N`.

### Inputs

- `receptor` (upload) or `receptor_uri` — receptor PDB (**required**)
- `ligand` (upload) or `ligand_uri` — ligand PDB (**required**)
- `restraints` (upload) or `restraints_uri` — optional LightDock restraints file

### Key parameters

| Field | Default | Notes |
|---|---|---|
| `swarms` | 0 (auto) | Number of GSO swarms. **0 = auto from surface (can be hundreds).** |
| `glowworms` | 200 | Poses per swarm |
| `steps` | 100 | GSO steps |
| `scoring_function` | `fastdfire` | `dna` for protein-DNA; see `/healthz/detail` for the list |
| `cores` | 8 | Multiprocessing cores |
| `top` | 10 | Ranked complexes to generate |
| `use_anm` | false | ANM backbone flexibility |
| `cluster_cutoff` | 4.0 | Per-swarm BSAS RMSD cutoff (Å) |
| `noxt` / `noh` / `now` | false | Strip OXT / H / water during setup |

### Outputs

Under `output/`:
- `top/top_1.pdb` … `top_N.pdb` — ranked docked complexes (best score first)
- `rank_by_scoring.list` — global ranking table
- `setup.json` — resolved LightDock config

## Runtime & sizing (read before running)

GSO sampling cost scales with **swarms × glowworms × steps**. The upstream
production preset (400 swarms × 200 glowworms × 100 steps) takes **hours** and
is **not** suited to an interactive FC call.

- **Interactive / FC**: set a small `swarms` (10–40) and supply `restraints` to
  focus sampling near the known interface. Keep `steps` ≤ 100.
- **Production / large-scale**: run via the CLI batch mode on Slurm/ECS with
  full sampling.

## CLI batch mode (Slurm / sbatch / SIF)

Same image, synchronous one-shot execution:

```bash
python -m server dock \
    --receptor receptor.pdb --ligand ligand.pdb \
    --output-dir results/ \
    --swarms 20 --glowworms 100 --steps 50 --top 10 --cores 8
```

Apptainer/SIF:

```bash
apptainer exec lightdock-server.sif \
    .venv/bin/python -m server dock \
        --receptor receptor.pdb --ligand ligand.pdb --output-dir results/
```

## Weights

None. LightDock ships no neural-network weights — its scoring parameters travel
inside the pip package. There is no NAS weight mount; `/healthz/detail` reports
the installed `lightdock_version` and the available `scoring_functions` as the
readiness signal.

## Build & deploy

```bash
# 1. Vendor the upstream source at the pinned SHA (once; re-run to bump).
./services/lightdock-server/scripts/vendor.sh

# 2. Build the image (from project root).
make build-lightdock            # or: docker build -f services/lightdock-server/Dockerfile .
```

FC deployment: CPU instance (8 vCPU / 16–32 GB), async task mode enabled,
data-plane OSS bucket `bioagent-inputs` mounted at `/mnt/oss` (RW). Set the
timeout generously for the requested sampling size.

## Config (env, prefix `LIGHTDOCK_`)

`LIGHTDOCK_JOBS_BASE_DIR`, `LIGHTDOCK_PYTHON`, `LIGHTDOCK_DRIVER_SCRIPT`,
`LIGHTDOCK_BIN_DIR`, `LIGHTDOCK_DEFAULT_SCORING`, `LIGHTDOCK_DEFAULT_CORES`,
`LIGHTDOCK_MAX_CONCURRENT_JOBS`, `LIGHTDOCK_OSS_OUTPUT_MOUNT`. See
[settings.py](settings.py).
