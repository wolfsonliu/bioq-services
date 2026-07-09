# openbpmd-server

HTTP + CLI wrapper around [OpenBPMD](https://github.com/Gervasiolab/OpenBPMD)
(Lukauskis et al., *JCIM* 2022) — **Binding Pose Metadynamics** for scoring the
stability of a protein–ligand binding pose.

Design doc: [`engineering/decisions/2026-07-08-openbpmd-server-design.md`](../../engineering/decisions/2026-07-08-openbpmd-server-design.md).

## What it does

Given a **pre-solvated, parametrised** protein–ligand complex, OpenBPMD runs a
short metadynamics protocol (minimise → 500 ps NVT equilibration → *N* reps ×
10 ns metadynamics biasing the ligand RMSD) and reports a stability score:

```
CompScore = PoseScore − 5 × ContactScore
```

- **PoseScore** — ligand heavy-atom RMSD (lower = more stable)
- **ContactScore** — fraction of native contacts retained (higher = better)
- **CompScore** — combined; **more negative = more stable pose**

Use it to re-rank / triage docking poses between fast docking rescore (seconds)
and rigorous FEP ΔG (days). It is **not** an affinity / ΔG predictor.

> OpenBPMD does **not** prepare systems. Provide a solvated, parametrised
> complex (ligand FF params + water + ions). Inputs are Amber (`.prm7`/`.rst7`)
> or Gromacs (`.top`/`.gro`).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/score` | Score a pose (submit/poll) |
| POST | `/api/tasks/score` | Same, FC async task mode (returns 202) |
| GET | `/healthz`, `/healthz/detail` | Health + OpenMM/CUDA probe |
| GET | `/api/jobs/*`, `/api/manifest`, `/openapi.json` | Framework endpoints |

### Request fields (`ScoreRequest`)

| Field | Default | Notes |
|---|---|---|
| `lig_resname` | `MOL` | Ligand residue name in the topology |
| `nreps` | `10` | Independent metadynamics repeats (serial) |
| `hill_height` | `0.3` | kcal/mol; standard is 0.3 |
| `system_format` | auto | `amber` / `gromacs`; auto-detect by extension |
| `sim_ns` | *(unset)* | **ADVANCED/TESTING** — production length in ns; unset = 10. Non-standard values break score comparability |
| `equil_steps` | *(unset)* | **ADVANCED/TESTING** — equilibration steps; unset = 250000 |

File inputs: `structure`+`parameters` (multipart) or `structure_uri`+`parameters_uri`
(`oss://` / `job://` / `file://` / `http(s)://`). Prefer URIs — a `.prm7` is
~8 MB and exceeds the FC async 128 KiB payload cap for multipart.

## Usage

### HTTP (submit/poll)

```bash
curl -X POST $URL/api/score \
    -F structure=@solvated.rst7 -F parameters=@solvated.prm7 \
    -F lig_resname=MOL -F nreps=10 -F hill_height=0.3
# → {"job_id": "...", "status": "pending"}; poll GET /api/jobs/<id>
```

### CLI batch mode (HPC / sbatch)

```bash
apptainer exec --nv openbpmd-server.sif \
    python -m server score \
    --structure  solvated.rst7 \
    --parameters solvated.prm7 \
    --output-dir /scratch/$SLURM_JOB_ID/ \
    --lig-resname MOL --nreps 10 --hill-height 0.3
```

## Outputs

Under `<job_dir>/output/`:

- `results.csv` — final CompScore/PoseScore/ContactScore + SD (one row)
- `scoring_stats.json` — wrapper summary (nreps_done, scores, wall time, platform)
- `rep_*/bpm_results.csv` — time-resolved per-frame scores
- `minimized_system.pdb`, `equil_system.pdb`, `centred_equil_system.pdb`

Job is `completed` only when `results.csv` exists (written after all reps finish).

## Build

```bash
./services/openbpmd-server/scripts/vendor.sh   # vendor upstream/ (once)
make build-openbpmd-server                      # or: docker build -f services/openbpmd-server/Dockerfile .
```

No model weights to fetch — OpenBPMD is a pure OpenMM workflow.

## Implementation notes

- **`simtk`→`openmm` shim** (`inference.py`): upstream uses the `simtk`
  namespace removed in OpenMM 8.x; the wrapper aliases it at runtime so the
  vendored source stays unmodified. Pinning OpenMM 7.7 is rejected (no CUDA
  kernels for recent GPUs).
- **Configurable platform**: upstream hardcodes CUDA; the wrapper honours
  `OPENBPMD_PLATFORM` (production `CUDA`, offline smoke `CPU`).
- **One patch** (`patches/0001-parametrize-md-lengths.patch`): parametrizes the
  hardcoded 10 ns / 250000-step lengths (defaults preserve upstream behaviour)
  so integration tests can run a short trajectory.

## Testing

```bash
# Offline (mocked subprocess)
uv run python -m pytest services/openbpmd-server/tests/test_app.py \
    services/openbpmd-server/tests/test_cli.py -v

# FC integration (opt-in; requires deployed service + GPU).
# Fixtures resolve from opensource/OpenBPMD/tests/files/ (ligand resname UNK),
# or override via OPENBPMD_TEST_STRUCTURE / OPENBPMD_TEST_PARAMETERS.
RUN_FC_TESTS=1 uv run python -m pytest -m fc \
    services/openbpmd-server/tests/test_fc.py \
    services/openbpmd-server/tests/test_fc_task.py -v
```

## Deployment

- **FC async task mode** (primary): enable the async-task console toggle; call
  `/api/tasks/score`. A single scoring job maps to one async task.
- **HPC apptainer** (long tail): full 10×10 ns can exceed FC's 24 h ceiling on
  slow GPUs / large systems — run via `python -m server score` on a GPU node.

Runtime scales with system size × GPU: ~5–10 h for 10 reps on an A100, longer on
T4/A10. Reduce `nreps` for a faster, lower-confidence triage.
