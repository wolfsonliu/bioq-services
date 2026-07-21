"""Service-wide policy for diffdock-pp-server."""

from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter

from .settings import DiffDockPPSettings


class DiffDockPPAdapter(JobAdapter):
    name = "diffdock-pp"

    settings: DiffDockPPSettings  # narrow for IDEs

    def __init__(self, settings: DiffDockPPSettings) -> None:
        super().__init__(settings)

    # ---- Output detection ----

    def detect_outputs(self, job_dir: Path) -> bool:
        """Job is `completed` only if at least dock_pose_1.pdb was written."""
        out = self.output_dir(job_dir) / "dock_pose_1.pdb"
        return out.exists() and out.stat().st_size > 0

    # ---- Subprocess env ----

    def subprocess_cwd(self) -> Path | None:
        # upstream `main_inf.py` + `args.py:process_args` do relative-path
        # imports (`from args import parse_args`) and joins (`os.path.join(
        # args.save_path, args.args_file)`). CWD must be the upstream repo root.
        return self.settings.root

    # ---- Manifest enrichments (agent protocol) ----

    def manifest_extras(self) -> dict:
        return {
            "model": {
                "name": "DiffDock-PP",
                "paper": "arXiv:2304.03889 (ICLR 2023 MLDD Workshop)",
                "task": "rigid protein-protein docking (score + confidence, "
                        "diffusion)",
                "training_data": "DIPS (subset of DOCKGROUND)",
                "note": "Ligand/receptor terminology from EquiDock — both are "
                        "proteins. Model is rigid (no torsion / no side-chain "
                        "flexibility). For flexible or induced-fit binding, "
                        "consider a co-folding service (AlphaFold-Multimer, "
                        "Boltz-2) instead.",
                "auxiliary_model": (
                    "ESM-2 t33_650M_UR50D (residue embeddings; ~2.5 GB) — "
                    "loaded from torch.hub cache on NAS."
                ),
            },
            "tool_outputs": {
                "dock": (
                    "output/dock_pose_<rank>.pdb (rank in [1, top_k]) — "
                    "receptor + rotated/translated ligand in a single PDB "
                    "per pose, ordered by confidence (descending). "
                    "output/confidence_scores.json — the full ranked list "
                    "with per-pose confidence values. "
                    "output/raw_samples.pkl — all N raw samples "
                    "(HeteroData objects) for reanalysis / RMSD scoring."
                ),
            },
            "config_tips": {
                "num_samples": (
                    "40 is the paper default. Downgrade to 10-20 for T4 "
                    "quick runs; runtime ~ linear."
                ),
                "top_k": (
                    "5 covers most 'ensemble score & pick' agent flows "
                    "without hoarding disk. Set higher if downstream will "
                    "rerank with a separate scoring service."
                ),
                "use_confidence_model": (
                    "true (default) is recommended — the confidence pass "
                    "adds ~30% runtime but gives you a ranking signal. "
                    "Set false only when you plan to rerank externally."
                ),
                "actual_steps": (
                    "Keep at 40 unless you know you're compute-bound. "
                    "Reducing below ~30 significantly degrades pose quality."
                ),
            },
            "input_uri_schemes": {
                "upload": (
                    "multipart/form-data — two .pdb files (receptor + ligand)"
                ),
                "oss://<bucket>/<key>": "fetched at submit-time",
                "job://<prev_job_id>/<file>": (
                    "pipeline chaining (e.g. RFantibody or AlphaFold-Multimer "
                    "output → DiffDock-PP resampling)"
                ),
                "file:///<path>": "absolute path on the FC NAS mount",
                "http(s)://...": "streamed at submit-time",
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/dock": [
                EndpointExample(
                    title="basic docking (40 samples, top-5, with confidence)",
                    curl=(
                        "curl -X POST $URL/api/dock "
                        "-F receptor=@receptor.pdb "
                        "-F ligand=@ligand.pdb "
                        "-F num_samples=40 "
                        "-F top_k=5"
                    ),
                    notes=(
                        "Smallest useful call. Returns a JobInfo; poll "
                        "/api/jobs/<id> until completed, then GET "
                        "/api/jobs/<id>/files to download dock_pose_1.pdb etc."
                    ),
                ),
                EndpointExample(
                    title="fast, no confidence model",
                    curl=(
                        "curl -X POST $URL/api/dock "
                        "-F receptor=@receptor.pdb "
                        "-F ligand=@ligand.pdb "
                        "-F num_samples=20 "
                        "-F top_k=5 "
                        "-F use_confidence_model=false"
                    ),
                    notes=(
                        "~30% faster than the default. Use when you'll "
                        "rerank externally or just need a quick pose sample."
                    ),
                ),
                EndpointExample(
                    title="reproducible run with URI input",
                    curl=(
                        "curl -X POST $URL/api/dock "
                        "-F receptor_uri=oss://mybucket/rec.pdb "
                        "-F ligand_uri=oss://mybucket/lig.pdb "
                        "-F num_samples=40 "
                        "-F top_k=10 "
                        "-F seed=42"
                    ),
                    notes=(
                        "Uses OSS URIs (no file upload). Explicit seed gives "
                        "deterministic sampling."
                    ),
                ),
            ],
            "/api/tasks/dock": [
                EndpointExample(
                    title="FC async task mode (single 202 + blocking compute)",
                    curl=(
                        "curl -X POST $URL/api/tasks/dock "
                        "-H 'X-Fc-Invocation-Type: Async' "
                        "-H 'X-Bioagent-Job-Id: my-dock-001' "
                        "-F receptor=@receptor.pdb "
                        "-F ligand=@ligand.pdb "
                        "-F num_samples=40 "
                        "-F top_k=5"
                    ),
                    notes=(
                        "Returns 202 immediately; FC keeps the instance alive "
                        "until the diffusion finishes. Use this for "
                        "production calls behind the FC async-task console "
                        "toggle. Duplicate submits with the same "
                        "X-Bioagent-Job-Id are deduped."
                    ),
                ),
            ],
        }
