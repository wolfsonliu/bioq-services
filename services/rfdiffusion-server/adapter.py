"""Service-wide policy for rfdiffusion-server.

Three overrides over the framework defaults:

  * `detect_outputs` walks `output/` (and `output/traj/` if trajectories are
    enabled) for any `*.pdb`, since every generation mode writes there.
  * `subprocess_cwd` returns the RFdiffusion source root — the script's Hydra
    `config_dir` is resolved relative to it.
  * `manifest_extras` documents the five endpoints' output filename convention,
    the supported input-URI schemes (PDB upload alternatives), and the available
    model checkpoints baked into the image.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import EndpointExample, JobAdapter, JobInfo, JobStatus

from .models import MODEL_CHOICES
from .settings import RFdiffusionSettings
from .tools import OUTPUT_STEM


class RFdiffusionAdapter(JobAdapter):
    name = "rfdiffusion"

    settings: RFdiffusionSettings  # narrow type for IDEs

    def __init__(self, settings: RFdiffusionSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        """Any non-empty `output/design_*.pdb` counts as success."""
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        for f in out.glob(f"{OUTPUT_STEM}_*.pdb"):
            try:
                if f.stat().st_size > 0:
                    return True
            except OSError:
                continue
        return False

    def subprocess_cwd(self) -> Path | None:
        """run_inference.py uses Hydra; cwd needs to be the project root."""
        return self.settings.root

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/generate/unconditional": [
                EndpointExample(
                    title="150aa unconditional monomer",
                    curl=(
                        "curl -X POST $URL/api/generate/unconditional "
                        "-F min_length=150 -F max_length=150 -F num_designs=10"
                    ),
                    notes="No PDB required. Set cyclic=true for an RFpeptides macrocycle.",
                ),
            ],
            "/api/generate/motif": [
                EndpointExample(
                    title="scaffold RSV-F site 5",
                    curl=(
                        "curl -X POST $URL/api/generate/motif "
                        "-F input_pdb=@5TPN.pdb "
                        "-F 'contigs=10-40/A163-181/10-40' "
                        "-F num_designs=10"
                    ),
                    notes="Letters in contigs reference chain+residue in the input PDB.",
                ),
            ],
            "/api/generate/binder": [
                EndpointExample(
                    title="binder vs insulin receptor with hotspots",
                    curl=(
                        "curl -X POST $URL/api/generate/binder "
                        "-F input_pdb=@insulin_target.pdb "
                        "-F 'contigs=A1-150/0 70-100' "
                        "-F 'hotspots=A59,A83,A91' "
                        "-F num_designs=10 -F noise_scale=0.0"
                    ),
                    notes=(
                        "Binder mode defaults to noise_scale=0 (per upstream "
                        "README). Provide 3–6 hotspots on the target."
                    ),
                ),
            ],
            "/api/generate/symmetry": [
                EndpointExample(
                    title="C6 oligomer with olig_contacts potential",
                    curl=(
                        "curl -X POST $URL/api/generate/symmetry "
                        "-F symmetry=c6 -F total_length=480 -F num_designs=10 "
                        '-F \'guiding_potentials=["type:olig_contacts,weight_intra:1,weight_inter:0.1"]\' '
                        "-F guide_scale=2.0 -F guide_decay=quadratic "
                        "-F olig_intra_all=true -F olig_inter_all=true"
                    ),
                    notes="`total_length` must be divisible by the chain count for the chosen symmetry.",
                ),
            ],
            "/api/generate": [
                EndpointExample(
                    title="partial diffusion around an existing fold",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F input_pdb=@2KL8.pdb "
                        "-F 'contigs=79-79' "
                        "-F num_designs=10 "
                        "-F 'extra_overrides={\"diffuser.partial_T\": 10}'"
                    ),
                    notes=(
                        "Use this escape hatch for partial diffusion, fold-conditioning, "
                        "scaffold-guided runs, or any Hydra key not exposed by the structured "
                        "endpoints."
                    ),
                ),
            ],
        }

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                "all_modes": f"output/{OUTPUT_STEM}_*.pdb (+ matching .trb metadata)",
                "trajectory": "output/traj/*.pdb (only when write_trajectory=true)",
                "note": (
                    "Per-trajectory backbones land at output/design_<N>.pdb. The "
                    "matching .trb file holds the contig used, the resolved config, "
                    "and the input→output residue mapping. Every residue is glycine "
                    "since RFdiffusion is backbone-only; run ProteinMPNN downstream "
                    "for sequence design."
                ),
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data UploadFile (`input_pdb`) — works for motif/binder/generic.",
                "job://<job_id>/<filename>": "Pull a PDB from a previous job on this NAS.",
                "file:///abs/path": "Direct NAS path (shared across services on the same mount).",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object.",
                "http(s)://...": "Generic HTTP(S) URL — incl. signed OSS URLs.",
                "unconditional": "No input PDB needed for `/api/generate/unconditional`.",
            },
            "endpoints_summary": {
                "/api/generate/unconditional": "Length-only monomer or macrocycle (cyclic=true).",
                "/api/generate/motif": "Scaffold around motif residues in an input PDB.",
                "/api/generate/binder": "PPI binder design vs a target PDB; optional hotspots.",
                "/api/generate/symmetry": "Cyclic / dihedral / tetrahedral oligomers.",
                "/api/generate": "Raw contig + freeform Hydra overrides (partial diffusion, fold conditioning, ...).",
            },
            "models": {
                "directory": str(self.settings.models_dir),
                "checkpoints": MODEL_CHOICES,
                "note": (
                    "Leave `model` unset to let run_inference.py pick the right checkpoint "
                    "based on the inputs (provide_seq → InpaintSeq, scaffoldguided → Fold, etc)."
                ),
            },
            "downstream_pipeline_tip": (
                "RFdiffusion outputs backbone-only poly-G PDBs. Standard next steps: "
                "(1) ProteinMPNN for sequence design (see proteinmpnn-server), "
                "(2) AF2 / RF2 for structure validation. The proteinmpnn-server accepts "
                "`input_uri=job://<rfdiffusion_job_id>/design_0.pdb` for zero-copy chaining."
            ),
        }

    def infer_job_from_dir(self, job_dir: Path) -> JobInfo:
        """Recovered jobs surface the number of PDBs already on disk as a hint."""
        out = self.output_dir(job_dir)
        pdbs = (
            [f for f in out.glob(f"{OUTPUT_STEM}_*.pdb") if f.is_file() and f.stat().st_size > 0]
            if out.is_dir()
            else []
        )
        if pdbs:
            return JobInfo(
                job_id=job_dir.name,
                status=JobStatus.COMPLETED,
                message=f"Recovered from disk ({len(pdbs)} PDB outputs)",
            )
        return JobInfo(
            job_id=job_dir.name,
            status=JobStatus.FAILED,
            message="Recovered from disk (no PDB outputs)",
        )
