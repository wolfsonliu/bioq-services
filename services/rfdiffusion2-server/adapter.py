"""Service-wide policy for rfdiffusion2-server.

Four overrides over the framework defaults:

  * `detect_outputs` walks `output/` for any `design_*.pdb`.
  * `subprocess_cwd` returns the RFdiffusion2 source root — `run_inference.py`
    resolves its Hydra `config_path` relative to its own location, but other
    sub-tools (rf2aa, ipd, ...) walk PYTHONPATH from the CWD.
  * `subprocess_env` injects `PYTHONPATH=<root>` so `rf_diffusion`/`rf2aa`
    imports resolve without requiring a `pip install -e .` step (the upstream
    install path uses `export PYTHONPATH=...` everywhere).
  * `manifest_extras` documents the three endpoints, the supported PDB input
    URI schemes, and the available model checkpoints.
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter, JobInfo, JobStatus

from .models import MODEL_CHOICES
from .settings import RFdiffusion2Settings
from .tools import OUTPUT_STEM


class RFdiffusion2Adapter(JobAdapter):
    name = "rfdiffusion2"

    settings: RFdiffusion2Settings  # narrow type for IDEs

    def __init__(self, settings: RFdiffusion2Settings) -> None:
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
        """run_inference.py is Hydra-based; cwd is the repo root."""
        return self.settings.root

    def subprocess_env(self) -> dict[str, str]:
        """Mirror `export PYTHONPATH=<repo>` from upstream setup instructions."""
        return {"PYTHONPATH": str(self.settings.pythonpath)}

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/generate/active_site": [
                EndpointExample(
                    title="lactate dehydrogenase NAD+OXM active site (unindexed)",
                    curl=(
                        "curl -X POST $URL/api/generate/active_site "
                        "-F input_pdb=@M0584_1ldm.pdb "
                        "-F 'contigs=46,A106-106,59,A166-166,2,A169-169,23,A193-193,46' "
                        "-F 'ligand=NAD,OXM' "
                        "-F 'contig_atoms={\"A106\":\"NE,CD,CZ\",\"A166\":\"OD1,CG\","
                        "\"A169\":\"NH2,CZ\",\"A193\":\"NE2,CD2,CE1\"}' "
                        "-F contig_as_guidepost=true "
                        "-F num_designs=10"
                    ),
                    notes=(
                        "Reproduces the `active_site_unindexed_atomic` demo. "
                        "Set `contig_as_guidepost=false` for the harder indexed variant; "
                        "supply `partially_fixed_ligand=<JSON>` and/or "
                        "`only_guidepost_positions=A106` for the other three demo flavours."
                    ),
                ),
            ],
            "/api/generate/small_molecule_binder": [
                EndpointExample(
                    title="150aa buried binder around PH2 with RASA conditioning",
                    curl=(
                        "curl -X POST $URL/api/generate/small_molecule_binder "
                        "-F input_pdb=@trimmed_ec2_M0151_NO_ORI_zero_com0.pdb "
                        "-F contigs=150 -F length=150-150 -F ligand=PH2 "
                        "-F rasa_active=true -F rasa_target=0 "
                        "-F num_designs=10"
                    ),
                    notes=(
                        "Reproduces `small_molecule_binder_rasa_buried`. "
                        "Lower rasa_target (0.0) → ligand fully buried in the binder. "
                        "Set rasa_active=false to skip RASA conditioning entirely."
                    ),
                ),
            ],
            "/api/generate": [
                EndpointExample(
                    title="custom: `aa_ppi.yaml` config + raw Hydra overrides",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F input_pdb=@target.pdb "
                        "-F config_name=aa_ppi "
                        "-F input_pdb_required=true "
                        "-F 'contigs=A1-150/0 70-100' "
                        "-F num_designs=10 "
                        "-F 'extra_overrides={\"ppi.hotspot_res\":\"[A59,A83,A91]\"}'"
                    ),
                    notes=(
                        "Escape hatch for any inference config or override not exposed "
                        "by the structured endpoints. See "
                        "`rf_diffusion/config/inference/` in the vendored upstream "
                        "source for the catalogue of configs."
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
                    "Each backbone lands at output/design_<N>.pdb. The matching "
                    ".trb file pickles the contig, the resolved config, and the "
                    "input→output residue mapping. For the active_site endpoint "
                    "the motif residues' sidechain rotamers are preserved; "
                    "diffused residues are backbone-only (run LigandMPNN downstream "
                    "for sequence design — see ProteinMPNN-server for an inline option)."
                ),
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data UploadFile (`input_pdb`).",
                "job://<job_id>/<filename>": "Pull a PDB from a previous job on this NAS.",
                "file:///abs/path": "Direct NAS path (shared across services on the same mount).",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object.",
                "http(s)://...": "Generic HTTP(S) URL — incl. signed OSS URLs.",
            },
            "endpoints_summary": {
                "/api/generate/active_site": (
                    "Active-site scaffolding with atomic motif + ligand. Reproduces "
                    "open_source_demo.json `active_site_*` cases."
                ),
                "/api/generate/small_molecule_binder": (
                    "Small-molecule binder design with optional RASA conditioning."
                ),
                "/api/generate": (
                    "Raw contig + freeform Hydra overrides (any config under "
                    "rf_diffusion/config/inference/)."
                ),
            },
            "models": {
                "directory": str(self.settings.models_dir),
                "checkpoints": MODEL_CHOICES,
                "default": "RFD_140.pt (set by aa.yaml). Pass `model=rfd_173` to override.",
            },
            "paper_reference": (
                "Ahern et al. 2025 — \"Atom level enzyme active site scaffolding "
                "using RFdiffusion2\" (bioRxiv). Compared with RFdiffusion v1 "
                "(rfdiffusion-server), v2 handles ATOMIC motifs (sidechain "
                "anchoring) + small molecules; v1 handles RESIDUE motifs + "
                "protein-only PPI binders. Use v1 for ProteinMPNN-only sequence "
                "design pipelines, v2 for enzyme/ligand-aware design."
            ),
        }

    def infer_job_from_dir(self, job_dir: Path) -> JobInfo:
        """Recovered jobs surface the PDB count as a hint."""
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
