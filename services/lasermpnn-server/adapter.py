"""LASErMPNN service adapter.

Output detection scans for the batch scripts' design PDBs / FASTA (label-agnostic
— the framework writes no per-job manifest). Subprocess cwd points at the
LASErMPNN package parent dir so `python -m LASErMPNN.<script>` resolves.
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter, JobInfo, JobStatus  # noqa: F401

from .settings import LASErMPNNSettings


class LASErMPNNAdapter(JobAdapter):
    name = "lasermpnn"

    settings: LASErMPNNSettings  # narrow for IDEs

    def __init__(self, settings: LASErMPNNSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        """Both endpoints write design_*.pdb (per-input subdirs or flat) plus an
        optional designs.fasta. Recursively scan output/ for any non-empty
        design PDB or FASTA."""
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        for pattern in ("**/design_*.pdb", "**/*.fasta", "**/*.fa"):
            for f in out.glob(pattern):
                try:
                    if f.is_file() and f.stat().st_size > 0:
                        return True
                except OSError:
                    continue
        return False

    def subprocess_cwd(self) -> Path | None:
        return self.settings.root

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                "design": "output/**/design_<i>.pdb (+ output/designs.fasta if output_fasta)",
                "design_ligandmpnn": "output/**/design_<i>.pdb (+ output/designs.fasta)",
                "note": (
                    "Each design_<i>.pdb is a redesigned sequence with packed side "
                    "chains. FASTA headers carry a log10-probability score. Use GET "
                    "/api/jobs/{id}/files to enumerate and /api/jobs/{id}/file/{relpath} "
                    "to download a single file."
                ),
            },
            "model_variants": {
                "nothing_heldout": {
                    "weights_file": "laser_weights_0p1A_nothing_heldout.pt",
                    "description": "Default hyperparameter-tuned LASErMPNN model.",
                },
                "ligandmpnn_split": {
                    "weights_file": "laser_weights_0p1A_noise_ligandmpnn_split.pt",
                    "description": "Trained on the LigandMPNN data split.",
                },
                "soluble": {
                    "weights_file": "soluble_weights_no_heldout_drop_clusters_optstep_65000.pt",
                    "description": "Soluble-protein variant.",
                },
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data (field name `pdb`).",
                "job://<id>/<file>": "Re-use a file from a prior lasermpnn job (same NAS).",
                "file:///abs/path": "Direct NAS path; works across services on the shared mount.",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object (needs OSS creds).",
                "http(s)://...": "Generic URL, including OSS signed URLs.",
            },
            "important": (
                "LASErMPNN was trained on PROTONATED structures. The input ligand "
                "MUST carry hydrogens in the correct protonation state, or you may "
                "get unexpected results. This service does NOT protonate inputs — see "
                "the NISE repo's protonate_and_add_conect_records.py."
            ),
            "config_tips": {
                "first_shell_sequence_temp": (
                    "Set a separate (usually lower) temperature for ligand-binding "
                    "residues to keep the binding site conservative while sampling the "
                    "rest of the fold more freely."
                ),
                "fix_beta": (
                    "Set B-factor=1.0 on residues to keep fixed (sequence + rotamer) "
                    "and 0.0 on residues to design."
                ),
                "designs_per_batch": (
                    "Lower this (e.g. 5-10) if you hit GPU OOM on large complexes."
                ),
            },
            "endpoints_summary": {
                "/api/design": "LASErMPNN batch design (ligand-conditioned sequence + side chains).",
                "/api/design_ligandmpnn": "Retrained LigandMPNN variant (paper comparison).",
            },
            "not_in_scope_v0_0_1": (
                "run_inference_tied (symmetric/tied design), proofreading, partial-charge "
                "prediction, and training are deferred. Input protonation is the caller's "
                "responsibility."
            ),
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/design": [
                EndpointExample(
                    title="basic ligand-conditioned design",
                    curl=(
                        "curl -X POST $URL/api/design "
                        "-F pdb=@4jnj-1_prot.pdb "
                        "-F designs_per_input=5 "
                        "-F sequence_temp=0.3 "
                        "-F model_variant=nothing_heldout"
                    ),
                    notes="Output: output/**/design_0..4.pdb (+ designs.fasta).",
                ),
                EndpointExample(
                    title="conservative binding site, ALA/GLY budget",
                    curl=(
                        "curl -X POST $URL/api/design "
                        "-F pdb=@complex.pdb "
                        "-F designs_per_input=10 "
                        "-F sequence_temp=0.5 "
                        "-F first_shell_sequence_temp=0.1 "
                        "-F constrain_ala_gly=true -F ala_budget=4"
                    ),
                    notes="Lower first-shell temp keeps the pocket conservative.",
                ),
            ],
            "/api/design_ligandmpnn": [
                EndpointExample(
                    title="retrained LigandMPNN variant",
                    curl=(
                        "curl -X POST $URL/api/design_ligandmpnn "
                        "-F pdb=@4jnj-1_prot.pdb "
                        "-F designs_per_input=5 "
                        "-F output_fasta=true"
                    ),
                    notes="Uses the ligandmpnn-split checkpoint. For paper comparison.",
                ),
            ],
        }
