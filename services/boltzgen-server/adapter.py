"""BoltzGen service adapter.

Detects pipeline outputs (final ranked designs or intermediate CIFs) and
exposes manifest metadata for the binder design pipeline.
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter

from .settings import BoltzGenSettings


class BoltzGenAdapter(JobAdapter):
    name = "boltzgen"

    settings: BoltzGenSettings

    def __init__(self, settings: BoltzGenSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        final = out / "final_ranked_designs"
        if final.is_dir():
            if any(final.rglob("*.cif")) or any(final.rglob("*.csv")):
                return True
        for subdir_name in (
            "intermediate_designs_inverse_folded",
            "intermediate_designs",
        ):
            subdir = out / subdir_name
            if subdir.is_dir() and any(subdir.glob("*.cif")):
                return True
        return False

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                "design": (
                    "output/final_ranked_designs/ — filtered CIF structures + metrics CSV. "
                    "Also: output/**/aggregate_metrics_analyze.csv for all-design metrics."
                ),
                "inverse_fold": (
                    "output/final_ranked_designs/ — same structure as design, but starting "
                    "from inverse folding of a provided backbone."
                ),
                "note": (
                    "Use GET /api/jobs/{id}/files to enumerate; download a single file "
                    "via /api/jobs/{id}/file/{relpath} or zip the whole dir via /download."
                ),
            },
            "protocols": {
                "protein-anything": "De novo protein binder design (default)",
                "peptide-anything": "Peptide binder design (auto-filters Cys, stricter RMSD)",
                "protein-small_molecule": "Protein binder for small molecule (includes affinity prediction)",
                "nanobody-anything": "Nanobody binder design (auto-filters Cys)",
                "antibody-anything": "Fab antibody CDR design (auto-filters Cys)",
                "protein-redesign": "Redesign existing protein interfaces",
            },
            "models": {
                "design-diverse": "boltzgen1_diverse.ckpt — diverse backbone generation",
                "design-adherence": "boltzgen1_adherence.ckpt — adherence-optimized generation",
                "inverse-fold": "boltzgen1_ifold.ckpt — sequence prediction from backbone",
                "folding": "boltz2_conf_final.ckpt — structure refolding + confidence",
                "affinity": "boltz2_aff.ckpt — binding affinity prediction",
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data fields `design_yaml` and `ref_files`.",
                "design_yaml_uri": "URI to the design spec YAML (instead of a `design_yaml` upload).",
                "ref_files_zip_uri": (
                    "URI to a zip of the CIF/PDB files the YAML references, extracted "
                    "flat next to the spec. Use this when `ref_files` can't be uploaded "
                    "(e.g. via the gateway, which dispatches form fields only)."
                ),
                "job://<id>/<file>": "Re-use a file from a prior job on the same NAS.",
                "file:///abs/path": "Direct NAS path.",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object.",
                "http(s)://...": "Generic URL, including OSS signed URLs.",
            },
            "config_tips": {
                "num_designs": "default 100 (upstream default 10000); FC recommended 50-500",
                "budget": "final selection count after quality+diversity filtering; default 30",
                "protocol": "auto-configures filtering/analysis settings per use case",
                "use_kernels": "auto=detect GPU capability; set 'false' for older GPUs",
                "skip_inverse_folding": "when design spec fully specifies sequences",
                "design_yaml": (
                    "BoltzGen design spec YAML — see "
                    "https://github.com/HannesStark/boltzgen/tree/main/example for format"
                ),
            },
            "not_in_scope_v0_0_1": (
                "merge, per-step execution, multi-GPU, training, "
                "structured sequence input (use YAML directly)"
            ),
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/design": [
                EndpointExample(
                    title="de novo protein binder design",
                    curl=(
                        "curl -X POST $URL/api/design "
                        "-F design_yaml=@design_spec.yaml "
                        "-F ref_files=@target.cif "
                        "-F protocol=protein-anything "
                        "-F num_designs=100 "
                        "-F budget=30"
                    ),
                    notes=(
                        "design_spec.yaml defines the target structure and designed region. "
                        "ref_files should include all CIF/PDB files referenced by the YAML."
                    ),
                ),
                EndpointExample(
                    title="nanobody CDR design",
                    curl=(
                        "curl -X POST $URL/api/design "
                        "-F design_yaml=@nanobody.yaml "
                        "-F ref_files=@scaffold.cif "
                        "-F protocol=nanobody-anything "
                        "-F num_designs=50"
                    ),
                    notes="Protocol auto-filters Cys in inverse fold step.",
                ),
            ],
            "/api/inverse_fold": [
                EndpointExample(
                    title="inverse fold existing structure",
                    curl=(
                        "curl -X POST $URL/api/inverse_fold "
                        "-F design_yaml=@backbone.yaml "
                        "-F ref_files=@structure.cif "
                        "-F inverse_fold_num_sequences=10"
                    ),
                    notes=(
                        "Skips design diffusion; runs inverse_folding -> folding -> "
                        "analysis -> filtering on the provided backbone."
                    ),
                ),
            ],
        }
