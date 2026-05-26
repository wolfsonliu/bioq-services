"""ODesign service adapter.

Detects CIF prediction outputs and exposes manifest metadata for the
cross-modality biomolecular interaction design pipeline.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import EndpointExample, JobAdapter

from .settings import ODesignSettings


class ODesignAdapter(JobAdapter):
    name = "odesign"

    settings: ODesignSettings

    def __init__(self, settings: ODesignSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        return any(out.rglob("predictions/*.cif"))

    def subprocess_cwd(self) -> Path | None:
        return self.settings.root

    def subprocess_env(self) -> dict[str, str]:
        return {
            "DATA_ROOT_DIR": str(self.settings.data_root_dir),
            "CKPT_ROOT_DIR": str(self.settings.ckpt_root_dir),
            "CUTLASS_PATH": "/kernels/cutlass",
            "LAYERNORM_TYPE": "fast_layernorm",
        }

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                "predictions": (
                    "output/<sample>/seed_<seed>/predictions/*.cif — "
                    "generated structures (backbone + inverse-folded sequence)."
                ),
                "note": (
                    "Use GET /api/jobs/{id}/files to enumerate; download a single "
                    "file via /api/jobs/{id}/file/{relpath} or zip via /download."
                ),
            },
            "models": {
                "odesign_base_prot_flex": "Protein binder design (flexible receptor)",
                "odesign_base_prot_rigid": "Protein binder design (rigid receptor)",
                "odesign_base_ligand_rigid": "Small-molecule ligand design",
                "odesign_base_na_rigid": "DNA/RNA aptamer design (specify design_modality=dna or rna)",
            },
            "task_types": {
                "binder_design": "Protein-protein binder design (hotspot conditioning)",
                "ligand_binding_protein": "Ligand-binding protein design",
                "motif_scaffolding": "Motif scaffolding (motif_scaffolding=true in JSON)",
                "atom_scaffolding": "Atomic motif scaffolding (condition_atom in JSON)",
                "cyclic_peptide": "Cyclic peptide binder (if_cyc='true' in JSON chain)",
                "partial_diffusion": "Partial re-diffusion of existing structure",
                "rna_aptamer": "Protein-binding RNA aptamer design",
                "dna_aptamer": "Protein-binding DNA aptamer design",
                "monomer_design": "Free monomer generation (no ref_file)",
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data fields `input_json` and `ref_files`.",
                "job://<id>/<file>": "Re-use a file from a prior job on the same NAS.",
                "file:///abs/path": "Direct NAS path.",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object.",
                "http(s)://...": "Generic URL, including OSS signed URLs.",
            },
            "config_tips": {
                "n_sample": "backbones per seed; default 5. Lower for faster runs.",
                "seeds": "JSON array of ints; default [42]. More seeds = more diversity.",
                "model": "prot_flex for most protein tasks; na_rigid + design_modality for NA.",
                "invfold_topk": "sequence variants per backbone; default 1.",
                "input_json": (
                    "ODesign JSON spec — see "
                    "https://odesign1.github.io for format documentation"
                ),
            },
            "paper_reference": "Zhang et al. 2025 — arXiv:2510.22304",
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/design": [
                EndpointExample(
                    title="protein binder design (flex receptor)",
                    curl=(
                        "curl -X POST $URL/api/design "
                        "-F input_json=@design_spec.json "
                        "-F ref_files=@target.pdb "
                        "-F model=odesign_base_prot_flex "
                        "-F n_sample=5"
                    ),
                    notes=(
                        "design_spec.json defines chains, hotspot, and ref_file. "
                        "ref_files must include all PDB/CIF files referenced in the JSON."
                    ),
                ),
                EndpointExample(
                    title="RNA aptamer design",
                    curl=(
                        "curl -X POST $URL/api/design "
                        "-F input_json=@rna_spec.json "
                        "-F ref_files=@target.cif "
                        "-F model=odesign_base_na_rigid "
                        "-F design_modality=rna "
                        "-F n_sample=5"
                    ),
                    notes="design_modality is required for na_rigid model.",
                ),
            ],
        }
