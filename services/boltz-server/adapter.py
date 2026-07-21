"""Boltz service adapter.

Detects outputs across both endpoints (`predict_structure` and
`predict_affinity`), supplies the service-specific manifest section, and lists
the subprocess cwd (Boltz repo root) so the `boltz` CLI's relative imports
resolve consistently.
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter

from .settings import BoltzSettings


class BoltzAdapter(JobAdapter):
    name = "boltz"

    settings: BoltzSettings  # narrow for IDEs

    def __init__(self, settings: BoltzSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        """Recognize boltz predict's output layout.

        Boltz writes predictions to `<out_dir>/boltz_results_<yaml_stem>/predictions/<yaml_stem>/`.
        The service fixes the YAML stem to `input` (see tools.build_yaml), so a
        single glob covers both endpoints. Affinity adds an extra json file in
        the same directory, but at least one `*_model_*.{cif,pdb}` is required
        for the job to count as completed (no model = nothing useful to return).
        """
        pred_dir = self.output_dir(job_dir) / "boltz_results_input" / "predictions" / "input"
        if not pred_dir.is_dir():
            return False
        for ext in ("cif", "pdb"):
            for path in pred_dir.glob(f"*_model_*.{ext}"):
                if path.stat().st_size > 0:
                    return True
        return False

    def subprocess_cwd(self) -> Path | None:
        return self.settings.root

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                "predict_structure": (
                    "output/boltz_results_input/predictions/input/input_model_*.{cif,pdb} (one per diffusion_samples). "
                    "confidence_input_model_*.json carries per-sample scores (confidence, ptm, iptm, "
                    "complex_plddt, chain-wise breakdowns)."
                ),
                "predict_affinity": (
                    "Same as predict_structure plus affinity_input.json containing "
                    "affinity_pred_value (log10 IC50, lower = stronger binding) and "
                    "affinity_probability_binary (binder vs decoy probability)."
                ),
                "extras": (
                    "pae_*.npz / pde_*.npz / plddt_*.npz when --write_full_pae/--write_full_pde are set; "
                    "use GET /api/jobs/{id}/files to enumerate."
                ),
            },
            "model": {
                "name": "boltz2",
                "supports_affinity": True,
                "weights": ["boltz2_conf.ckpt", "boltz2_aff.ckpt"],
                "ccd_data": "mols/ (rdkit-friendly per-molecule pickle directory; required by Boltz-2)",
                "notes": (
                    "v0.0.1 hard-codes --model boltz2; Boltz-1 is intentionally out of scope. "
                    "If Boltz-1 is needed, spin up a sibling boltz1-server (see decisions doc)."
                ),
            },
            "msa_modes": {
                "auto": (
                    "ColabFold MMseqs2 server (default https://api.colabfold.com). "
                    "~30-60 s per protein chain; needs outbound HTTPS from FC."
                ),
                "provided": (
                    "Per-chain .a3m file uploaded via multipart `msa_files` (filename must match "
                    "chain id, e.g. A.a3m) or referenced via msa_uri on the SequenceEntry."
                ),
                "empty": (
                    "Single-sequence mode. Recommended ONLY for antibody VH/VL chains where MSA hurts "
                    "more than it helps; everything else should use auto or provided."
                ),
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data — pdb/cif/a3m uploads as `msa_files` / `template_files` / `raw_yaml_upload`.",
                "job://<id>/<file>": "Re-use a file from a prior boltz job or any bioagent service's NAS output.",
                "file:///abs/path": "Direct NAS path; works across services on the shared mount.",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object (needs OSS_ACCESS_KEY_ID / _SECRET).",
                "http(s)://...": "Generic URL, including OSS pre-signed URLs.",
            },
            "config_tips": {
                "recycling_steps": (
                    "Default 3 (Boltz tutorial). Set 10 for AF3-style configuration "
                    "(~3x runtime, marginally higher accuracy)."
                ),
                "diffusion_samples": (
                    "Default 1 (single best-guess structure). Set 25 for AF3-style ensembling — "
                    "useful for downstream ranking but ~25x runtime."
                ),
                "use_potentials": (
                    "Adds inference-time potentials for physical plausibility; ~1.5x runtime."
                ),
                "msa_mode": (
                    "For antibody-antigen prediction: VH/VL chains use `empty`, antigen chain uses "
                    "`auto` (per-chain msa_file is per-SequenceEntry, not global)."
                ),
                "affinity_binder": (
                    "Must reference a ligand SequenceEntry (smiles or ccd). Max 128 heavy atoms "
                    "(model trained on ≤56; quality degrades past that)."
                ),
                "raw_yaml": (
                    "Drop down to the upstream YAML schema for advanced features (covalent bonds, "
                    "multi-template, complex constraints). When set, all structured fields "
                    "(sequences/constraints/templates) must be empty."
                ),
            },
            "endpoints_summary": {
                "/api/predict_structure": "Complex structure prediction (proteins + ligands + NA + templates).",
                "/api/predict_affinity": "Ligand binding affinity (log10 IC50 + binder probability) + structure.",
            },
            "not_in_scope_v0_0_1": (
                "Boltz-1 support (intentional), batch YAML (one-job-per-target convention), "
                "Boltz training, MSA-server auth, CSV-format paired MSA."
            ),
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/predict_structure": [
                EndpointExample(
                    title="single protein, auto MSA",
                    curl=(
                        "curl -X POST $URL/api/predict_structure "
                        "-F name=test "
                        "-F msa_mode=auto "
                        "-F 'sequences=[{\"type\":\"protein\",\"id\":\"A\","
                        "\"sequence\":\"QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG\"}]'"
                    ),
                    notes=(
                        "ColabFold MSA server is hit at job time (~30-60s). For predictable "
                        "behavior in tests, pass `msa_mode=empty` instead (lower accuracy but no network)."
                    ),
                ),
                EndpointExample(
                    title="antibody-antigen complex (VH/VL empty MSA, antigen auto MSA)",
                    curl=(
                        "curl -X POST $URL/api/predict_structure "
                        "-F name=ab_ag "
                        "-F msa_mode=provided "
                        "-F 'sequences=["
                        "{\"type\":\"protein\",\"id\":\"H\",\"sequence\":\"EVQLV...\",\"msa_uri\":\"empty\"},"
                        "{\"type\":\"protein\",\"id\":\"L\",\"sequence\":\"DIQMT...\",\"msa_uri\":\"empty\"},"
                        "{\"type\":\"protein\",\"id\":\"A\",\"sequence\":\"MAHHH...\"}"
                        "]' "
                        "-F msa_files=@antigen.a3m"
                    ),
                    notes=(
                        "Per-chain MSA control: each SequenceEntry's `msa_uri` overrides the global "
                        "`msa_mode`. Set msa_uri=empty to skip MSA per chain; otherwise the matching "
                        "uploaded a3m (matched by basename = chain id) is used."
                    ),
                ),
            ],
            "/api/predict_affinity": [
                EndpointExample(
                    title="protein target + SMILES ligand",
                    curl=(
                        "curl -X POST $URL/api/predict_affinity "
                        "-F name=binding "
                        "-F binder_id=B "
                        "-F msa_mode=auto "
                        "-F 'sequences=["
                        "{\"type\":\"protein\",\"id\":\"A\",\"sequence\":\"MVTPEGNVSL...\"},"
                        "{\"type\":\"ligand\",\"id\":\"B\",\"smiles\":\"N[C@@H](Cc1ccc(O)cc1)C(=O)O\"}"
                        "]'"
                    ),
                    notes=(
                        "Output includes affinity_input.json: affinity_pred_value (log10 IC50, lower="
                        "stronger binder) and affinity_probability_binary (0-1 binder vs decoy)."
                    ),
                ),
            ],
        }
