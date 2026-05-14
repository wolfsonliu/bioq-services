"""ProteinMPNN service adapter.

Detects outputs across the three endpoint modes (design / score / probs),
points subprocess cwd at the ProteinMPNN repo root, and contributes the
service-specific manifest section. Real `manifest_extras` / `endpoint_examples`
get fleshed out in later tasks.
"""

from __future__ import annotations

from pathlib import Path

from bioagent_service import EndpointExample, JobAdapter, JobInfo, JobStatus  # noqa: F401

from .settings import ProteinMPNNSettings


class ProteinMPNNAdapter(JobAdapter):
    name = "proteinmpnn"

    settings: ProteinMPNNSettings  # narrow for IDEs

    def __init__(self, settings: ProteinMPNNSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        """Recognize any of the three modes' artifacts:
          - design  : output/seqs/*.fa
          - score   : output/score_only/*.npz
          - probs   : output/probs/*.npz
        """
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        for pattern in ("seqs/*.fa", "score_only/*.npz", "probs/*.npz"):
            for f in out.glob(pattern):
                try:
                    if f.stat().st_size > 0:
                        return True
                except OSError:
                    continue
        return False

    def subprocess_cwd(self) -> Path | None:
        return self.settings.root

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                "design": "output/seqs/<name>.fa",
                "score": "output/score_only/<name>_pdb.npz",
                "probs": "output/probs/<name>.npz",
                "note": (
                    "Use GET /api/jobs/{id}/files to enumerate; download a single "
                    "file via /api/jobs/{id}/file/{relpath}."
                ),
            },
            "model_variants": {
                "vanilla": {
                    "weights_dir": "vanilla_model_weights/",
                    "model_names": ["v_48_002", "v_48_010", "v_48_020", "v_48_030"],
                    "description": "Full backbone (default).",
                },
                "soluble": {
                    "weights_dir": "soluble_model_weights/",
                    "model_names": ["v_48_002", "v_48_010", "v_48_020", "v_48_030"],
                    "description": "Trained on soluble proteins only.",
                },
                "ca_only": {
                    "weights_dir": "ca_model_weights/",
                    "model_names": ["v_48_002", "v_48_010", "v_48_020"],
                    "description": "CA-only structures (no v_48_030 available).",
                },
                "abmpnn": {
                    "weights_dir": "AbMPNN_model_weights/",
                    "model_names": ["abmpnn"],
                    "description": "Antibody-specific weights (ICML 2023 Workshop).",
                },
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data (field name `pdb`).",
                "job://<id>/<file>": "Re-use a file from a prior proteinmpnn job (same NAS).",
                "file:///abs/path": "Direct NAS path; works across services on the shared mount.",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object (needs OSS_ACCESS_KEY_ID / _SECRET env vars).",
                "http(s)://...": "Generic URL, including OSS signed URLs.",
            },
            "config_tips": {
                "fixed_positions": (
                    "Format '<idx idx ...>, <idx idx ...>' — comma segments aligned to "
                    "chains_to_design. Residue indices are 1-based and refer to position in "
                    "the chain, NOT PDB residue numbers."
                ),
                "tied_positions": (
                    "Same format as fixed_positions. Set homooligomer=true to repeat the "
                    "first chain's tied positions across symmetric subunits."
                ),
                "sampling_temp": (
                    "Space-separated list of temperatures, e.g. '0.1 0.2'. Each one is "
                    "sampled num_seq_per_target times. Suggested 0.1–0.3."
                ),
                "bias_AA": (
                    "JSON dict mapping single-letter AA → bias float. Positive = more likely. "
                    "Example: {\"D\":1.39,\"E\":1.39} biases toward acidic residues."
                ),
            },
            "endpoints_summary": {
                "/api/design": "Generate new sequences for the given backbone.",
                "/api/score":  "Score a (structure, sequence) pair via --score_only.",
                "/api/probs":  "Output per-residue AA probabilities (conditional / conditional_backbone / unconditional).",
            },
            "not_in_scope_v0_0_1": (
                "PSSM inputs, batch multi-PDB designs, and score-from-fasta are "
                "deferred to v0.0.2. For now, submit one job per PDB."
            ),
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/design": [
                EndpointExample(
                    title="basic vanilla design",
                    curl=(
                        "curl -X POST $URL/api/design "
                        "-F pdb=@input.pdb "
                        "-F model_variant=vanilla "
                        "-F model_name=v_48_020 "
                        "-F num_seq_per_target=8 "
                        "-F sampling_temp=0.1 "
                        "-F name=run1"
                    ),
                    notes="Output: output/seqs/run1.fa once the job completes.",
                ),
                EndpointExample(
                    title="abmpnn CDR design with fixed framework",
                    curl=(
                        "curl -X POST $URL/api/design "
                        "-F pdb=@antibody.pdb "
                        "-F model_variant=abmpnn "
                        "-F model_name=abmpnn "
                        "-F 'chains_to_design=H L' "
                        "-F 'fixed_positions=1 2 3 4 5, 1 2 3 4 5' "
                        "-F name=cdr_run"
                    ),
                    notes="AbMPNN weights; framework positions fixed, CDRs free to design.",
                ),
            ],
            "/api/score": [
                EndpointExample(
                    title="score complex against vanilla weights",
                    curl=(
                        "curl -X POST $URL/api/score "
                        "-F pdb=@complex.pdb "
                        "-F num_seq_per_target=10 "
                        "-F name=score1"
                    ),
                    notes="Score output: output/score_only/score1_pdb.npz.",
                ),
            ],
            "/api/probs": [
                EndpointExample(
                    title="conditional residue probabilities",
                    curl=(
                        "curl -X POST $URL/api/probs "
                        "-F pdb=@input.pdb "
                        "-F kind=conditional "
                        "-F name=p1"
                    ),
                    notes="Output: output/probs/p1.npz with per-residue 21-vector log probs.",
                ),
            ],
        }
