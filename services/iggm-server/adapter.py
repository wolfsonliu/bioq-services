"""Service-wide policy for iggm-server."""

from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter

from .settings import IgGMSettings


class IgGMAdapter(JobAdapter):
    name = "iggm"

    settings: IgGMSettings  # narrow for IDEs

    def __init__(self, settings: IgGMSettings) -> None:
        super().__init__(settings)

    # ---- Output detection ----

    def detect_outputs(self, job_dir: Path) -> bool:
        """Completed iff the output dir has a real artifact.

        Covers all endpoints: design/fr_design write <name>_<i>.pdb +
        <name>_<i>.fasta, inverse_design writes .fasta only,
        affinity_maturation writes many .pdb/.fasta, epitope writes
        epitope.json.
        """
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        for pattern in ("*.pdb", "*.fasta", "epitope.json"):
            for p in out.glob(pattern):
                if p.stat().st_size > 0:
                    return True
        return False

    # ---- Subprocess env ----

    def subprocess_cwd(self) -> Path | None:
        # Upstream resolves checkpoints from os.getcwd()/checkpoints, which is
        # symlinked to the NAS weights_dir; PYTHONPATH also includes this root.
        return self.settings.root

    # ---- Manifest enrichments (agent protocol) ----

    def manifest_extras(self) -> dict:
        return {
            "model": {
                "name": "IgGM",
                "paper": "Wang et al. ICLR 2025 + bioRxiv 2025.09.12.675771 "
                "(generative foundation model for antibody design)",
                "task": "antibody/nanobody sequence + structure co-design",
                "run_tasks": [
                    "design (CDR sequence + complex structure co-design)",
                    "inverse_design (sequence design on fixed backbone)",
                    "fr_design (framework-region redesign / humanization)",
                    "affinity_maturation (per-position mutation scan)",
                ],
                "output_format": "complex PDB + designed-sequence FASTA "
                "(inverse_design: FASTA only)",
            },
            "tool_outputs": {
                "design": "output/<name>_<i>.pdb (antibody-antigen complex) + "
                "output/<name>_<i>.fasta (designed sequence), one per sample. "
                "inverse_design emits FASTA only.",
                "affinity-maturation": "output/<id>_<pos>_<n>.{pdb,fasta} — "
                "many candidates; count = num_samples x masked positions.",
                "epitope": "output/epitope.json — {epitope: [residue numbers "
                "(1-based)], antigen_id, antigen_length}. Feed epitope back "
                "into /api/design.",
            },
            "input_spec": {
                "fasta": "1-3 chains, antigen LAST; 'X' marks residues to "
                "design. 2 chains = nanobody (H,A); 3 = antibody (H,L,A).",
                "antigen": "antigen structure as PDB (chain id must match the "
                "last FASTA record).",
                "fasta_origin": "affinity-maturation only: the original "
                "antibody sequence to mature from.",
            },
            "config_tips": {
                "epitope": "Optional JSON-encoded int list, e.g. "
                "epitope=[7,8,9]. If omitted, cal_ppi infers it from the "
                "complex structure. Precompute with POST /api/epitope.",
                "max_antigen_size": "Set 384 for large antigens to avoid OOM.",
                "num_samples": "affinity-maturation output scales as "
                "num_samples x masked positions; keep modest on FC.",
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data",
                "uri": "job:// / file:// / oss:// / http(s)://",
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/design": [
                EndpointExample(
                    title="CDR design + structure (antibody vs antigen)",
                    curl=(
                        "curl -X POST $URL/api/design "
                        "-F run_task=design "
                        "-F fasta=@ab_CDR_H3.fasta "
                        "-F antigen=@antigen.pdb "
                        "-F 'epitope=[7,8,9,10,11]'"
                    ),
                    notes="fasta has 'X' at the CDR to design; antigen chain "
                    "id matches the last FASTA record. Poll /api/jobs/<id>, "
                    "then GET /api/jobs/<id>/download.",
                ),
                EndpointExample(
                    title="inverse design (sequence only, fixed backbone)",
                    curl=(
                        "curl -X POST $URL/api/design "
                        "-F run_task=inverse_design "
                        "-F fasta=@complex.fasta -F antigen=@complex.pdb"
                    ),
                    notes="Produces FASTA only.",
                ),
            ],
            "/api/tasks/design": [
                EndpointExample(
                    title="FC async task mode",
                    curl=(
                        "curl -X POST $URL/api/tasks/design "
                        "-H 'X-Fc-Invocation-Type: Async' "
                        "-H 'X-Bioagent-Job-Id: ab-001' "
                        "-F run_task=design -F fasta=@ab.fasta -F antigen=@ag.pdb"
                    ),
                    notes="Returns 202 immediately; preferred production entry.",
                ),
            ],
            "/api/affinity-maturation": [
                EndpointExample(
                    title="affinity maturation",
                    curl=(
                        "curl -X POST $URL/api/affinity-maturation "
                        "-F fasta=@ab_CDR_H3.fasta -F antigen=@antigen.pdb "
                        "-F fasta_origin=@ab_native.fasta -F num_samples=10"
                    ),
                    notes="Requires fasta_origin. Output count scales with "
                    "num_samples x masked positions.",
                ),
            ],
            "/api/epitope": [
                EndpointExample(
                    title="compute interface epitope from a complex",
                    curl=(
                        "curl -X POST $URL/api/epitope "
                        "-F fasta=@complex.fasta -F antigen=@complex.pdb"
                    ),
                    notes="Returns epitope.json with 1-based residue numbers.",
                ),
            ],
        }
