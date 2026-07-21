"""Service-wide policy for chembounce-server."""

from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter

from .settings import ChemBounceSettings


class ChemBounceAdapter(JobAdapter):
    name = "chembounce"

    settings: ChemBounceSettings  # narrow for IDEs

    def __init__(self, settings: ChemBounceSettings) -> None:
        super().__init__(settings)

    # ---- Output detection ----

    def detect_outputs(self, job_dir: Path) -> bool:
        """Completed iff `overall_result.txt` exists and is non-empty.

        Upstream writes per-fragment intermediates regardless of success;
        only the final `overall_result.txt` is the authoritative success
        signal.
        """
        result = self.output_dir(job_dir) / "overall_result.txt"
        return result.exists() and result.stat().st_size > 0

    # ---- Subprocess env ----

    def subprocess_cwd(self) -> Path | None:
        # Upstream chembounce.py does `sys.path.append(os.path.dirname(__file__))`
        # then imports relative modules.  cwd does NOT have to match upstream
        # source dir for that to work, but setting it there is the safest bet.
        return self.settings.root

    # ---- Manifest enrichments (agent protocol) ----

    def manifest_extras(self) -> dict:
        return {
            "model": {
                "name": "ChemBounce",
                "method": "classical scaffold hopping (fragmentation + similarity search + drug-likeness filtering)",
                "task": "ligand-based scaffold hopping (no pocket / protein target required)",
                "output_format": "TSV (overall_result.txt) + per-fragment TSVs + resource_cost.json",
                "license_note": "INTERNAL RESEARCH USE ONLY — upstream repo has no LICENSE file. See design doc §10.1.",
            },
            "tool_outputs": {
                "scaffold_hop": (
                    "output/overall_result.txt — main TSV with columns: Fragment_no, "
                    "Final structure, Standardized final structure, Tanimoto Similarity, "
                    "Electron shape Similarity"
                ),
            },
            "config_tips": {
                "database": (
                    "Start with '250mw' (default — scaffolds with MW ≤ 250, fast). "
                    "Switch to 'full' only for comprehensive paper-grade search "
                    "(~4M scaffolds, needs ≥64 GB RAM)."
                ),
                "frag_max_n": (
                    "100 is a sensible default.  >1000 stresses runtime "
                    "exponentially on multi-ring molecules (upstream caveat)."
                ),
                "tanimoto_threshold": (
                    "0.5 retains broad diversity.  Raise to 0.6–0.7 for "
                    "tighter similarity to the original molecule."
                ),
                "wo_lipinski": (
                    "Enable for macrocycles / peptides that naturally violate "
                    "Lipinski's rule of five (MW > 500, > 5 H-bond donors, etc)."
                ),
                "core_smiles": (
                    "Use to lock a substructure (e.g. a known pharmacophore) "
                    "that must be preserved in the final candidates."
                ),
            },
            "input_uri_schemes": {
                "string": "SMILES passed directly as a form field (recommended)",
            },
            "task_comparison": {
                "vs_diffusion_hopping_server": (
                    "Both do 'scaffold hopping' but differ fundamentally: "
                    "ChemBounce is ligand-based (no protein required, classical "
                    "similarity search); diffusion-hopping-server is "
                    "pocket-conditional (requires protein PDB + reference ligand, "
                    "neural diffusion preserves binding pose).  Pick ChemBounce "
                    "for early hit-to-lead medicinal chemistry, anti-patent "
                    "redesign, or when no protein structure is available."
                ),
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        # SMILES is the losartan-like example from upstream README.
        losartan_smiles = "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl"
        return {
            "/api/scaffold_hop": [
                EndpointExample(
                    title="basic scaffold hopping (250 MW DB, default thresholds)",
                    curl=(
                        f"curl -X POST $URL/api/scaffold_hop "
                        f"-F 'input_smiles={losartan_smiles}' "
                        f"-F 'frag_max_n=100' "
                        f"-F 'tanimoto_threshold=0.5'"
                    ),
                    notes=(
                        "Smallest useful call.  Returns a JobInfo; poll "
                        "/api/jobs/<id> until completed, then GET "
                        "/api/jobs/<id>/file/output/overall_result.txt."
                    ),
                ),
                EndpointExample(
                    title="full DB + tight Tanimoto + core lock",
                    curl=(
                        f"curl -X POST $URL/api/scaffold_hop "
                        f"-F 'input_smiles={losartan_smiles}' "
                        f"-F 'database=full' "
                        f"-F 'tanimoto_threshold=0.7' "
                        f"-F 'core_smiles=C4=NNN=N4' "
                        f"-F 'frag_max_n=50'"
                    ),
                    notes=(
                        "Lock the tetrazole core; search the full 4M-scaffold "
                        "DB.  Significantly more memory + time than the default."
                    ),
                ),
                EndpointExample(
                    title="macrocycle / peptide (Lipinski off)",
                    curl=(
                        "curl -X POST $URL/api/scaffold_hop "
                        "-F 'input_smiles=<peptide SMILES>' "
                        "-F 'wo_lipinski=true' "
                        "-F 'mw_max=4000'"
                    ),
                    notes=(
                        "wo_lipinski disables the rule-of-five gate; mw_max "
                        "still enforced because peptide candidates can exceed "
                        "5 kDa."
                    ),
                ),
            ],
            "/api/tasks/scaffold_hop": [
                EndpointExample(
                    title="FC async task mode",
                    curl=(
                        f"curl -X POST $URL/api/tasks/scaffold_hop "
                        f"-H 'X-Fc-Invocation-Type: Async' "
                        f"-H 'X-Bioagent-Job-Id: my-hop-001' "
                        f"-F 'input_smiles={losartan_smiles}' "
                        f"-F 'frag_max_n=100'"
                    ),
                    notes=(
                        "Returns 202 immediately; FC keeps the instance alive "
                        "until the search completes.  Use in production behind "
                        "the FC async-task console toggle."
                    ),
                ),
            ],
        }
