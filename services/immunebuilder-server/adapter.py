"""ImmuneBuilder service adapter.

Detects outputs (final_model.pdb), contributes the service-specific manifest
section with predictor/output documentation.
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter

from .settings import ImmuneBuilderSettings


class ImmuneBuilderAdapter(JobAdapter):
    name = "immunebuilder"

    settings: ImmuneBuilderSettings

    def __init__(self, settings: ImmuneBuilderSettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        output = self.output_dir(job_dir)
        if not output.is_dir():
            return False
        return (output / "final_model.pdb").is_file()

    def manifest_extras(self) -> dict:
        return {
            "tool_outputs": {
                "final_model": "output/final_model.pdb",
                "unrefined_models": "output/rank{0-3}_unrefined.pdb (save_all_models=True)",
                "error_estimates": "output/error_estimates.npy (save_all_models=True)",
            },
            "predictors": {
                "antibody": "ABodyBuilder2 — H + L chains",
                "nanobody": "NanoBodyBuilder2 — H chain only",
                "tcr": "TCRBuilder2 — A + B chains",
            },
            "numbering_schemes": [
                "imgt", "chothia", "kabat", "aho", "wolfguy", "martin", "raw",
            ],
            "paper_reference": "DOI:10.1038/s42003-023-04927-7",
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/predict_antibody": [
                EndpointExample(
                    title="predict antibody structure",
                    curl=(
                        "curl -X POST $URL/api/predict_antibody "
                        "-F heavy_sequence=EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS "
                        "-F light_sequence=DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK "
                        "-F name=ab1"
                    ),
                    notes="Output: final_model.pdb + optional rank*_unrefined.pdb + error_estimates.npy",
                ),
            ],
            "/api/predict_nanobody": [
                EndpointExample(
                    title="predict nanobody structure",
                    curl=(
                        "curl -X POST $URL/api/predict_nanobody "
                        "-F heavy_sequence=QVQLVESGGGLVQAGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAINSGGGSTYYPDSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAKDGYGGSFDYWGQGTQVTVSS "
                        "-F name=nb1"
                    ),
                ),
            ],
            "/api/predict_tcr": [
                EndpointExample(
                    title="predict TCR structure",
                    curl=(
                        "curl -X POST $URL/api/predict_tcr "
                        "-F alpha_sequence=METLLGVSLVILWLQLARVNSQQGEEDPQALSIQEGENATMNCSYKTSINNLQWYRQNSGRGLVHLILIRSNEREKHSGRLRVTLDTSKKSSSLLITASRAADTASYFCAVNFGGGKLIFGQGTELSVKP "
                        "-F beta_sequence=NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSFSTCSANYGYTFGSGTRLTVVEDL "
                        "-F name=tcr1"
                    ),
                ),
            ],
        }
