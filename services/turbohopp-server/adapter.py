"""Service-wide policy for turbohopp-server."""

from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter

from .settings import TurboHoppSettings


class TurboHoppAdapter(JobAdapter):
    name = "turbohopp"

    settings: TurboHoppSettings  # narrow for IDEs

    def __init__(self, settings: TurboHoppSettings) -> None:
        super().__init__(settings)

    # ---- Output detection ----

    def detect_outputs(self, job_dir: Path) -> bool:
        """Job is ``completed`` only if at least one valid .sdf was written."""
        out_dir = self.output_dir(job_dir)
        if not out_dir.exists():
            return False
        for p in out_dir.glob("output_*.sdf"):
            if p.is_file() and p.stat().st_size > 0:
                return True
        return False

    # ---- Subprocess env ----

    def subprocess_cwd(self) -> Path | None:
        # DiffHopp/TurboHopp ObabelTransform / ReduceTransform write temp
        # files relative to CWD, so pin CWD to a writable dir.
        return self.settings.root

    # ---- Manifest enrichments (agent protocol) ----

    def manifest_extras(self) -> dict:
        return {
            "model": {
                "name": "TurboHopp",
                "paper": "arXiv:2410.20660 (NeurIPS 2024)",
                "task": (
                    "scaffold hopping via consistency-model distillation of DiffHopp "
                    "— 30×+ faster sampling at comparable QED/SA quality"
                ),
                "architecture": "GVP backbone + consistency-model distillation",
                "output_format": "SDF (one molecule per file: output_<i>.sdf)",
                "compared_to": (
                    "See services/diffusion-hopping-server (slow, high-quality DiffHopp) "
                    "and services/chembounce-server (CPU, SMILES-only, non-ML)"
                ),
            },
            "tool_outputs": {
                "generate": (
                    "output/output_<i>.sdf — one SDF per generated scaffold "
                    "(i in [0, num_samples)). Files where the generated graph "
                    "couldn't be assembled into a valid Mol are omitted."
                ),
            },
            "config_tips": {
                "num_samples": (
                    "10 is a reasonable default; sampling is batched, so 50 takes "
                    "~4× a single sample, not 50×."
                ),
                "num_sampling_steps": (
                    "Paper default 40 (upper bound for find_best). "
                    "For interactive workflows 5-10 is often sufficient — "
                    "the consistency model stays coherent at very low step counts, "
                    "unlike raw diffusion."
                ),
                "find_best": (
                    "Enables post-hoc QED + normalized-SA rescoring over the last "
                    "num_sampling_steps candidates per graph. Doubles effective "
                    "sample count but noticeably improves realistic-drug-likeness."
                ),
            },
            "input_uri_schemes": {
                "upload": (
                    "multipart/form-data — protein (.pdb) and reference ligand "
                    "(.sdf/.mol2/.pdb)"
                ),
                "job://<job_id>/<filename>": (
                    "chain from a previous job's output on the same NAS "
                    "(e.g. reuse a Boltz-2 output pocket)"
                ),
                "file:///abs/path": "read directly from a pre-staged NAS path",
                "oss://<bucket>/<key>": "fetched at submit-time via configured OSS region",
                "http(s)://...": "generic HTTP(S) download",
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/generate": [
                EndpointExample(
                    title="basic scaffold hopping (10 candidates, 40 steps)",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F protein=@1abc_pocket.pdb "
                        "-F reference_ligand=@ref.sdf "
                        "-F num_samples=10 "
                        "-F num_sampling_steps=40"
                    ),
                    notes=(
                        "Smallest useful call. Returns a JobInfo; poll "
                        "/api/jobs/<id> until completed, then GET "
                        "/api/jobs/<id>/files."
                    ),
                ),
                EndpointExample(
                    title="fast interactive mode (few-step consistency sampling)",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F protein=@pocket.pdb "
                        "-F reference_ligand=@ref.sdf "
                        "-F num_samples=20 "
                        "-F num_sampling_steps=5 "
                        "-F seed=42"
                    ),
                    notes=(
                        "5-step sampling for agent-in-the-loop workflows. "
                        "~10× faster than default at slightly reduced diversity."
                    ),
                ),
                EndpointExample(
                    title="highest-quality mode with post-hoc rescoring",
                    curl=(
                        "curl -X POST $URL/api/generate "
                        "-F protein=@pocket.pdb "
                        "-F reference_ligand=@ref.sdf "
                        "-F num_samples=10 "
                        "-F num_sampling_steps=40 "
                        "-F find_best=true"
                    ),
                    notes=(
                        "Doubles effective compute per candidate; picks QED+SA argmax "
                        "over the last num_sampling_steps intermediate mols."
                    ),
                ),
            ],
            "/api/tasks/generate": [
                EndpointExample(
                    title="FC async task mode (single 202 + blocking compute)",
                    curl=(
                        "curl -X POST $URL/api/tasks/generate "
                        "-H 'X-Fc-Invocation-Type: Async' "
                        "-H 'X-Bioagent-Job-Id: my-job-001' "
                        "-F protein=@pocket.pdb "
                        "-F reference_ligand=@ref.sdf "
                        "-F num_samples=10 "
                        "-F num_sampling_steps=20"
                    ),
                    notes=(
                        "Returns 202 immediately; FC keeps the instance alive "
                        "until sampling completes. Preferred for production behind "
                        "the FC async-task console toggle."
                    ),
                ),
            ],
        }
