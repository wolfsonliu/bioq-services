"""Service-wide policy for openbpmd-server."""

from __future__ import annotations

from pathlib import Path

from bioagent_service import EndpointExample, JobAdapter

from .settings import OpenBPMDSettings


class OpenBPMDAdapter(JobAdapter):
    name = "openbpmd"

    settings: OpenBPMDSettings  # narrow for IDEs

    def __init__(self, settings: OpenBPMDSettings) -> None:
        super().__init__(settings)

    # ---- Output detection ----

    def detect_outputs(self, job_dir: Path) -> bool:
        """Completed only if the final aggregate results.csv was written.

        Upstream writes per-rep bpm_results.csv incrementally but only calls
        collect_results() (-> results.csv) after ALL reps finish. A truncated
        job leaves rep_*/ dirs but no results.csv -> FAILED. See design §5.
        """
        out = self.output_dir(job_dir) / "results.csv"
        return out.exists() and out.stat().st_size > 0

    # ---- Subprocess env ----

    def subprocess_cwd(self) -> Path | None:
        # The wrapper takes absolute --output; cwd is set to <job_dir>/output by
        # the argv builder's mkdir + our tools, but upstream also writes rep_*/
        # relative to args.output (absolute), so cwd is not load-bearing. Return
        # None to let the runner default to the job dir.
        return None

    # ---- Manifest enrichments (agent protocol) ----

    def manifest_extras(self) -> dict:
        return {
            "model": {
                "name": "OpenBPMD",
                "paper": "Lukauskis et al., J. Chem. Inf. Model. 2022, "
                "62, 6209-6216 (DOI:10.1021/acs.jcim.2c01142)",
                "method": "Binding Pose Metadynamics (OpenMM), Clark et al. "
                "2016 protocol",
                "task": "ligand binding-pose stability scoring / post-docking "
                "re-ranking (NOT a binding free-energy / affinity predictor)",
                "long_running": True,
                "typical_duration": "hours (10 reps x 10 ns, serial, single GPU)",
                "score_semantics": "CompScore = PoseScore - 5*ContactScore; "
                "MORE NEGATIVE = more stable pose. PoseScore = ligand heavy-atom "
                "RMSD (lower better); ContactScore = fraction of native contacts "
                "retained (higher better).",
            },
            "tool_outputs": {
                "results.csv": "final CompScore/PoseScore/ContactScore + SD, "
                "averaged over the last 2 ns of each rep (single row).",
                "rep_*/bpm_results.csv": "time-resolved per-frame scores per replica.",
                "scoring_stats.json": "wrapper summary: nreps_done, comp/pose/"
                "contact score, wall_time_s, platform, sim_ns.",
                "*_system.pdb": "minimized_system.pdb / equil_system.pdb / "
                "centred_equil_system.pdb intermediate structures.",
            },
            "input_requirements": {
                "system": "PRE-SOLVATED + PARAMETRISED complex (Amber "
                ".prm7/.rst7 OR Gromacs .top/.gro). OpenBPMD does NOT prepare "
                "systems — provide ligand FF params, water, and ions upstream.",
                "lig_resname": "must match the ligand residue name in the "
                "topology (default 'MOL').",
            },
            "config_tips": {
                "nreps": "10 is the standard BPMD protocol; reduce for a "
                "faster/cheaper triage at lower confidence.",
                "hill_height": "0.3 kcal/mol is standard; do not change unless "
                "replicating a specific study.",
                "sim_ns": "ADVANCED/TESTING ONLY. Leave unset. Non-standard "
                "lengths break comparability with published BPMD scores.",
                "large_inputs": "Use *_uri (oss:// / job:// / file://) — a "
                "typical .prm7 is ~8 MB and exceeds the FC async 128 KiB "
                "event-payload cap for multipart uploads.",
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data — structure + parameters files",
                "oss://<bucket>/<key>": "fetched at submit-time",
                "job://<prev_job_id>/<file>": "pipeline chaining",
                "file:///<path>": "absolute path on the FC NAS mount",
                "http(s)://...": "streamed at submit-time",
            },
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/api/score": [
                EndpointExample(
                    title="score a docked pose (Amber system)",
                    curl=(
                        "curl -X POST $URL/api/score "
                        "-F structure=@solvated.rst7 "
                        "-F parameters=@solvated.prm7 "
                        "-F lig_resname=MOL -F nreps=10 -F hill_height=0.3"
                    ),
                    notes=(
                        "Long-running (hours on 1 GPU). Returns a JobInfo; poll "
                        "/api/jobs/<id> until completed, then GET "
                        "/api/jobs/<id>/files for results.csv."
                    ),
                ),
                EndpointExample(
                    title="Gromacs system via OSS URIs",
                    curl=(
                        "curl -X POST $URL/api/score "
                        "-F structure_uri=oss://mybucket/solvated.gro "
                        "-F parameters_uri=oss://mybucket/solvated.top "
                        "-F lig_resname=LIG -F nreps=10"
                    ),
                    notes="URI inputs avoid the 128 KiB multipart cap.",
                ),
            ],
            "/api/tasks/score": [
                EndpointExample(
                    title="FC async task mode (single 202 + blocking compute)",
                    curl=(
                        "curl -X POST $URL/api/tasks/score "
                        "-H 'X-Fc-Invocation-Type: Async' "
                        "-H 'X-Bioagent-Job-Id: bpmd-001' "
                        "-F structure_uri=oss://mybucket/solvated.rst7 "
                        "-F parameters_uri=oss://mybucket/solvated.prm7 "
                        "-F lig_resname=MOL -F nreps=10"
                    ),
                    notes=(
                        "Returns 202; FC keeps the instance alive until the "
                        "metadynamics finishes. Duplicate submits with the same "
                        "X-Bioagent-Job-Id are deduped. Prefer URI inputs "
                        "(128 KiB async payload cap)."
                    ),
                ),
            ],
        }
