"""Service-wide policy for rfantibody-server.

Three quirks over the framework defaults:

  * `detect_outputs` looks for *any* of the three known output filenames so all
    three endpoints share a single success condition without coordinating.
  * `subprocess_cwd` returns the RFantibody source tree, since the scripts
    expect to be launched from there (Hydra picks up its config dirs that way).
  * `infer_job_from_dir` annotates recovered jobs with the step that left the
    most-recent output behind.
"""

from __future__ import annotations

from pathlib import Path

from bioq_service import EndpointExample, JobAdapter, JobInfo, JobStatus

from .settings import RFantibodySettings
from .tools import (
    ALL_OUTPUT_FILENAMES,
    PROTEINMPNN_OUTPUT,
    RF2_OUTPUT,
    RFDIFFUSION_OUTPUT,
)


class RFantibodyAdapter(JobAdapter):
    name = "rfantibody"

    settings: RFantibodySettings  # narrow the type for IDE help

    def __init__(self, settings: RFantibodySettings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        """COMPLETED if any of the three known artifact files exists + non-empty."""
        for name in ALL_OUTPUT_FILENAMES:
            f = job_dir / "output" / name
            if f.exists() and f.stat().st_size > 0:
                return True
        return False

    def subprocess_cwd(self) -> Path | None:
        """RFantibody scripts use Hydra; they assume cwd == project root."""
        return self.settings.root

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        """Three working examples that mirror the recommended chaining pattern."""
        return {
            "/api/rfdiffusion": [
                EndpointExample(
                    title="basic backbone design",
                    curl=(
                        "curl -X POST $URL/api/rfdiffusion "
                        "-F target=@antigen.pdb "
                        "-F framework=@framework.pdb "
                        "-F num_designs=100 "
                        "-F 'design_loops=H1:7,H2:6,H3:5-13' "
                        "-F 'hotspots=B146,B170,B177'"
                    ),
                    python=(
                        "import httpx\n"
                        "with open('antigen.pdb','rb') as t, open('framework.pdb','rb') as f:\n"
                        "    r = httpx.post(\n"
                        "        f'{base_url}/api/rfdiffusion',\n"
                        "        files={'target': t, 'framework': f},\n"
                        "        data={'num_designs': 100, 'hotspots': 'B146,B170,B177'},\n"
                        "        timeout=120,\n"
                        "    )\n"
                        "job_id = r.json()['job_id']"
                    ),
                    notes="Both PDB files are required uploads. Returns JobInfo immediately.",
                ),
            ],
            "/api/proteinmpnn": [
                EndpointExample(
                    title="chain from a previous rfdiffusion job",
                    curl=(
                        "curl -X POST $URL/api/proteinmpnn "
                        "-F 'input_quiver_uri=job://abc123def456/1_rfdiffusion.qv' "
                        "-F seqs_per_struct=4"
                    ),
                    python=(
                        "r = httpx.post(\n"
                        "    f'{base_url}/api/proteinmpnn',\n"
                        "    data={\n"
                        "        'input_quiver_uri': f'job://{rfdiffusion_job_id}/1_rfdiffusion.qv',\n"
                        "        'seqs_per_struct': 4,\n"
                        "    },\n"
                        ")"
                    ),
                    notes=(
                        "Preferred path: cite the previous job's output via job:// rather "
                        "than re-uploading the Quiver. The NAS is shared so this is a "
                        "zero-copy operation."
                    ),
                ),
                EndpointExample(
                    title="upload a Quiver file directly",
                    curl=(
                        "curl -X POST $URL/api/proteinmpnn "
                        "-F input_quiver=@1_rfdiffusion.qv "
                        "-F seqs_per_struct=4"
                    ),
                    notes="Fallback when the upstream job isn't accessible on this NAS.",
                ),
            ],
            "/api/rf2": [
                EndpointExample(
                    title="chain from a previous proteinmpnn job",
                    curl=(
                        "curl -X POST $URL/api/rf2 "
                        "-F 'input_quiver_uri=job://xyz789ghi012/2_proteinmpnn.qv' "
                        "-F num_recycles=10"
                    ),
                    python=(
                        "r = httpx.post(\n"
                        "    f'{base_url}/api/rf2',\n"
                        "    data={\n"
                        "        'input_quiver_uri': f'job://{proteinmpnn_job_id}/2_proteinmpnn.qv',\n"
                        "        'num_recycles': 10,\n"
                        "    },\n"
                        ")"
                    ),
                    notes="Final step; output is 3_rf2.qv with binder + target structures + scores.",
                ),
            ],
        }

    def manifest_extras(self) -> dict:
        """Service-specific protocol knowledge surfaced on `/api/manifest`.

        Tells an agent (1) where each tool's output lands, (2) which URI schemes
        the upload-or-fetch helpers accept, and (3) the most efficient way to
        chain tools across this service's own jobs.
        """
        return {
            "tool_outputs": {
                "rfdiffusion": "output/" + RFDIFFUSION_OUTPUT,
                "proteinmpnn": "output/" + PROTEINMPNN_OUTPUT,
                "rf2": "output/" + RF2_OUTPUT,
            },
            "input_uri_schemes": {
                "upload": "multipart/form-data UploadFile (target / framework / input_quiver)",
                "job://<job_id>/<filename>": "Pull a file from a previous job on this service's NAS",
                "file:///abs/path": "Direct NAS path (works across services on the shared mount)",
                "oss://<bucket>/<key>": "Alibaba Cloud OSS object",
                "http(s)://...": "Generic HTTP(S) URL — incl. signed OSS URLs",
            },
            "chaining_tip": (
                "Run /api/rfdiffusion first, then call /api/proteinmpnn with "
                "input_quiver_uri=job://<rfdiffusion_job_id>/1_rfdiffusion.qv. Same pattern "
                "for /api/rf2 with input_quiver_uri=job://<proteinmpnn_job_id>/2_proteinmpnn.qv. "
                "Avoids re-uploading multi-MB Quiver files since the NAS is shared."
            ),
            "weights": {
                "rfdiffusion": "RFdiffusion_Ab.pt",
                "proteinmpnn": "ProteinMPNN_v48_noise_0.2.pt",
                "rf2": "RF2_ab.pt",
                "note": "Weights are baked into the Docker image at <weights_dir>/. Missing files are tolerated (script defaults apply).",
            },
        }

    def infer_job_from_dir(self, job_dir: Path) -> JobInfo:
        """Surface which step's output is on disk (rf2 > proteinmpnn > rfdiffusion)."""
        job_id = job_dir.name
        progress: str | None = None
        # Latest step first — that's the one whose output represents the furthest
        # the pipeline got.
        for name, step_label in (
            (RF2_OUTPUT, "rf2"),
            (PROTEINMPNN_OUTPUT, "proteinmpnn"),
            (RFDIFFUSION_OUTPUT, "rfdiffusion"),
        ):
            f = job_dir / "output" / name
            if f.exists() and f.stat().st_size > 0:
                progress = step_label
                break

        if progress is not None:
            return JobInfo(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                progress=progress,
                message=f"Recovered from disk (last step: {progress})",
            )
        return JobInfo(
            job_id=job_id,
            status=JobStatus.FAILED,
            message="Recovered from disk (no outputs)",
        )
