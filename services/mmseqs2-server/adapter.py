"""MMseqs2 service adapter.

Detects ColabFold-style `.a3m` outputs under `<job_dir>/output/`, exposes a
manifest describing the four ColabFold-protocol endpoints, and forwards the
caller's `CUDA_VISIBLE_DEVICES` (if any) to the subprocess so single-GPU FC
instances don't need to be re-baked when the visible-devices list changes.
"""

from __future__ import annotations

import os
from pathlib import Path

from bioagent_service import EndpointExample, JobAdapter

from .settings import MMseqs2Settings


class MMseqs2JobAdapter(JobAdapter):
    """JobAdapter for the vendored ColabFold MSA orchestrator.

    Outputs live under `<job_dir>/output/` as one or more `.a3m` files:
      - Monomer (unpaired): `<safe_jobname>.a3m` (one per FASTA record)
      - Multimer paired:    `<safe_jobname>_paired.a3m` plus per-chain a3m's
        depending on which step in the orchestrator emitted them.

    Either way, ``rglob("*.a3m")`` finds them — see ``orchestrator.main`` lines
    ~1166-1191 for the rename / emit logic.
    """

    name = "mmseqs2"

    settings: MMseqs2Settings  # narrow for IDEs

    def __init__(self, settings: MMseqs2Settings) -> None:
        super().__init__(settings)

    def detect_outputs(self, job_dir: Path) -> bool:
        out = self.output_dir(job_dir)
        if not out.is_dir():
            return False
        for path in out.rglob("*.a3m"):
            try:
                if path.stat().st_size > 0:
                    return True
            except OSError:
                continue
        return False

    def subprocess_env(self) -> dict[str, str]:
        """Forward CUDA_VISIBLE_DEVICES from caller env if set.

        The mmseqs binary is already on PATH via the Dockerfile
        (`/opt/mmseqs-gpu/bin`), so no PATH munging is required. We only need
        to make sure the orchestrator subprocess sees the same GPU visibility
        as the FastAPI worker that spawned it.
        """
        env: dict[str, str] = {}
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda_devices
        return env

    def manifest_extras(self) -> dict:
        return {
            "protocol": (
                "ColabFold MSA HTTP protocol — four endpoints "
                "(POST /ticket/msa, POST /ticket/pair, GET /ticket/<id>, "
                "GET /result/download/<id>). All errors are returned as "
                "{'status': 'ERROR'} with HTTP 200 (the ColabFold client "
                "treats 4xx as fatal)."
            ),
            "modes": {
                "env":              "UniRef30 + ColabFoldDB env, with filter (monomer).",
                "all":              "UniRef30 only, with filter (monomer).",
                "env-nofilter":     "UniRef30 + env, no filter (monomer).",
                "nofilter":         "UniRef30 only, no filter (monomer).",
                "pairgreedy":       "Paired multimer, pairaln strategy=0 (greedy per species).",
                "paircomplete":     "Paired multimer, pairaln strategy=1 (all chains required).",
                "pairgreedy-env":   "pairgreedy + env DB.",
                "paircomplete-env": "paircomplete + env DB.",
            },
            "tool_outputs": {
                "msa": (
                    "<job_dir>/output/<safe_jobname>.a3m — one per FASTA record "
                    "in the query. Multimer paired runs additionally emit "
                    "<safe_jobname>_paired.a3m and per-chain auxiliary files."
                ),
            },
            "endpoints_summary": {
                "/ticket/msa":            "Submit a monomer (unpaired) MSA job (ColabFold protocol).",
                "/ticket/pair":           "Submit a multimer (paired) MSA job (ColabFold protocol).",
                "/ticket/{id}":           "Poll job status (PENDING/RUNNING/COMPLETE/ERROR).",
                "/result/download/{id}":  "Download a tar.gz of *.a3m output files.",
                "/api/tasks/msa":         "Monomer MSA as an atomic task; blocks until completion (FC async task mode).",
                "/api/tasks/pair":        "Multimer paired MSA as an atomic task; blocks until completion (FC async task mode).",
                "/healthz/detail":        "Extended health (db_loaded, gpu_free_mb, active_jobs).",
            },
            "limits": {
                "max_residues_per_chain": 1023,
                "valid_alphabet":         "20 standard AAs + X + *",
            },
            "not_in_scope_v0_0_1": (
                "Templates, AF3-JSON output, custom DBs, MSA caching, "
                "rate-limit / maintenance responses (RATELIMIT / MAINTENANCE "
                "are part of the protocol for forward compatibility but the "
                "server never emits them in v0.0.1)."
            ),
        }

    def endpoint_examples(self) -> dict[str, list[EndpointExample]]:
        return {
            "/ticket/msa": [
                EndpointExample(
                    title="monomer MSA (UniRef30 + env)",
                    curl=(
                        "curl -X POST $URL/ticket/msa "
                        "-F 'q=>query1\\nMKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEV' "
                        "-F mode=env"
                    ),
                    notes=(
                        "Returns {'id': '<job_id>', 'status': 'PENDING'}. Poll "
                        "GET /ticket/<id> until status is COMPLETE then GET "
                        "/result/download/<id> for the .a3m tarball."
                    ),
                ),
            ],
            "/ticket/pair": [
                EndpointExample(
                    title="multimer paired MSA (greedy)",
                    curl=(
                        "curl -X POST $URL/ticket/pair "
                        "-F 'q=>chainA\\nMKQHKAM...\\n>chainB\\nLLLLLLL...' "
                        "-F mode=pairgreedy"
                    ),
                    notes=(
                        "Requires at least 2 FASTA records. Use pairgreedy / "
                        "paircomplete (strategy 0 / 1). The -env variants add "
                        "the ColabFoldDB environmental DB on top."
                    ),
                ),
            ],
            "/api/tasks/msa": [
                EndpointExample(
                    title="monomer MSA via FC async task mode",
                    curl=(
                        "curl -X POST $URL/api/tasks/msa "
                        "-H 'X-Fc-Invocation-Type: Async' "
                        "-H 'X-Bioagent-Job-Id: mmseqs-$(date +%s)' "
                        "-H 'X-Fc-Async-Task-Id: mmseqs-$(date +%s)' "
                        "-F 'q=>query1\\nMKQHKAMIVALIVICITAVVAAL...' "
                        "-F mode=env"
                    ),
                    notes=(
                        "Returns HTTP 202 immediately; FC enqueues the task and "
                        "the function runs synchronously inside the instance. "
                        "Poll JobInfo via the framework's GET /api/jobs/{id}, "
                        "or use FCDispatcher.get_status for production "
                        "(HTTP polling is rate-limit-prone — see "
                        "engineering/decisions/2026-06-17-fc-async-task-mode.md)."
                    ),
                ),
            ],
            "/api/tasks/pair": [
                EndpointExample(
                    title="multimer paired MSA via FC async task mode",
                    curl=(
                        "curl -X POST $URL/api/tasks/pair "
                        "-H 'X-Fc-Invocation-Type: Async' "
                        "-H 'X-Bioagent-Job-Id: mmseqs-$(date +%s)' "
                        "-H 'X-Fc-Async-Task-Id: mmseqs-$(date +%s)' "
                        "-F 'q=>chainA\\nMKQHKAM...\\n>chainB\\nLLLLLLL...' "
                        "-F mode=pairgreedy"
                    ),
                    notes=(
                        "Same task-mode semantics as /api/tasks/msa, but for "
                        "paired multimer modes. Returns JobInfo (200 + "
                        "status=completed/failed) — distinct from the "
                        "ColabFold-protocol 200 + {status: ERROR} convention "
                        "of /ticket/*."
                    ),
                ),
            ],
        }


__all__ = ["MMseqs2JobAdapter"]
