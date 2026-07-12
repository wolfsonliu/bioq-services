#!/usr/bin/env python3
"""Unified concurrency benchmark for all bioagent FC services.

Usage:
    uv run python services/bench_concurrency.py --service boltz-server --concurrency 20
    uv run python services/bench_concurrency.py --service rfdiffusion-server -c 10 -n 3
    uv run python services/bench_concurrency.py --list

Measures:
  1. Single-job baseline latency (cold + warm)
  2. Concurrent submission behavior
  3. Optional sequential throughput
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# FC URL + session header
# ---------------------------------------------------------------------------

SESSION_HEADER = "bioagent-session-id"
_SERVICES_DIR = Path(__file__).resolve().parent


def _fc_url(service: str) -> str:
    from bioagent_service.service_registry import fc_url

    return fc_url(service, start=Path(__file__))


# ---------------------------------------------------------------------------
# Per-service configurations
# ---------------------------------------------------------------------------


def _read_file(rel_path: str) -> bytes:
    return (_SERVICES_DIR / rel_path).read_bytes()


def _service_configs() -> dict[str, dict]:
    """Return {service_name: {endpoint, data, files?}} for each service."""

    short_protein = "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSC"
    nanobody_seq = (
        "QVQLVESGGGLVQAGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAINSGGGST"
        "YYPDSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAKDGYGGSFDYWGQGTQVTVSS"
    )
    antibody_heavy = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGY"
        "TRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
    )
    antibody_light = (
        "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSG"
        "VPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
    )

    return {
        "boltz-server": {
            "endpoint": "/api/predict_structure",
            "data": {
                "name": "bench",
                "msa_mode": "empty",
                "diffusion_samples": "1",
                "recycling_steps": "1",
                "sampling_steps": "50",
                "sequences": json.dumps([
                    {"type": "protein", "id": "A", "sequence": short_protein,
                     "msa_uri": "empty"},
                ]),
            },
        },
        "genie3-server": {
            "endpoint": "/api/generate/unconditional",
            "data": {
                "n_sample": "1",
                "batch_size": "1",
                "min_length": "50",
                "max_length": "50",
                "length_step": "50",
            },
        },
        "rfdiffusion-server": {
            "endpoint": "/api/generate/unconditional",
            "data": {
                "num_designs": "1",
                "diffuser_t": "25",
                "min_length": "60",
                "max_length": "60",
            },
        },
        "rfdiffusion2-server": {
            "endpoint": "/api/generate/small_molecule_binder",
            "data": {
                "contigs": "50",
                "length": "50-50",
                "ligand": "PH2",
                "rasa_active": "true",
                "rasa_target": "0",
                "num_designs": "1",
                "diffuser_t": "10",
            },
            "files_lazy": lambda: {
                "input_pdb": (
                    "ligand.pdb",
                    _read_file("rfdiffusion2-server/tests/data/"
                               "trimmed_ec2_M0151_NO_ORI_zero_com0.pdb"),
                    "chemical/x-pdb",
                ),
            },
        },
        "ppiflow-server": {
            "endpoint": "/api/sample/monomer",
            "data": {
                "samples_per_target": "1",
                "length_subset": "[80]",
            },
        },
        "rfantibody-server": {
            "endpoint": "/api/rfdiffusion",
            "data": {
                "num_designs": "1",
                "diffuser_t": "25",
                "design_loops": "L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13",
                "hotspots": "T305,T456",
                "deterministic": "true",
            },
            "files_lazy": lambda: {
                "target": (
                    "rsv_site3.pdb",
                    _read_file("rfantibody-server/tests/data/rsv_site3.pdb"),
                    "chemical/x-pdb",
                ),
                "framework": (
                    "hu-4D5-8_Fv.pdb",
                    _read_file("rfantibody-server/tests/data/hu-4D5-8_Fv.pdb"),
                    "chemical/x-pdb",
                ),
            },
        },
        "proteinmpnn-server": {
            "endpoint": "/api/design",
            "data": {
                "name": "bench",
                "model_variant": "vanilla",
                "model_name": "v_48_020",
                "num_seq_per_target": "1",
            },
            "files_lazy": lambda: {
                "pdb": (
                    "5L33.pdb",
                    _read_file("proteinmpnn-server/tests/data/5L33.pdb"),
                    "chemical/x-pdb",
                ),
            },
        },
        "immunebuilder-server": {
            "endpoint": "/api/predict_nanobody",
            "data": {
                "heavy_sequence": nanobody_seq,
            },
        },
        "esmfold2-server": {
            "endpoint": "/api/fold",
            "data": {
                "sequences": json.dumps([
                    {"type": "protein", "id": "A", "sequence": short_protein},
                ]),
                "num_loops": "1",
                "num_sampling_steps": "10",
                "num_diffusion_samples": "1",
            },
        },
        "dockq-server": {
            "endpoint": "/api/score",
            "data": {"name": "bench"},
            "files_lazy": lambda: {
                "model": (
                    "model.pdb",
                    _read_file("dockq-server/tests/data/model.pdb"),
                    "chemical/x-pdb",
                ),
                "native": (
                    "native.pdb",
                    _read_file("dockq-server/tests/data/native.pdb"),
                    "chemical/x-pdb",
                ),
            },
        },
        "deeprank-ab-server": {
            "endpoint": "/api/score",
            "data": {
                "heavy_chain_id": "H",
                "light_chain_id": "L",
                "antigen_chain_id": "A",
            },
            "files_lazy": lambda: {
                "input_pdb": (
                    "test.pdb",
                    _read_file("deeprank-ab-server/tests/data/test.pdb"),
                    "chemical/x-pdb",
                ),
            },
        },
        "boltzgen-server": {
            "endpoint": "/api/design",
            "data": {
                "protocol": "protein-anything",
                "num_designs": "10",
                "budget": "5",
            },
            "files_lazy": lambda: {
                "design_yaml": (
                    "vanilla.yaml",
                    _read_file("boltzgen-server/tests/data/vanilla.yaml"),
                    "application/x-yaml",
                ),
            },
        },
        "alphafold-server": {
            "endpoint": "/api/fold",
            "data": {
                "models_to_relax": "none",
            },
            "files_lazy": lambda: {
                "input_fasta": (
                    "bench.fasta",
                    b">A\nMKTAYIAKQRQISFVKSHFSRQLE\n",
                    "text/plain",
                ),
            },
        },
    }


# ---------------------------------------------------------------------------
# Poll + submit helpers
# ---------------------------------------------------------------------------


def _poll_job(
    client: httpx.Client,
    base_url: str,
    job_id: str,
    timeout_s: int = 1800,
    interval_s: int = 15,
    extra_headers: dict[str, str] | None = None,
    tag: str = "",
) -> dict:
    deadline = time.monotonic() + timeout_s
    hdrs = extra_headers or {}
    body: dict = {}
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            resp = client.get(f"{base_url}/api/jobs/{job_id}", headers=hdrs)
            resp.raise_for_status()
            body = resp.json()
        except Exception as exc:
            print(f"    [{tag}] poll #{attempt} exception: {exc!r}")
            time.sleep(interval_s)
            continue
        status = body.get("status")
        if attempt <= 3 or status in ("completed", "failed"):
            print(f"    [{tag}] poll #{attempt} status={status} duration={body.get('duration_seconds')}")
        if status in ("completed", "failed"):
            if status == "failed":
                tail = body.get("error_tail") or ""
                print(f"    [{tag}] FAILED: {body.get('error_summary')} | "
                      f"kind={body.get('failure_kind')} | tail={tail[:200]}")
            return body
        time.sleep(interval_s)
    raise TimeoutError(
        f"job {job_id} did not finish within {timeout_s}s; last: {body.get('status')}"
    )


@dataclass
class JobResult:
    job_id: str
    submit_status: int
    submit_time: float
    total_time: float | None = None
    final_status: str | None = None
    duration_server: float | None = None
    error: str | None = None


@dataclass
class BenchResult:
    label: str
    jobs: list[JobResult] = field(default_factory=list)
    wall_time: float = 0.0


def submit_and_poll(
    base_url: str,
    endpoint: str,
    data: dict,
    files: dict | None = None,
    timeout_s: int = 1800,
    tag: str = "",
) -> JobResult:
    t0 = time.monotonic()
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(180.0)) as c:
        r = c.post(endpoint, data=data, files=files)
        submit_time = time.monotonic() - t0

        if r.status_code != 200:
            return JobResult(
                job_id="",
                submit_status=r.status_code,
                submit_time=submit_time,
                total_time=time.monotonic() - t0,
                final_status=f"submit_failed_{r.status_code}",
                error=r.text[:300],
            )

        job_id = r.json()["job_id"]
        session_hdrs: dict[str, str] = {}
        session_val = r.headers.get(SESSION_HEADER)
        if session_val:
            session_hdrs[SESSION_HEADER] = session_val
        print(f"  [{tag}] submitted {job_id} in {submit_time:.1f}s (session={session_val})")

        final = _poll_job(
            c, base_url, job_id,
            timeout_s=timeout_s, interval_s=10,
            extra_headers=session_hdrs, tag=tag,
        )
        total = time.monotonic() - t0

        if final["status"] == "failed":
            try:
                log_r = c.get(f"{base_url}/api/jobs/{job_id}/log", headers=session_hdrs)
                log_text = log_r.json().get("log", "") if log_r.status_code == 200 else ""
                if log_text:
                    print(f"  [{tag}] LOG (last 500 chars):\n{log_text[-500:]}")
            except Exception:
                pass

        return JobResult(
            job_id=job_id,
            submit_status=r.status_code,
            submit_time=submit_time,
            total_time=total,
            final_status=final["status"],
            duration_server=final.get("duration_seconds"),
            error=final.get("error_summary"),
        )


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def run_sequential(
    base_url: str, endpoint: str, data: dict, files: dict | None, n: int,
) -> BenchResult:
    result = BenchResult(label=f"sequential x{n}")
    t0 = time.monotonic()
    for i in range(n):
        jr = submit_and_poll(base_url, endpoint, data, files, tag=f"seq-{i+1}/{n}")
        result.jobs.append(jr)
        print(f"  [seq-{i+1}] {jr.final_status} total={jr.total_time:.1f}s "
              f"server={jr.duration_server}s")
    result.wall_time = time.monotonic() - t0
    return result


def run_concurrent(
    base_url: str, endpoint: str, data: dict, files: dict | None, n: int,
) -> BenchResult:
    result = BenchResult(label=f"concurrent x{n}")
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {
            pool.submit(
                submit_and_poll, base_url, endpoint, data, files,
                1800, f"par-{i+1}/{n}",
            ): i
            for i in range(n)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                jr = fut.result()
            except Exception as e:
                jr = JobResult(
                    job_id="", submit_status=-1, submit_time=0,
                    error=str(e), final_status="exception",
                )
            result.jobs.append(jr)
            print(f"  [par-{idx+1}] {jr.final_status} "
                  f"total={jr.total_time:.1f}s submit_rc={jr.submit_status}")
    result.wall_time = time.monotonic() - t0
    return result


def print_report(results: list[BenchResult]) -> None:
    print("\n" + "=" * 70)
    print("BENCHMARK REPORT")
    print("=" * 70)
    for r in results:
        print(f"\n--- {r.label} (wall: {r.wall_time:.1f}s) ---")
        succeeded = [j for j in r.jobs if j.final_status == "completed"]
        rejected = [j for j in r.jobs if j.submit_status == 503]
        failed = [
            j for j in r.jobs
            if j.final_status not in ("completed", None) and j.submit_status != 503
        ]

        if succeeded:
            server_times = [j.duration_server for j in succeeded if j.duration_server]
            total_times = [j.total_time for j in succeeded if j.total_time]
            print(f"  completed: {len(succeeded)}/{len(r.jobs)}")
            if server_times:
                print(f"  server duration: min={min(server_times):.1f}s "
                      f"avg={sum(server_times)/len(server_times):.1f}s "
                      f"max={max(server_times):.1f}s")
            if total_times:
                print(f"  e2e latency:     min={min(total_times):.1f}s "
                      f"avg={sum(total_times)/len(total_times):.1f}s "
                      f"max={max(total_times):.1f}s")
            if len(succeeded) > 1 and r.wall_time > 0:
                print(f"  throughput: {len(succeeded)/r.wall_time*3600:.1f} jobs/hour")
        if rejected:
            print(f"  rejected (503): {len(rejected)}/{len(r.jobs)}")
            for j in rejected:
                print(f"    {j.error[:100] if j.error else '(no detail)'}")
        if failed:
            print(f"  failed: {len(failed)}/{len(r.jobs)}")
            for j in failed:
                print(f"    {j.job_id}: {j.final_status} — {j.error or ''}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    configs = _service_configs()

    parser = argparse.ArgumentParser(
        description="Unified bioagent FC concurrency benchmark",
    )
    parser.add_argument(
        "--service", "-s",
        choices=sorted(configs.keys()),
        help="Service to benchmark",
    )
    parser.add_argument("--concurrency", "-c", type=int, default=3)
    parser.add_argument("--jobs", "-n", type=int, default=0,
                        help="Sequential jobs after concurrent phase (0=skip)")
    parser.add_argument("--list", action="store_true",
                        help="List available services and exit")
    args = parser.parse_args()

    if args.list:
        print("Available services:")
        for name, cfg in sorted(configs.items()):
            needs_files = "files_lazy" in cfg
            print(f"  {name:30s}  {cfg['endpoint']:40s}  "
                  f"{'(needs test data files)' if needs_files else ''}")
        return

    if not args.service:
        parser.error("--service is required (use --list to see options)")

    cfg = configs[args.service]
    endpoint = cfg["endpoint"]
    data = cfg["data"]
    files = cfg["files_lazy"]() if "files_lazy" in cfg else None

    base_url = _fc_url(args.service)
    print(f"Target: {base_url}")
    print(f"Service: {args.service}")
    print(f"Endpoint: {endpoint}")
    print(f"Concurrency: {args.concurrency}, Sequential: {args.jobs}")

    # Warm-up / healthcheck
    print("\nWaiting for instance (cold start may take ~2 min)...")
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(180.0)) as c:
        for attempt in range(5):
            try:
                r = c.get("/healthz/detail")
                r.raise_for_status()
                detail = r.json()
                break
            except Exception as e:
                print(f"  attempt {attempt+1}/5: {e!r}")
                time.sleep(15)
        else:
            print("ERROR: could not reach service after 5 attempts")
            sys.exit(1)
        print(f"Service: {detail['service']} v{detail['version']}")
        print(f"Active jobs: {detail['active_jobs']}, "
              f"max_concurrent: {detail['max_concurrent_jobs']}")
        print(f"Disk: {detail['disk_usage_mb']:.0f}/{detail['disk_limit_mb']} MB")

    results: list[BenchResult] = []

    # Phase 1: single job baseline
    print("\n[Phase 1] Single job baseline...")
    single = BenchResult(label="single job baseline")
    t0 = time.monotonic()
    jr = submit_and_poll(base_url, endpoint, data, files, tag="baseline")
    single.jobs.append(jr)
    single.wall_time = time.monotonic() - t0
    results.append(single)
    print(f"  Baseline: {jr.final_status} server={jr.duration_server}s "
          f"e2e={jr.total_time:.1f}s")

    if jr.final_status != "completed":
        print("\nBaseline failed — skipping concurrency phases.")
        print_report(results)
        sys.exit(1)

    # Phase 2: concurrent submission
    if args.concurrency > 0:
        print(f"\n[Phase 2] Concurrent submission x{args.concurrency}...")
        conc = run_concurrent(base_url, endpoint, data, files, args.concurrency)
        results.append(conc)

    # Phase 3: sequential throughput
    if args.jobs > 0:
        print(f"\n[Phase 3] Sequential throughput x{args.jobs}...")
        seq = run_sequential(base_url, endpoint, data, files, args.jobs)
        results.append(seq)

    print_report(results)


if __name__ == "__main__":
    main()
