#!/usr/bin/env python3
"""Boltz-server concurrency benchmark against live FC deployment.

Usage:
    uv run python services/boltz-server/tests/bench_concurrency.py [--concurrency N] [--jobs N]

Measures:
  1. Single-job baseline latency (cold + warm)
  2. Concurrent submission behavior (503 rejection vs queuing)
  3. Sequential throughput (N jobs back-to-back)
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
import re

_URL_RE = re.compile(r"^([A-Za-z0-9_-]+):\s*(https?://\S+)\s*$")


def _fc_url(service: str) -> str:
    md = Path(__file__).resolve().parents[2] / "aliyun_fc_url.md"
    for line in md.read_text().splitlines():
        m = _URL_RE.match(line.strip())
        if m and m.group(1) == service:
            return m.group(2).rstrip("/")
    raise KeyError(f"{service} not found in {md}")


def _poll_job(client: httpx.Client, base_url: str, job_id: str,
              timeout_s: int = 1800, interval_s: int = 15) -> dict:
    deadline = time.monotonic() + timeout_s
    body: dict = {}
    while time.monotonic() < deadline:
        try:
            resp = client.get(f"{base_url}/api/jobs/{job_id}")
            resp.raise_for_status()
            body = resp.json()
        except Exception:
            time.sleep(interval_s)
            continue
        if body.get("status") in ("completed", "failed"):
            return body
        time.sleep(interval_s)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_s}s; last: {body.get('status')}")


SHORT_PROTEIN = "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSC"
FAST_PARAMS = {
    "name": "bench",
    "msa_mode": "empty",
    "diffusion_samples": "1",
    "recycling_steps": "1",
    "sampling_steps": "50",
    "sequences": json.dumps([
        {"type": "protein", "id": "A", "sequence": SHORT_PROTEIN, "msa_uri": "empty"},
    ]),
}


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
    base_url: str, timeout_s: int = 600, tag: str = "",
) -> JobResult:
    t0 = time.monotonic()
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
        r = c.post("/api/predict_structure", data=FAST_PARAMS)
        submit_time = time.monotonic() - t0

        if r.status_code != 200:
            return JobResult(
                job_id="", submit_status=r.status_code,
                submit_time=submit_time, total_time=time.monotonic() - t0,
                final_status=f"submit_failed_{r.status_code}",
                error=r.text[:200],
            )

        job_id = r.json()["job_id"]
        print(f"  [{tag}] submitted {job_id} in {submit_time:.1f}s")

        final = _poll_job(c, base_url, job_id, timeout_s=timeout_s, interval_s=10)
        total = time.monotonic() - t0

        return JobResult(
            job_id=job_id,
            submit_status=r.status_code,
            submit_time=submit_time,
            total_time=total,
            final_status=final["status"],
            duration_server=final.get("duration_seconds"),
        )


def run_sequential(base_url: str, n: int) -> BenchResult:
    result = BenchResult(label=f"sequential x{n}")
    t0 = time.monotonic()
    for i in range(n):
        jr = submit_and_poll(base_url, tag=f"seq-{i+1}/{n}")
        result.jobs.append(jr)
        print(f"  [seq-{i+1}] {jr.final_status} total={jr.total_time:.1f}s server={jr.duration_server}s")
    result.wall_time = time.monotonic() - t0
    return result


def run_concurrent(base_url: str, n: int) -> BenchResult:
    result = BenchResult(label=f"concurrent x{n}")
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {
            pool.submit(submit_and_poll, base_url, 900, f"par-{i+1}/{n}"): i
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
            print(f"  [par-{idx+1}] {jr.final_status} total={jr.total_time:.1f}s submit_rc={jr.submit_status}")
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
        failed = [j for j in r.jobs if j.final_status not in ("completed", None) and j.submit_status != 503]

        if succeeded:
            server_times = [j.duration_server for j in succeeded if j.duration_server]
            total_times = [j.total_time for j in succeeded if j.total_time]
            print(f"  completed: {len(succeeded)}/{len(r.jobs)}")
            if server_times:
                print(f"  server duration: min={min(server_times):.1f}s avg={sum(server_times)/len(server_times):.1f}s max={max(server_times):.1f}s")
            if total_times:
                print(f"  e2e latency:     min={min(total_times):.1f}s avg={sum(total_times)/len(total_times):.1f}s max={max(total_times):.1f}s")
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


def main():
    parser = argparse.ArgumentParser(description="Boltz-server concurrency bench")
    parser.add_argument("--concurrency", "-c", type=int, default=3)
    parser.add_argument("--jobs", "-n", type=int, default=3)
    args = parser.parse_args()

    base_url = _fc_url("boltz-server")
    print(f"Target: {base_url}")

    # Warm-up / healthcheck (FC cold start can take 60-120s for GPU instances)
    print("Waiting for instance (cold start may take ~2 min)...")
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
        print(f"Active jobs: {detail['active_jobs']}, max_concurrent: {detail['max_concurrent_jobs']}")
        print(f"Disk: {detail['disk_usage_mb']:.0f}/{detail['disk_limit_mb']} MB")

    results: list[BenchResult] = []

    # Phase 1: single job baseline
    print("\n[Phase 1] Single job baseline...")
    single = BenchResult(label="single job baseline")
    t0 = time.monotonic()
    jr = submit_and_poll(base_url, tag="baseline")
    single.jobs.append(jr)
    single.wall_time = time.monotonic() - t0
    results.append(single)
    print(f"  Baseline: {jr.final_status} server={jr.duration_server}s e2e={jr.total_time:.1f}s")

    # Phase 2: concurrent submission
    print(f"\n[Phase 2] Concurrent submission x{args.concurrency}...")
    conc = run_concurrent(base_url, args.concurrency)
    results.append(conc)

    # Phase 3: sequential throughput
    print(f"\n[Phase 3] Sequential throughput x{args.jobs}...")
    seq = run_sequential(base_url, args.jobs)
    results.append(seq)

    print_report(results)


if __name__ == "__main__":
    main()
