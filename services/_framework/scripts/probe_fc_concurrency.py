"""FC 异步任务模式并发能力探测工具.

Fire N concurrent async invocations against a deployed task endpoint and observe:
  - How many submissions FC accepts vs rejects (HTTP 202 / 429 / other)
  - How many unique GPU instances FC provisions
  - Peak concurrent running tasks observed via started_at..completed_at overlap
  - Per-task duration + queue-to-start latency distribution

This is an operational probe (not a pytest) for sizing FC's max-async-concurrency
and GPU quota. Use after migrating a new service to task endpoints, or whenever
FC console concurrency settings change.

Usage
-----
Minimum (boltz-server default payload, N=10)::

    FC_URL=https://fc-boltz-XXX.cn-hangzhou-vpc.fcapp.run \
        uv run python services/_framework/scripts/probe_fc_concurrency.py

Generic, with custom payload file::

    FC_URL=https://fc-genie3-XXX.cn-hangzhou-vpc.fcapp.run \
        uv run python services/_framework/scripts/probe_fc_concurrency.py \
        --endpoint /api/tasks/generate/unconditional \
        --payload-file payload.json \
        --n 20

Payload file format (JSON object → form fields, list/dict values are JSON-stringified)::

    {
      "name": "probe",
      "msa_mode": "empty",
      "sequences": [{"type": "protein", "id": "A", "sequence": "MKQH..."}]
    }

What it reports
---------------
- Submit-phase HTTP code histogram (202 vs 429 vs others)
- Unique GPU instance IDs (= max parallel GPU instances FC spun up)
- Peak concurrent running tasks (from server-side started_at/completed_at overlap)
- Duration distribution + queue-to-first-visible latency

Interpreting the numbers
------------------------
- If submit returns mostly 429: FC console's "max async concurrency" is too low or
  the GPU quota is exhausted at account level.
- If unique instances < N: FC is reusing instances (sequential execution within
  same instance, possible if instanceConcurrency > 1; uncommon for GPU services).
- If peak concurrent < N: tasks are being serialized — could be max-async-concurrency
  cap, GPU quota, or single-instance-concurrency=1 with limited cold-start budget.
- Healthy concurrency target for GPU services: peak == N == unique instance count
  (each task gets its own GPU instance, all run in parallel).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import httpx


# ---------------------------------------------------------------------------
# Default payload: boltz-server /api/tasks/predict_structure minimal.
# Override via --payload-file.
# ---------------------------------------------------------------------------
DEFAULT_ENDPOINT = "/api/tasks/predict_structure"
DEFAULT_PAYLOAD: dict = {
    "name": "probe",
    "msa_mode": "empty",
    "diffusion_samples": 1,
    "recycling_steps": 1,
    "sampling_steps": 10,
    "sequences": [
        {
            "type": "protein",
            "id": "A",
            "sequence": "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSC",
            "msa_uri": "empty",
        }
    ],
}


@dataclass
class TaskRecord:
    task_id: str
    submit_status: Optional[int] = None
    submit_t: Optional[float] = None
    fc_request_id: Optional[str] = None
    first_seen_t: Optional[float] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    instance_id: Optional[str] = None
    final_status: Optional[str] = None
    duration_seconds: Optional[float] = None
    state_log: list[tuple[float, str]] = field(default_factory=list)


def _form_encode(payload: dict) -> dict[str, str]:
    """Convert payload dict to form-field strings (list/dict values → JSON strings)."""
    out: dict[str, str] = {}
    for k, v in payload.items():
        if isinstance(v, (list, dict)):
            out[k] = json.dumps(v)
        else:
            out[k] = str(v)
    return out


async def submit_one(
    client: httpx.AsyncClient,
    *,
    url: str,
    endpoint: str,
    form_data: dict[str, str],
    rec: TaskRecord,
    t0: float,
) -> None:
    headers = {
        "X-Fc-Invocation-Type": "Async",
        "X-Bioagent-Job-Id": rec.task_id,
        "X-Fc-Async-Task-Id": rec.task_id,
    }
    try:
        r = await client.post(
            f"{url}{endpoint}",
            data=form_data,
            headers=headers,
            timeout=60.0,
        )
        rec.submit_status = r.status_code
        rec.submit_t = time.time() - t0
        rec.fc_request_id = r.headers.get("x-fc-request-id")
    except Exception as e:
        rec.submit_status = -1
        rec.submit_t = time.time() - t0
        rec.state_log.append((time.time() - t0, f"submit_error: {e}"))


async def submit_all(
    records: list[TaskRecord],
    *,
    url: str,
    endpoint: str,
    payload: dict,
) -> None:
    print(f"[{time.strftime('%F %T')}] Submitting {len(records)} tasks concurrently to {endpoint}...")
    form_data = _form_encode(payload)
    t0 = time.time()
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[
            submit_one(client, url=url, endpoint=endpoint, form_data=form_data, rec=r, t0=t0)
            for r in records
        ])
    print(f"[{time.strftime('%F %T')}] Submit phase took {time.time()-t0:.1f}s")


async def poll_one(
    client: httpx.AsyncClient,
    *,
    url: str,
    rec: TaskRecord,
    t0: float,
    poll_interval_s: float,
    timeout_s: float,
) -> None:
    while True:
        elapsed = time.time() - t0
        if elapsed > timeout_s:
            rec.state_log.append((elapsed, "TIMEOUT"))
            return
        try:
            r = await client.get(f"{url}/api/jobs/{rec.task_id}", timeout=30.0)
            if r.status_code == 404:
                await asyncio.sleep(poll_interval_s)
                continue
            r.raise_for_status()
            body = r.json()
        except Exception as e:
            rec.state_log.append((elapsed, f"poll_error: {e}"))
            await asyncio.sleep(poll_interval_s)
            continue

        if rec.first_seen_t is None:
            rec.first_seen_t = elapsed
            rec.state_log.append((elapsed, f"first_seen status={body['status']}"))

        rec.created_at = body.get("created_at")
        rec.started_at = body.get("started_at")
        rec.completed_at = body.get("completed_at")
        rec.instance_id = body.get("instance_id")
        rec.final_status = body["status"]
        rec.duration_seconds = body.get("duration_seconds")

        status = body["status"]
        last_status = rec.state_log[-1][1].split()[-1] if rec.state_log else None
        if not rec.state_log or status != last_status:
            rec.state_log.append((elapsed, status))

        if status in ("completed", "failed"):
            return
        await asyncio.sleep(poll_interval_s)


async def poll_all(
    records: list[TaskRecord],
    *,
    url: str,
    poll_interval_s: float,
    timeout_s: float,
) -> None:
    t0 = time.time()
    print(
        f"[{time.strftime('%F %T')}] Polling {len(records)} tasks "
        f"(interval {poll_interval_s}s, timeout {timeout_s}s)..."
    )
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[
            poll_one(
                client, url=url, rec=r, t0=t0,
                poll_interval_s=poll_interval_s, timeout_s=timeout_s,
            )
            for r in records
        ])
    print(f"[{time.strftime('%F %T')}] Polling phase took {time.time()-t0:.1f}s")


def _parse_iso(t: Optional[str]) -> Optional[dt.datetime]:
    if not t:
        return None
    return dt.datetime.fromisoformat(t.replace("Z", "+00:00"))


def report(records: list[TaskRecord], *, url: str, endpoint: str) -> None:
    print()
    print("=" * 80)
    print(f"FC Concurrency Probe Report (N={len(records)})")
    print(f"  URL:      {url}")
    print(f"  Endpoint: {endpoint}")
    print("=" * 80)

    # Submit-phase HTTP codes
    submit_codes = Counter(r.submit_status for r in records)
    print(f"\n## Submit-phase HTTP codes")
    for code, count in sorted(submit_codes.items(), key=lambda x: str(x[0])):
        label = {202: "Accepted", 409: "Conflict (dedup)", 429: "Quota exhausted", -1: "submit_error"}.get(code, "")
        print(f"  {code} {label:<24} {count}")

    # GPU instance fanout
    instance_ids = [r.instance_id for r in records if r.instance_id]
    unique_instances = set(instance_ids)
    print(f"\n## GPU instance fanout")
    print(f"  Unique instance IDs: {len(unique_instances)}")
    if unique_instances:
        per_inst = Counter(instance_ids)
        for inst, cnt in per_inst.most_common():
            print(f"    {inst}: {cnt} task(s)")

    # Final status outcomes
    final_statuses = Counter(r.final_status for r in records)
    print(f"\n## Final status")
    for s, c in final_statuses.most_common():
        print(f"  {s or '<unknown>'}: {c}")

    # Durations
    durations = sorted(r.duration_seconds for r in records if r.duration_seconds)
    if durations:
        print(f"\n## Duration (server-side compute, seconds)")
        print(
            f"  min={durations[0]:.1f}  "
            f"median={durations[len(durations)//2]:.1f}  "
            f"max={durations[-1]:.1f}  n={len(durations)}"
        )

    # Queue-to-first-visible latency
    first_seens = sorted(r.first_seen_t for r in records if r.first_seen_t is not None)
    if first_seens:
        print(f"\n## Latency from probe-start to first JobInfo (seconds)")
        print(
            f"  min={first_seens[0]:.0f}  "
            f"median={first_seens[len(first_seens)//2]:.0f}  "
            f"max={first_seens[-1]:.0f}"
        )

    # Peak concurrent running
    ranges: list[tuple[dt.datetime, dt.datetime]] = []
    for r in records:
        st = _parse_iso(r.started_at)
        et = _parse_iso(r.completed_at)
        if st and et:
            ranges.append((st, et))

    if ranges:
        global_start = min(r[0] for r in ranges)
        global_end = max(r[1] for r in ranges)
        total_secs = max(int((global_end - global_start).total_seconds()), 1)
        bin_sec = 30
        n_bins = total_secs // bin_sec + 1
        active_per_bin = [0] * n_bins
        for st, et in ranges:
            s_idx = int((st - global_start).total_seconds() // bin_sec)
            e_idx = int((et - global_start).total_seconds() // bin_sec)
            for i in range(s_idx, min(e_idx + 1, n_bins)):
                active_per_bin[i] += 1
        peak = max(active_per_bin)
        print(f"\n## Concurrent running peak (30s bins via started_at..completed_at)")
        print(f"  Peak concurrent tasks: {peak} (of {len(records)} submitted)")
        if peak < len(records) * 0.8 and peak > 0:
            print(
                f"  ⚠ Peak << N — FC max-async-concurrency, GPU quota, or "
                f"single-instance concurrency cap is limiting parallelism."
            )

    # Per-task summary
    print(f"\n## Per-task summary")
    print(f"  {'task_id':<36} {'submit':<8} {'status':<12} {'instance':<14} {'dur_s':<8}")
    for r in records:
        dur = f"{r.duration_seconds:.1f}" if r.duration_seconds else "-"
        inst = (r.instance_id or "-")[:14]
        print(f"  {r.task_id:<36} {str(r.submit_status):<8} {str(r.final_status):<12} {inst:<14} {dur:<8}")

    print()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("FC_URL"),
        help="FC function URL, e.g. https://fc-boltz-XXX.cn-hangzhou-vpc.fcapp.run "
        "(default: $FC_URL)",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"Task endpoint path (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--payload-file",
        type=str,
        default=None,
        help="Path to JSON file with form-field payload "
        "(default: built-in boltz-server minimal predict_structure payload)",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=10,
        help="Number of concurrent submissions (default: 10)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Task ID prefix; defaults to 'probe-<timestamp>'",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=15.0,
        help="Seconds between polls (default: 15)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2400.0,
        help="Per-task polling timeout in seconds (default: 2400 = 40 min)",
    )
    return parser


async def _main() -> int:
    args = _build_arg_parser().parse_args()
    if not args.url:
        print("error: --url is required (or set FC_URL env var)")
        return 2
    url = args.url.rstrip("/")

    if args.payload_file:
        with open(args.payload_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = DEFAULT_PAYLOAD

    label = args.label or f"probe-{int(time.time())}"
    records = [TaskRecord(task_id=f"{label}-{i:02d}") for i in range(args.n)]

    await submit_all(records, url=url, endpoint=args.endpoint, payload=payload)

    accepted = sum(1 for r in records if r.submit_status == 202)
    if accepted == 0:
        print("No tasks accepted; skipping poll phase.")
        report(records, url=url, endpoint=args.endpoint)
        return 1

    await poll_all(
        records,
        url=url,
        poll_interval_s=args.poll_interval,
        timeout_s=args.timeout,
    )
    report(records, url=url, endpoint=args.endpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
