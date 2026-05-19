"""Batch DockQ driver.

Invoked by `tools.batch_argv()` as a subprocess. Loops over every PDB/CIF file
under --models-dir, runs `DockQ <model> <native> --json ...` against the shared
native, then writes a sorted summary CSV.

Output layout (relative to --output-dir):

    scores.csv            — one row per model, sorted by --sort-by (descending)
    per_model/<m>.json    — raw DockQ JSON output per model
    failed.csv            — list of models that errored (basename + stderr tail)

The DockQ CLI flags this script accepts are forwarded verbatim to each `DockQ`
invocation; anything not understood here is passed through.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dockq-batch")


_SUMMARY_COLUMNS = (
    "model", "DockQ", "DockQ_F1", "iRMSD", "LRMSD",
    "fnat", "fnonnat", "F1", "clashes", "n_interfaces",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dockq-bin", default="DockQ")
    parser.add_argument("--sort-by", default="DockQ")
    # Capture every other --foo / --foo VALUE flag and forward to DockQ.
    parser.add_argument("--mapping", default=None)
    parser.add_argument("--small_molecule", action="store_true")
    parser.add_argument("--capri_peptide", action="store_true")
    parser.add_argument("--no_align", action="store_true")
    parser.add_argument("--optDockQF1", action="store_true")
    parser.add_argument("--allowed_mismatches", type=int, default=0)
    parser.add_argument("--n_cpu", type=int, default=4)
    return parser.parse_args()


def _forwarded_flags(ns: argparse.Namespace) -> list[str]:
    """Reconstruct the DockQ flag set from parsed args."""
    flags: list[str] = ["--n_cpu", str(ns.n_cpu)]
    if ns.mapping:
        flags += ["--mapping", ns.mapping]
    if ns.small_molecule:
        flags.append("--small_molecule")
    if ns.capri_peptide:
        flags.append("--capri_peptide")
    if ns.no_align:
        flags.append("--no_align")
    if ns.optDockQF1:
        flags.append("--optDockQF1")
    if ns.allowed_mismatches:
        flags += ["--allowed_mismatches", str(ns.allowed_mismatches)]
    return flags


def _summarize(model_basename: str, dockq_json: dict) -> dict:
    """Collapse DockQ's per-interface JSON into one summary row.

    DockQ writes a top-level `"total_DockQ"` and a nested `"best_result"` dict
    keyed by chain-mapping. We take total_DockQ as the headline score and
    average the per-interface metrics across the best mapping.
    """
    total = dockq_json.get("total_DockQ", dockq_json.get("DockQ", float("nan")))
    interfaces = dockq_json.get("best_result", {})
    if not interfaces:
        # Some output shapes put the interfaces at top-level (single interface).
        # Use whatever numeric fields are present.
        interfaces = {("?", "?"): dockq_json}

    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    for _key, vals in interfaces.items():
        for k in ("DockQ", "DockQ_F1", "iRMSD", "LRMSD", "fnat", "fnonnat", "F1", "clashes"):
            v = vals.get(k)
            if isinstance(v, (int, float)):
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
                metric_counts[k] = metric_counts.get(k, 0) + 1

    row: dict = {"model": model_basename, "n_interfaces": len(interfaces)}
    for k in _SUMMARY_COLUMNS:
        if k in ("model", "n_interfaces"):
            continue
        if k == "DockQ":
            row[k] = total
        elif k in metric_sums:
            row[k] = metric_sums[k] / metric_counts[k]
        else:
            row[k] = ""
    return row


def _run_one(
    model: Path,
    native: Path,
    json_out: Path,
    dockq_bin: str,
    forwarded: list[str],
) -> tuple[int, str]:
    """Run DockQ on one model; return (returncode, stderr_tail)."""
    cmd = [dockq_bin, str(model), str(native), "--json", str(json_out), "--short", *forwarded]
    logger.info("invoking: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        logger.warning("DockQ failed on %s (rc=%d): %s", model.name, proc.returncode, tail)
        return proc.returncode, tail
    return 0, ""


def main() -> int:
    ns = _parse_args()
    out_dir: Path = ns.output_dir
    per_model_dir = out_dir / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(
        [p for p in ns.models_dir.iterdir() if p.suffix.lower() in (".pdb", ".cif", ".gz")],
    )
    if not models:
        logger.error("no model files under %s", ns.models_dir)
        return 2

    forwarded = _forwarded_flags(ns)

    rows: list[dict] = []
    failures: list[dict] = []
    for model in models:
        json_out = per_model_dir / f"{model.stem}.json"
        rc, tail = _run_one(model, ns.native, json_out, ns.dockq_bin, forwarded)
        if rc != 0 or not json_out.exists():
            failures.append({"model": model.name, "stderr_tail": tail})
            continue
        try:
            data = json.loads(json_out.read_text())
        except json.JSONDecodeError as e:
            failures.append({"model": model.name, "stderr_tail": f"unparseable JSON: {e}"})
            continue
        rows.append(_summarize(model.stem, data))

    # Sort rows by the requested column. Fall back to DockQ if column is missing.
    sort_key = ns.sort_by if rows and ns.sort_by in rows[0] else "DockQ"

    def _sort_value(row: dict) -> float:
        v = row.get(sort_key)
        try:
            return -float(v)  # descending — negate for sorted()'s ascending default
        except (TypeError, ValueError):
            return 0.0

    rows.sort(key=_sort_value)

    scores_csv = out_dir / "scores.csv"
    with open(scores_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    if failures:
        with open(out_dir / "failed.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "stderr_tail"])
            writer.writeheader()
            writer.writerows(failures)

    logger.info(
        "batch complete: %d scored / %d failed → %s",
        len(rows), len(failures), scores_csv,
    )
    # Exit non-zero only if no model produced a valid result. Partial failures
    # are recorded in failed.csv and the job is still considered successful.
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
