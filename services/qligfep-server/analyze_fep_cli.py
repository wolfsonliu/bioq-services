"""Wrapper for qligfep analyze_FEP.py — FEP DDG post-processing."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import vendor_shim


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end-state-catastrophe", type=float, required=True)
    p.add_argument("--use-pdb", action="store_true")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args(argv)


def _parse_results(results_txt: str) -> list[tuple[str, float, float]]:
    """Extract (pair, DDG, std) tuples from analyze_FEP.py results.txt."""
    rows: list[tuple[str, float, float]] = []
    for line in results_txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(
            r"^(\S+)\s*(?:->|to|\s+)\s*(\S+).*?"
            r"([-+]?\d*\.\d+|\d+)\s*(?:kcal/mol)?.*?"
            r"([-+]?\d*\.\d+|\d+)",
            line,
        )
        if m:
            rows.append((f"{m.group(1)}->{m.group(2)}",
                         float(m.group(3)), float(m.group(4))))
    return rows


def run(*, run_dir: Path, temperature: float, start: str,
        end_state_catastrophe: float, use_pdb: bool,
        work_dir: Path, output_dir: Path) -> int:
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in run_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, work_dir / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, work_dir / item.name)

    script = vendor_shim.script("analyze_FEP.py")
    argv = [
        sys.executable, str(script),
        "-F", ".", "-T", str(temperature), "-l", start,
        "-esc", str(end_state_catastrophe),
    ]
    if use_pdb:
        argv += ["-pdb", "true"]
    r = subprocess.run(argv, cwd=work_dir, capture_output=True, text=True)
    (output_dir / "analyze.log").write_text(
        r.stdout + "\n---STDERR---\n" + r.stderr
    )
    if r.returncode != 0:
        return r.returncode

    results = work_dir / "results.txt"
    if results.exists():
        shutil.copy2(results, output_dir / "results.txt")
        rows = _parse_results(results.read_text())
        with open(output_dir / "DDG.csv", "w") as f:
            f.write("pair,DDG_kcalmol,std_kcalmol\n")
            for pair, ddg, std in rows:
                f.write(f"{pair},{ddg:.2f},{std:.2f}\n")
    return 0


def main():
    a = parse_args()
    return run(
        run_dir=a.run_dir, temperature=a.temperature, start=a.start,
        end_state_catastrophe=a.end_state_catastrophe, use_pdb=a.use_pdb,
        work_dir=a.work_dir, output_dir=a.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
