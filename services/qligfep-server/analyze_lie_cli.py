"""Wrapper for qligfep analyze_LIE.py — LIE post-processing."""
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
    p.add_argument("--radius", type=float, required=True)
    p.add_argument("--cofactor", action="append", default=[], dest="cofactors")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args(argv)


def _parse_lie(results_txt: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for line in results_txt.splitlines():
        m = re.match(r"LIE\s+(\S+):\s*dG\s*=\s*([-+]?\d*\.\d+|\d+)", line.strip())
        if m:
            rows.append((m.group(1), float(m.group(2))))
    return rows


def run(*, run_dir, radius, cofactors, work_dir, output_dir):
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in run_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, work_dir / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, work_dir / item.name)

    script = vendor_shim.script("analyze_LIE.py")
    argv = [sys.executable, str(script), "-R", str(radius)]
    for c in cofactors or []:
        argv += ["-c", c]

    r = subprocess.run(argv, cwd=work_dir, capture_output=True, text=True)
    (output_dir / "analyze.log").write_text(
        r.stdout + "\n---STDERR---\n" + r.stderr
    )
    if r.returncode != 0:
        return r.returncode

    results = work_dir / "results.txt"
    if results.exists():
        shutil.copy2(results, output_dir / "results.txt")
        rows = _parse_lie(results.read_text())
        with open(output_dir / "LIE_energies.csv", "w") as f:
            f.write("ligand,dG_kcalmol\n")
            for lig, dg in rows:
                f.write(f"{lig},{dg:.2f}\n")
    return 0


def main():
    a = parse_args()
    return run(
        run_dir=a.run_dir, radius=a.radius, cofactors=a.cofactors,
        work_dir=a.work_dir, output_dir=a.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
