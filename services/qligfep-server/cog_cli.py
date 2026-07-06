"""Wrapper for qligfep scripts/COG.py — return center-of-geometry as JSON."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from . import vendor_shim


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pdb", type=Path, required=True)
    p.add_argument("--mode", choices=["all", "atomrange"], default="all")
    p.add_argument("--atom-range", default=None)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args(argv)


def run(pdb: Path, mode: str, atom_range: str | None,
        work_dir: Path, output_dir: Path) -> int:
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    script = vendor_shim.script("scripts/COG.py")
    argv = [sys.executable, str(script), str(pdb)]
    if mode == "atomrange":
        if not atom_range:
            print("--atom-range required when --mode=atomrange", file=sys.stderr)
            return 2
        argv += [atom_range]
    r = subprocess.run(argv, cwd=work_dir, capture_output=True, text=True)
    (output_dir / "cog.log").write_text(r.stdout + "\n---STDERR---\n" + r.stderr)
    if r.returncode != 0:
        return r.returncode

    floats = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", r.stdout)]
    if len(floats) < 3:
        print(f"failed to parse 3 floats from COG.py stdout: {r.stdout!r}", file=sys.stderr)
        return 1
    cx, cy, cz = floats[-3], floats[-2], floats[-1]
    (output_dir / "result.json").write_text(json.dumps(
        {"cx": cx, "cy": cy, "cz": cz, "mode": mode}
    ))
    return 0


def main() -> int:
    a = parse_args()
    return run(a.pdb, a.mode, a.atom_range, a.work_dir, a.output_dir)


if __name__ == "__main__":
    sys.exit(main())
