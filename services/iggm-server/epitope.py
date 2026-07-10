"""Interface epitope calculation wrapper for iggm-server.

Upstream `design.py --cal_epitope` computes the antigen interface residues via
IgGM.protein.cal_ppi and only prints them to stdout.  This wrapper reuses the
same call and writes a structured epitope.json so the service can return it as
a job artifact (and callers can feed it straight back into /api/design's
`epitope` field — residue numbers are 1-based, matching design.py).

Run with cwd=/opt/iggm and PYTHONPATH=/opt/iggm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="IgGM epitope calculation")
    p.add_argument("--fasta", required=True)
    p.add_argument("--antigen", required=True)
    p.add_argument("--output", required=True, help="Output directory")
    a = p.parse_args()

    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    if not Path(a.fasta).is_file():
        print(f"ERROR: fasta not found: {a.fasta}", file=sys.stderr)
        sys.exit(2)
    if not Path(a.antigen).is_file():
        print(f"ERROR: antigen not found: {a.antigen}", file=sys.stderr)
        sys.exit(2)

    import torch

    from IgGM.protein import cal_ppi
    from IgGM.protein.parser import parse_fasta

    sequences, ids, _ = parse_fasta(a.fasta)
    epitope = cal_ppi(a.antigen, ids, sequences)
    residues = [int(i) + 1 for i in torch.nonzero(epitope).flatten().tolist()]

    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "epitope.json").write_text(
        json.dumps(
            {
                "epitope": residues,
                "antigen_id": ids[-1],
                "antigen_length": int(epitope.numel()),
            },
            indent=2,
        )
    )
    print(f"epitope: {' '.join(str(r) for r in residues)}")


if __name__ == "__main__":
    main()
