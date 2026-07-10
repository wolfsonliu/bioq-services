"""Thin wrapper over upstream IgGM design.py.

Why a wrapper instead of calling design.py directly:

1. **Seed injection.** design.py hardcodes setup(True) with seed=42 and does
   not expose --seed. We call IgGM.utils.setup(True, seed) ourselves before
   invoking design.predict so a request-supplied seed takes effect.
2. **Input pre-validation.** Upstream validates the FASTA with bare asserts
   (chain count, standard residues) that produce opaque tracebacks. We check
   chain count / antigen-last / design-region presence up front and exit(2)
   with a clear message so the service returns an actionable failure.

Everything else (model loading, sampling, PDB/FASTA writing) is upstream
design.predict, unchanged.  Run with cwd=/opt/iggm and PYTHONPATH=/opt/iggm so
`import design` and `from IgGM...` resolve the vendored upstream, and the
./checkpoints symlink to the NAS weights_dir is in scope.

Usage::

    python run_design.py --fasta in.fasta --antigen ag.pdb --output <dir> \
        --run_task design --steps 10 --num_samples 1 [--epitope 7 8 9 ...] \
        [--fasta_origin origin.fasta] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

RUN_TASKS = ("design", "inverse_design", "fr_design", "affinity_maturation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IgGM design wrapper")
    p.add_argument("--fasta", required=True)
    p.add_argument("--antigen", required=True)
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--run_task", default="design", choices=RUN_TASKS)
    p.add_argument("--fasta_origin", default=None)
    p.add_argument("--epitope", nargs="+", type=int, default=None)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--chunk_size", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--max_antigen_size", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def validate_fasta(fasta_path: str, run_task: str) -> None:
    """Check chain count / antigen-last / design-region before heavy imports pay off."""
    from IgGM.protein.parser import parse_fasta

    sequences, ids, _ = parse_fasta(fasta_path)
    if len(sequences) not in (2, 3):
        _fail(
            f"FASTA must have 2 chains (nanobody: H,A) or 3 chains "
            f"(antibody: H,L,A) with the antigen last; got {len(sequences)} "
            f"chains: {ids}"
        )
    # Antibody chains are all but the last (antigen).  design/fr_design mask the
    # design region with 'X'; warn (not fail) if none is present.
    antibody_seqs = sequences[:-1]
    if run_task in ("design", "fr_design") and not any("X" in s for s in antibody_seqs):
        print(
            f"WARNING: run_task={run_task} but no 'X' design region found in the "
            f"antibody chains; nothing will be redesigned.",
            file=sys.stderr,
        )
    if run_task == "affinity_maturation" and not any("X" in s for s in antibody_seqs):
        _fail("affinity_maturation requires an 'X'-masked design region in the antibody chains")


def main() -> None:
    a = parse_args()

    # Ensure the vendored upstream is importable even if PYTHONPATH is unset.
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    if not Path(a.fasta).is_file():
        _fail(f"fasta not found: {a.fasta}")
    if not Path(a.antigen).is_file():
        _fail(f"antigen not found: {a.antigen}")
    if a.run_task == "affinity_maturation" and not a.fasta_origin:
        _fail("affinity_maturation requires --fasta_origin")
    if a.fasta_origin and not Path(a.fasta_origin).is_file():
        _fail(f"fasta_origin not found: {a.fasta_origin}")

    validate_fasta(a.fasta, a.run_task)

    Path(a.output).mkdir(parents=True, exist_ok=True)

    from IgGM.utils import setup

    setup(True, seed=a.seed)

    import design  # upstream design.py at repo root; __main__ guard keeps it inert on import

    args = Namespace(
        fasta=a.fasta,
        fasta_origin=a.fasta_origin,
        antigen=a.antigen,
        output=a.output,
        epitope=a.epitope,
        device=None,
        steps=a.steps,
        chunk_size=a.chunk_size,
        temperature=a.temperature,
        num_samples=a.num_samples,
        cal_epitope=False,
        relax=False,  # PyRosetta not installed in v0.0.1
        max_antigen_size=a.max_antigen_size,
        run_task=a.run_task,
    )
    design.predict(args)


if __name__ == "__main__":
    main()
