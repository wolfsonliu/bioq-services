"""diamond-server subprocess driver (invoked as `python -m server.diamond_driver`).

Handles the multi-step flows that a single DIAMOND invocation can't:

* ``search``  — optional inline `makedb` (when a subject FASTA is given instead
  of a prebuilt `.dmnd`) followed by `blastp` / `blastx`.
* ``cluster`` — `makedb` the input FASTA, then `cluster` / `deepclust` /
  `linclust` into a representative↔member TSV.
* ``msa``     — `blastp` with an alignment-carrying `--outfmt 6`, then
  reconstruct a query-anchored a3m via `server.a3m`.

Kept CLI-only glue: pure logic (a3m reconstruction) lives in `server.a3m` so it
is unit-tested without a DIAMOND binary.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from server.a3m import BLAST_FIELDS, parse_blast_tab, read_first_fasta, reconstruct_a3m

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("diamond-driver")


def _run(argv: list[str]) -> int:
    logger.info("invoking: %s", " ".join(argv))
    return subprocess.run(argv).returncode


def _makedb(diamond_bin: str, fasta: Path, db_stem: Path, threads: int) -> int:
    db_stem.parent.mkdir(parents=True, exist_ok=True)
    return _run([diamond_bin, "makedb", "--in", str(fasta), "--db", str(db_stem), "-p", str(threads)])


def _sensitivity_flag(sensitivity: str | None) -> list[str]:
    return [f"--{sensitivity}"] if sensitivity else []


def cmd_search(ns: argparse.Namespace) -> int:
    out_path = Path(ns.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if ns.subject:
        db_stem = Path(ns.db_work) / "ref"
        rc = _makedb(ns.diamond_bin, Path(ns.subject), db_stem, ns.threads)
        if rc != 0:
            return rc
        db_path = str(db_stem)
    elif ns.db:
        db_path = ns.db
    else:
        logger.error("search requires either --db (.dmnd) or --subject (FASTA)")
        return 2

    argv = [
        ns.diamond_bin, ns.command,
        "-q", ns.query,
        "-d", db_path,
        "-o", str(out_path),
        "--outfmt", ns.outfmt,
        "-e", str(ns.evalue),
        "-k", str(ns.max_target_seqs),
        "-p", str(ns.threads),
        *_sensitivity_flag(ns.sensitivity),
    ]
    return _run(argv)


def cmd_cluster(ns: argparse.Namespace) -> int:
    out_path = Path(ns.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    db_stem = Path(ns.db_work) / "ref"
    rc = _makedb(ns.diamond_bin, Path(ns.sequences), db_stem, ns.threads)
    if rc != 0:
        return rc

    argv = [
        ns.diamond_bin, ns.algorithm,
        "-d", str(db_stem),
        "-o", str(out_path),
        "-p", str(ns.threads),
        *_sensitivity_flag(ns.sensitivity),
    ]
    if ns.approx_id is not None:
        argv += ["--approx-id", str(ns.approx_id)]
    if ns.member_cover is not None:
        argv += ["--member-cover", str(ns.member_cover)]
    return _run(argv)


def cmd_msa(ns: argparse.Namespace) -> int:
    out_path = Path(ns.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    query_id, query_seq = read_first_fasta(Path(ns.query))

    hits_tsv = out_path.parent / f"{out_path.stem}.hits.tsv"
    argv = [
        ns.diamond_bin, "blastp",
        "-q", ns.query,
        "-d", ns.db,
        "-o", str(hits_tsv),
        "--outfmt", "6", *BLAST_FIELDS,
        "-e", str(ns.evalue),
        "-k", str(ns.max_target_seqs),
        "-p", str(ns.threads),
        *_sensitivity_flag(ns.sensitivity),
    ]
    rc = _run(argv)
    if rc != 0:
        return rc

    hits = parse_blast_tab(hits_tsv)
    out_path.write_text(reconstruct_a3m(query_id, query_seq, hits))
    logger.info("wrote a3m with %d homolog(s) → %s", len(hits), out_path)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="subcommand", required=True)

    s = sub.add_parser("search", help="blastp/blastx (optional inline makedb)")
    s.add_argument("--command", required=True, choices=("blastp", "blastx"))
    s.add_argument("--query", required=True)
    s.add_argument("--db", default=None, help="prebuilt .dmnd (stem or path)")
    s.add_argument("--subject", default=None, help="subject FASTA to build DB inline")
    s.add_argument("--output", required=True)
    s.add_argument("--db-work", dest="db_work", required=True)
    s.add_argument("--diamond-bin", dest="diamond_bin", required=True)
    s.add_argument("--outfmt", default="6")
    s.add_argument("--evalue", type=float, default=0.001)
    s.add_argument("--max-target-seqs", dest="max_target_seqs", type=int, default=25)
    s.add_argument("--sensitivity", default=None)
    s.add_argument("-p", "--threads", type=int, default=8)
    s.set_defaults(func=cmd_search)

    c = sub.add_parser("cluster", help="makedb + cluster/deepclust/linclust")
    c.add_argument("--algorithm", required=True, choices=("cluster", "deepclust", "linclust"))
    c.add_argument("--sequences", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--db-work", dest="db_work", required=True)
    c.add_argument("--diamond-bin", dest="diamond_bin", required=True)
    c.add_argument("--sensitivity", default=None)
    c.add_argument("--approx-id", dest="approx_id", type=float, default=None)
    c.add_argument("--member-cover", dest="member_cover", type=float, default=None)
    c.add_argument("-p", "--threads", type=int, default=8)
    c.set_defaults(func=cmd_cluster)

    m = sub.add_parser("msa", help="blastp → query-anchored a3m")
    m.add_argument("--query", required=True)
    m.add_argument("--db", required=True)
    m.add_argument("--output", required=True)
    m.add_argument("--diamond-bin", dest="diamond_bin", required=True)
    m.add_argument("--evalue", type=float, default=0.001)
    m.add_argument("--max-target-seqs", dest="max_target_seqs", type=int, default=2000)
    m.add_argument("--sensitivity", default=None)
    m.add_argument("-p", "--threads", type=int, default=8)
    m.set_defaults(func=cmd_msa)

    return parser


def main() -> int:
    ns = _build_parser().parse_args()
    return ns.func(ns)


if __name__ == "__main__":
    sys.exit(main())
