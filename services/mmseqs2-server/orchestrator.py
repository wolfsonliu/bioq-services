"""Vendored ColabFold MSA orchestration.

Source: opensource/ColabFold/colabfold/mmseqs/search.py
Upstream commit: 26c0d46e12a98603a190231d643f6cafa49566b4
Upstream date:   2026-03-07
Upstream license: MIT (Copyright (c) 2021 Sergey Ovchinnikov)
                  see opensource/ColabFold/LICENSE

Why vendor (vs pip install colabfold):
  The MSA pipeline is *not* a single mmseqs subcommand — it is ~15 sequential
  ``mmseqs <subcmd>`` subprocess calls orchestrated by Python.  Installing
  the upstream ColabFold package would drag in biopython, numpy, alphafold,
  and AF3 JSON output code we don't need for a server-side MSA service.
  Instead we copy the two orchestration functions (``mmseqs_search_monomer``
  and ``mmseqs_search_pair``) into this repo, git-track them, and maintain
  them as our own code (see VENDOR_INFO.md for the sync procedure).

Local changes (vs upstream):
  - Drop ``run_mmseqs2()`` remote-API client (we ARE the server, never call
    out to api.colabfold.com).
  - Drop ``--af3-json`` / ``--af3-msa-as-path`` AF3 output mode and the
    ``AF3Utils`` dependency (out of scope — this service produces a3m only).
  - Drop ``--use-templates`` / template DB handling (Phase 2 will add a
    template service; Phase 1 ships a3m only).
  - Drop the multi-query batching loop in upstream's ``main()``.  Each
    invocation of this module handles exactly one query file because the
    HTTP / CLI front-end issues one job per request.
  - Replace argparse main() with a CLI matching this service's contract
    (see services/mmseqs2-server/tools.py:colabfold_search_argv()).
  - Vendor ``get_queries`` / ``msa_to_str`` / ``safe_filename`` from
    ``colabfold.input`` into ``_colabfold_helpers.py`` (separate file, same
    provenance treatment) instead of pulling them via package import.
  - Add detailed inline comments on every mmseqs subprocess call.

To find / refresh the upstream SHA:
    cd opensource/ColabFold && \\
        git log -1 --format=%H -- colabfold/mmseqs/search.py
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import List, Union

# Helpers vendored from colabfold/input.py.  When running under tests, the
# package is mounted as ``server`` (see tests/conftest.py); when running
# inside Docker via ``python -m server.orchestrator``, ``server`` is the
# /opt/mmseqs2/server package.  Either way, an absolute import works.
try:
    from server._colabfold_helpers import (  # type: ignore
        get_queries,
        msa_to_str,
        safe_filename,
    )
except ImportError:  # pragma: no cover — fallback for direct script execution
    from _colabfold_helpers import (  # type: ignore
        get_queries,
        msa_to_str,
        safe_filename,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subprocess plumbing
# ---------------------------------------------------------------------------

# Position of the *output* file/DB argument in the argv of each mmseqs
# subcommand we use.  Used by ``run_mmseqs`` to skip steps whose output
# already exists on disk (lets the orchestrator be resumed after a crash).
#
# Upstream reference: colabfold/mmseqs/search.py:21-32
MODULE_OUTPUT_POS = {
    "align":        4,
    "convertalis":  4,
    "expandaln":    5,
    "filterresult": 4,
    "lndb":         2,
    "mergedbs":     2,
    "mvdb":         2,
    "pairaln":      4,
    "result2msa":   4,
    "search":       3,
}


def run_mmseqs(mmseqs: Path, params: List[Union[str, Path]]) -> None:
    """Invoke ``mmseqs <module> <params>`` via subprocess, with resume logic.

    Upstream reference: colabfold/mmseqs/search.py:34-47

    If the module is in ``MODULE_OUTPUT_POS`` and the corresponding output
    ``.dbtype`` file already exists, the call is skipped — this is how the
    pipeline tolerates a re-run after partial failure.  ``rmdb``/``mvdb``/
    ``unpackdb``/``createdb`` etc. without an entry just always run.

    Sets ``MMSEQS_CALL_DEPTH=1`` so mmseqs prints a single-line parameter
    summary instead of its full default banner (which clogs the FC logs).

    Raises ``subprocess.CalledProcessError`` on non-zero exit — by design,
    so the SubprocessRunner / FC task fails fast.
    """
    module = params[0]
    if module in MODULE_OUTPUT_POS:
        output_pos = MODULE_OUTPUT_POS[module]
        output_path = Path(params[output_pos]).with_suffix(".dbtype")
        if output_path.exists():
            logger.info(f"Skipping {module} because {output_path} already exists")
            return

    params_log = " ".join(str(i) for i in params)
    logger.info(f"Running {mmseqs} {params_log}")
    os.environ["MMSEQS_CALL_DEPTH"] = "1"
    # Diverges from upstream: coerce mmseqs + each param to str so callers
    # can pass Path objects without subprocess raising TypeError on Python
    # versions where check_call's argv only accepts str/bytes (upstream uses
    # [mmseqs] + params raw, which relies on Path being str-compatible).
    subprocess.check_call([str(mmseqs)] + [str(p) for p in params])


# ---------------------------------------------------------------------------
# Monomer (unpaired) search
# ---------------------------------------------------------------------------

def mmseqs_search_monomer(
    dbbase: Path,
    base: Path,
    uniref_db: Path = Path("uniref30_2302_db"),
    metagenomic_db: Path = Path("colabfold_envdb_202108_db"),
    mmseqs: Path = Path("mmseqs"),
    use_env: bool = True,
    filter: bool = True,
    expand_eval: float = math.inf,
    align_eval: int = 10,
    diff: int = 3000,
    qsc: float = -20.0,
    max_accept: int = 1000000,
    prefilter_mode: int = 0,
    s: float = 8,
    db_load_mode: int = 2,
    threads: int = 32,
    gpu: int = 0,
    gpu_server: int = 0,
    unpack: bool = True,
) -> None:
    """Run ColabFold's unpaired (monomer) MSA pipeline.

    Upstream reference: colabfold/mmseqs/search.py:50-210

    Pipeline outline:
      1. Iterative profile search vs UniRef30 → res
      2. Move the iteration-1 profile out of tmp/ so we can re-use it
      3. expandaln + align + filterresult + result2msa → uniref.a3m
      4. (Optional) repeat against the metagenomic (env) DB using the
         UniRef profile → bfd.mgnify30.metaeuk30.smag30.a3m
      5. Merge UniRef + env a3m via mergedbs → final.a3m
      6. unpackdb final.a3m into loose .a3m files named by jobname

    Diverges from upstream:
      - ``template_db`` / ``use_templates`` parameters and the templates
        search block removed (Phase 2 will add a templates pipeline).

    Inputs:
      - ``base/qdb*`` — sequence DB built by ``main()`` from the query FASTA
      - ``dbbase/<uniref_db>*`` — UniRef30 GPU sub-DB on disk
      - ``dbbase/<metagenomic_db>*`` — ColabFoldDB env DB on disk (if use_env)
    Outputs:
      - ``base/<id>.a3m`` — one a3m per query sequence (unpacked) when
        ``unpack=True``; otherwise ``base/final.a3m*`` MMseqs DB.
    """
    # ColabFold tightens align_eval / qsc / max_accept when filter is on.
    # Upstream reference: colabfold/mmseqs/search.py:79-84
    if filter:
        align_eval = 10
        qsc = 0.8
        max_accept = 100000

    used_dbs = [uniref_db]
    if use_env:
        used_dbs.append(metagenomic_db)

    # Probe every DB before launching — the search step caches a profile in
    # tmp/, which is annoying to clean up if a later step blows up on a
    # missing DB.  Also pick the right suffix layout (indexed vs raw)
    # depending on whether the DB has a ``.idx`` index.
    # Upstream reference: colabfold/mmseqs/search.py:92-110
    # Diverges from upstream: hoisted dbSuffix1/dbSuffix2 = ".idx" defaults
    # out of the for-loop (dropping the else-branch); semantically equivalent
    # because the indexed case sets both to ".idx" anyway, and we don't carry
    # dbSuffix3 (templates dropped).
    dbSuffix1 = ".idx"
    dbSuffix2 = ".idx"
    for db in used_dbs:
        if not dbbase.joinpath(f"{db}.dbtype").is_file():
            raise FileNotFoundError(f"Database {db} does not exist")
        if (
            (
                not dbbase.joinpath(f"{db}.idx").is_file()
                and not dbbase.joinpath(f"{db}.idx.index").is_file()
            )
            or os.environ.get("MMSEQS_IGNORE_INDEX", False)
        ):
            logger.info("Search does not use index")
            db_load_mode = 0
            dbSuffix1 = "_seq"
            dbSuffix2 = "_aln"

    # ---- Shared param blocks (upstream lines 112-128) -----------------
    # ``search_param`` is appended to every ``mmseqs search`` invocation.
    # The flag groupings here intentionally match upstream so future
    # 3-way diffs stay clean.
    search_param: List[str] = [
        "--num-iterations", "3",                # 3-iter PSI-style profile
        "--db-load-mode", str(db_load_mode),    # mmap=2 keeps DB resident
        "-a",                                   # produce backtrace (needed
                                                # for downstream align/expand)
        "-e", "0.1",                            # report below this e-value
        "--max-seqs", "10000",                  # cap hits/query (perf knob)
    ]
    if gpu:
        # MMseqs2-GPU only supports ``--prefilter-mode 1`` (ungapped) and
        # always runs at max sensitivity (k-score is fixed).  Don't pass
        # ``-s`` / ``--k-score`` in this branch.
        search_param += ["--gpu", str(gpu), "--prefilter-mode", "1"]
    else:
        search_param += ["--prefilter-mode", str(prefilter_mode)]
        if s is not None:
            search_param += ["-s", "{:.1f}".format(s)]
        else:
            search_param += ["--k-score", "'seq:96,prof:80'"]
    if gpu_server:
        search_param += ["--gpu-server", str(gpu_server)]

    filter_param = [
        "--filter-msa", str(1 if filter else 0),
        "--filter-min-enable", "1000",
        "--diff", str(diff),
        "--qid", "0.0,0.2,0.4,0.6,0.8,1.0",
        "--qsc", "0",
        "--max-seq-id", "0.95",
    ]
    expand_param = [
        "--expansion-mode", "0",
        "-e", str(expand_eval),
        "--expand-filter-clusters", str(1 if filter else 0),
        "--max-seq-id", "0.95",
    ]

    # ======================================================================
    # UniRef30 block — 7 active steps + 4 rmdb cleanups
    # ======================================================================
    if not base.joinpath("uniref.a3m").with_suffix(".a3m.dbtype").exists():
        # --- Step 1/15: iterative profile search vs UniRef30 ---
        # Why:    builds the iteration-1 profile and a hit list (res) that
        #         drive every downstream UniRef step.
        # Input:  base/qdb (query DB), dbbase/uniref_db
        # Output: base/res (alignment DB, consumed by expandaln in step 4),
        #         base/tmp/latest/profile_1 (profile, claimed in step 2)
        # Why these params:
        #   --num-iterations 3 = standard ColabFold profile depth
        #   --max-seqs 10000   = upper bound on prefilter hits per query
        #   --gpu + --prefilter-mode 1 (GPU branch) = required by mmseqs-GPU
        # Upstream reference: colabfold/mmseqs/search.py:131
        run_mmseqs(mmseqs, ["search",
                            base.joinpath("qdb"),
                            dbbase.joinpath(uniref_db),
                            base.joinpath("res"),
                            base.joinpath("tmp"),
                            "--threads", str(threads)] + search_param)

        # --- Step 2/15: claim the iteration-1 profile out of tmp/ ---
        # Why:    ``tmp/`` is deleted at the end of the pipeline; the
        #         profile must be moved to a stable location so the env-DB
        #         search (step 8) can reuse it.
        # Input:  base/tmp/latest/profile_1 (DB written by search)
        # Output: base/prof_res (renamed DB, used in steps 5, 8, 11)
        # Why these params: none (mvdb takes no flags)
        # Upstream reference: colabfold/mmseqs/search.py:132
        run_mmseqs(mmseqs, ["mvdb",
                            base.joinpath("tmp/latest/profile_1"),
                            base.joinpath("prof_res")])

        # --- Step 3/15: symlink query headers onto the profile DB ---
        # Why:    profile DBs need a ``_h`` header companion; rather than
        #         copy, symlink the query headers (one per query sequence).
        # Input:  base/qdb_h (query header DB)
        # Output: base/prof_res_h (header DB used by every subsequent step
        #         operating on prof_res)
        # Why these params: none
        # Upstream reference: colabfold/mmseqs/search.py:133
        run_mmseqs(mmseqs, ["lndb",
                            base.joinpath("qdb_h"),
                            base.joinpath("prof_res_h")])

        # --- Step 4/15: expand alignments through cluster members ---
        # Why:    UniRef30 stores cluster representatives + members; the
        #         search above only sees representatives.  expandaln pulls
        #         each cluster's members back into the result list so the
        #         MSA reflects the full diversity of UniRef.
        # Input:  base/qdb, dbbase/<uniref_db>.idx (sequences),
        #         base/res (representative hits),
        #         dbbase/<uniref_db>.idx (cluster alignments)
        # Output: base/res_exp (expanded result DB, consumed by step 5)
        # Why these params:
        #   --expansion-mode 0 = use input alignment scores (no rescore)
        #   --expand-filter-clusters 0/1 = whether to filter inside clusters
        # Upstream reference: colabfold/mmseqs/search.py:134
        run_mmseqs(mmseqs, ["expandaln",
                            base.joinpath("qdb"),
                            dbbase.joinpath(f"{uniref_db}{dbSuffix1}"),
                            base.joinpath("res"),
                            dbbase.joinpath(f"{uniref_db}{dbSuffix2}"),
                            base.joinpath("res_exp"),
                            "--db-load-mode", str(db_load_mode),
                            "--threads", str(threads)] + expand_param)

        # --- Step 5/15: re-align expanded hits against the profile ---
        # Why:    expandaln carries over representative scores; we need
        #         proper alignments of every (now-expanded) hit against the
        #         iter-1 profile so backtraces are correct for MSA building.
        # Input:  base/prof_res (profile), dbbase/<uniref_db>.idx,
        #         base/res_exp
        # Output: base/res_exp_realign (input to step 6)
        # Why these params:
        #   -e <align_eval>     = standard re-align e-value
        #   --max-accept        = cap accepted alignments (perf / memory)
        #   --alt-ali 10        = keep up to 10 alternative alignments
        #   -a                  = produce backtrace
        # Upstream reference: colabfold/mmseqs/search.py:135
        run_mmseqs(mmseqs, ["align",
                            base.joinpath("prof_res"),
                            dbbase.joinpath(f"{uniref_db}{dbSuffix1}"),
                            base.joinpath("res_exp"),
                            base.joinpath("res_exp_realign"),
                            "--db-load-mode", str(db_load_mode),
                            "-e", str(align_eval),
                            "--max-accept", str(max_accept),
                            "--threads", str(threads),
                            "--alt-ali", "10",
                            "-a"])

        # --- Step 6/15: filter realigned results by quality ---
        # Why:    drops low-score / redundant hits before MSA building so
        #         the resulting a3m stays compact and informative.
        # Input:  base/qdb, dbbase/<uniref_db>.idx, base/res_exp_realign
        # Output: base/res_exp_realign_filter (input to step 7)
        # Why these params:
        #   --qid 0 / --qsc <qsc> / --diff 0 = filter by score threshold only
        #   --max-seq-id 1.0   = don't dedupe at this stage
        #   --filter-min-enable 100 = skip filter for tiny result sets
        # Upstream reference: colabfold/mmseqs/search.py:136-139
        run_mmseqs(mmseqs, ["filterresult",
                            base.joinpath("qdb"),
                            dbbase.joinpath(f"{uniref_db}{dbSuffix1}"),
                            base.joinpath("res_exp_realign"),
                            base.joinpath("res_exp_realign_filter"),
                            "--db-load-mode", str(db_load_mode),
                            "--qid", "0",
                            "--qsc", str(qsc),
                            "--diff", "0",
                            "--threads", str(threads),
                            "--max-seq-id", "1.0",
                            "--filter-min-enable", "100"])

        # --- Step 7/15: convert filtered results into a3m MSA ---
        # Why:    produces the per-query a3m MSA from the alignment DB.
        # Input:  base/qdb, dbbase/<uniref_db>.idx, base/res_exp_realign_filter
        # Output: base/uniref.a3m (DB; merged with env-DB output in step 13)
        # Why these params:
        #   --msa-format-mode 6 = A3M with alignment info (ColabFold default
        #                         for downstream tools like Boltz)
        #   filter_param        = diversity filter applied during MSA build
        # Upstream reference: colabfold/mmseqs/search.py:140-142
        run_mmseqs(mmseqs, ["result2msa",
                            base.joinpath("qdb"),
                            dbbase.joinpath(f"{uniref_db}{dbSuffix1}"),
                            base.joinpath("res_exp_realign_filter"),
                            base.joinpath("uniref.a3m"),
                            "--msa-format-mode", "6",
                            "--db-load-mode", str(db_load_mode),
                            "--threads", str(threads)] + filter_param)

        # --- Cleanup: drop UniRef intermediate result DBs ---
        # Why:    free disk on the FC instance; these DBs aren't needed
        #         after result2msa.  rmdb removes both the data + index.
        # Input:  the four intermediates from steps 4-6
        # Output: none (DBs deleted)
        # Why these params: none (rmdb takes one positional)
        # Upstream reference: colabfold/mmseqs/search.py:143-146
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp_realign_filter")])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp_realign")])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp")])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("res")])
    else:
        logger.info(f"Skipping {uniref_db} search because uniref.a3m already exists")

    # ======================================================================
    # Env DB block (ColabFoldDB) — 5 active steps + 4 rmdb cleanups
    # ======================================================================
    if (
        use_env
        and not base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m")
        .with_suffix(".a3m.dbtype").exists()
    ):
        # --- Step 8/15: profile search vs the env DB ---
        # Why:    extend MSA with environmental (metagenomic) sequences,
        #         using the UniRef-derived profile from step 2.
        # Input:  base/prof_res (UniRef profile), dbbase/<metagenomic_db>
        # Output: base/res_env (env hit DB), base/tmp3/latest/profile_1
        #         (env-side profile claimed in step 10 implicitly)
        # Why these params: see step 1 (same search_param block)
        # Upstream reference: colabfold/mmseqs/search.py:151
        run_mmseqs(mmseqs, ["search",
                            base.joinpath("prof_res"),
                            dbbase.joinpath(metagenomic_db),
                            base.joinpath("res_env"),
                            base.joinpath("tmp3"),
                            "--threads", str(threads)] + search_param)

        # --- Step 9/15: expand env-DB hits through cluster members ---
        # Why:    same logic as step 4 but against ColabFoldDB clusters.
        # Input:  base/prof_res, dbbase/<metagenomic_db>.idx, base/res_env,
        #         dbbase/<metagenomic_db>.idx (cluster aln)
        # Output: base/res_env_exp (consumed by step 10)
        # Why these params: ``-e inf`` keeps every cluster member;
        #         ``--expansion-mode 0`` reuses input alignment scores
        # Upstream reference: colabfold/mmseqs/search.py:153-155
        run_mmseqs(mmseqs, ["expandaln",
                            base.joinpath("prof_res"),
                            dbbase.joinpath(f"{metagenomic_db}{dbSuffix1}"),
                            base.joinpath("res_env"),
                            dbbase.joinpath(f"{metagenomic_db}{dbSuffix2}"),
                            base.joinpath("res_env_exp"),
                            "-e", str(expand_eval),
                            "--expansion-mode", "0",
                            "--db-load-mode", str(db_load_mode),
                            "--threads", str(threads)])

        # --- Step 10/15: re-align expanded env hits against env profile ---
        # Why:    proper alignments + backtraces against the env-side
        #         profile written by step 8 into tmp3/.  Note we use
        #         ``tmp3/latest/profile_1`` directly — there is no env-side
        #         counterpart of the UniRef ``mvdb`` from step 2 because
        #         tmp3 is not deleted until the end of the run.
        # Input:  base/tmp3/latest/profile_1, dbbase/<metagenomic_db>.idx,
        #         base/res_env_exp
        # Output: base/res_env_exp_realign (consumed by step 11)
        # Why these params: same as step 5
        # Upstream reference: colabfold/mmseqs/search.py:156-159
        run_mmseqs(mmseqs, ["align",
                            base.joinpath("tmp3/latest/profile_1"),
                            dbbase.joinpath(f"{metagenomic_db}{dbSuffix1}"),
                            base.joinpath("res_env_exp"),
                            base.joinpath("res_env_exp_realign"),
                            "--db-load-mode", str(db_load_mode),
                            "-e", str(align_eval),
                            "--max-accept", str(max_accept),
                            "--threads", str(threads),
                            "--alt-ali", "10",
                            "-a"])

        # --- Step 11/15: filter env-side realigned hits ---
        # Why:    same logic as step 6, applied to env-DB hits.
        # Input:  base/qdb, dbbase/<metagenomic_db>.idx, base/res_env_exp_realign
        # Output: base/res_env_exp_realign_filter (consumed by step 12)
        # Why these params: same as step 6
        # Upstream reference: colabfold/mmseqs/search.py:160-163
        run_mmseqs(mmseqs, ["filterresult",
                            base.joinpath("qdb"),
                            dbbase.joinpath(f"{metagenomic_db}{dbSuffix1}"),
                            base.joinpath("res_env_exp_realign"),
                            base.joinpath("res_env_exp_realign_filter"),
                            "--db-load-mode", str(db_load_mode),
                            "--qid", "0",
                            "--qsc", str(qsc),
                            "--diff", "0",
                            "--max-seq-id", "1.0",
                            "--threads", str(threads),
                            "--filter-min-enable", "100"])

        # --- Step 12/15: convert env-side results into a3m MSA ---
        # Why:    produces the env-DB MSA, ready to merge with uniref.a3m.
        # Input:  base/qdb, dbbase/<metagenomic_db>.idx,
        #         base/res_env_exp_realign_filter
        # Output: base/bfd.mgnify30.metaeuk30.smag30.a3m (DB; merged in step 13)
        # Why these params:
        #   --msa-format-mode 6 = A3M with alignment info
        #   filter_param        = diversity filter applied during MSA build
        # Upstream reference: colabfold/mmseqs/search.py:164-167
        run_mmseqs(mmseqs, ["result2msa",
                            base.joinpath("qdb"),
                            dbbase.joinpath(f"{metagenomic_db}{dbSuffix1}"),
                            base.joinpath("res_env_exp_realign_filter"),
                            base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m"),
                            "--msa-format-mode", "6",
                            "--db-load-mode", str(db_load_mode),
                            "--threads", str(threads)] + filter_param)

        # --- Cleanup: drop env-side intermediate result DBs ---
        # Why:    free disk; not needed after result2msa.
        # Input:  the four intermediates from steps 9-11
        # Output: none (DBs deleted)
        # Why these params: none
        # Upstream reference: colabfold/mmseqs/search.py:168-171
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_env_exp_realign_filter")])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_env_exp_realign")])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_env_exp")])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_env")])
    elif use_env:
        logger.info(
            f"Skipping {metagenomic_db} search because "
            "bfd.mgnify30.metaeuk30.smag30.a3m already exists"
        )

    # ======================================================================
    # Merge UniRef + env MSAs into final.a3m
    # ======================================================================
    if use_env:
        # --- Step 13/15: merge per-query UniRef + env a3m rows ---
        # Why:    each query gets a single a3m comprising UniRef hits
        #         followed by env-DB hits (ColabFold's standard MSA layout).
        # Input:  base/qdb (driver of merged keys),
        #         base/uniref.a3m, base/bfd.mgnify30.metaeuk30.smag30.a3m
        # Output: base/final.a3m (DB; either unpacked in step 14 or kept)
        # Why these params: none (mergedbs concatenates by key)
        # Upstream reference: colabfold/mmseqs/search.py:188
        run_mmseqs(mmseqs, ["mergedbs",
                            base.joinpath("qdb"),
                            base.joinpath("final.a3m"),
                            base.joinpath("uniref.a3m"),
                            base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m")])

        # --- Cleanup: drop the per-DB a3m intermediates ---
        # Why:    only final.a3m matters from here on.
        # Input:  the two per-DB a3m intermediates from steps 7 & 12
        # Output: none
        # Why these params: none
        # Upstream reference: colabfold/mmseqs/search.py:189-190
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m")])
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("uniref.a3m")])
    else:
        # --- Step 13/15 (no-env variant): rename uniref.a3m → final.a3m ---
        # Why:    when use_env is False, the UniRef MSA *is* the final MSA;
        #         mvdb is the cheapest way to give it the canonical name.
        # Input:  base/uniref.a3m
        # Output: base/final.a3m (DB; either unpacked in step 14 or kept)
        # Why these params: none
        # Upstream reference: colabfold/mmseqs/search.py:192
        run_mmseqs(mmseqs, ["mvdb",
                            base.joinpath("uniref.a3m"),
                            base.joinpath("final.a3m")])

        # --- Cleanup: remove the now-empty uniref.a3m handle ---
        # Why:    after mvdb the source DB index is gone but the wrapper
        #         entry can linger — rmdb is a no-op or scrubber here.
        # Input:  base/uniref.a3m
        # Output: none
        # Why these params: none
        # Upstream reference: colabfold/mmseqs/search.py:193
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("uniref.a3m")])

    # ======================================================================
    # Unpack final.a3m into one .a3m file per query sequence
    # ======================================================================
    if unpack:
        # --- Step 14/15: unpack final.a3m DB into loose files ---
        # Why:    callers want one ``<id>.a3m`` per query (we rename them
        #         to ``<safe_filename(jobname)>.a3m`` in main()).
        # Input:  base/final.a3m (DB)
        # Output: base/<id>.a3m for each entry in qdb.lookup
        # Why these params:
        #   --unpack-name-mode 0 = name files by numeric id (we rename later)
        #   --unpack-suffix .a3m = file extension to append
        # Upstream reference: colabfold/mmseqs/search.py:196
        run_mmseqs(mmseqs, ["unpackdb",
                            base.joinpath("final.a3m"),
                            base.joinpath("."),
                            "--unpack-name-mode", "0",
                            "--unpack-suffix", ".a3m"])

        # --- Cleanup: drop the final.a3m DB ---
        # Why:    the loose .a3m files are now the source of truth.
        # Input:  base/final.a3m
        # Output: none
        # Why these params: none
        # Upstream reference: colabfold/mmseqs/search.py:197
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("final.a3m")])

    # --- Step 15/15: drop the profile DB & header symlink ---
    # Why:    the profile is no longer needed once all MSAs are built.
    # Input:  base/prof_res, base/prof_res_h
    # Output: none
    # Why these params: none
    # Upstream reference: colabfold/mmseqs/search.py:204-205
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("prof_res")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("prof_res_h")])

    # Plain Python cleanup — tmp/ holds search-internal state; tmp3/ only
    # exists when use_env is True (env-DB search wrote there).
    shutil.rmtree(base.joinpath("tmp"))
    if use_env:
        shutil.rmtree(base.joinpath("tmp3"))


# ---------------------------------------------------------------------------
# Multimer (paired) search
# ---------------------------------------------------------------------------

def mmseqs_search_pair(
    dbbase: Path,
    base: Path,
    uniref_db: Path = Path("uniref30_2302_db"),
    spire_db: Path = Path("spire_ctg10_2401_db"),
    mmseqs: Path = Path("mmseqs"),
    pair_env: bool = False,
    filter: bool = False,
    prefilter_mode: int = 0,
    s: float = 8,
    threads: int = 64,
    # Diverges from upstream: gpu/gpu_server typed as int (0/1) instead of
    # bool to match our CLI's int-style flags (and so the value can be
    # forwarded into --gpu / --gpu-server argv unchanged).
    gpu: int = 0,
    gpu_server: int = 0,
    db_load_mode: int = 2,
    pairing_strategy: int = 0,
    unpack: bool = True,
) -> None:
    """Run ColabFold's paired (multimer) MSA pipeline.

    Upstream reference: colabfold/mmseqs/search.py:212-290

    Pipeline outline:
      1. Iterative profile search vs UniRef30 (or SPIRE if pair_env)
      2. Move iter-1 profile out of tmp/
      3. expandaln through cluster members
      4. align expanded hits against profile
      5. pairaln (pairing-dummy-mode 0) to keep only species-paired hits
      6. align paired hits w/ backtrace for accurate MSA construction
      7. pairaln (pairing-dummy-mode 1) to insert gap-only rows for
         species that lack one chain (needed for downstream pairing)
      8. result2msa → pair.a3m
      9. unpackdb → <id>.paired.a3m or <id>.env.paired.a3m

    The function is invoked twice when use_env_pairing is on:
    once with pair_env=False (UniRef30) and once with pair_env=True (SPIRE).
    """
    if not dbbase.joinpath(f"{uniref_db}.dbtype").is_file():
        raise FileNotFoundError(f"Database {uniref_db} does not exist")

    # Diverges from upstream: hoisted dbSuffix1/dbSuffix2 = ".idx" defaults
    # out of the if/else; semantically equivalent because the indexed branch
    # in upstream sets both to ".idx" anyway.
    dbSuffix1 = ".idx"
    dbSuffix2 = ".idx"
    if (
        (
            not dbbase.joinpath(f"{uniref_db}.idx").is_file()
            and not dbbase.joinpath(f"{uniref_db}.idx.index").is_file()
        )
        or os.environ.get("MMSEQS_IGNORE_INDEX", False)
    ):
        logger.info("Search does not use index")
        db_load_mode = 0
        dbSuffix1 = "_seq"
        dbSuffix2 = "_aln"

    if pair_env:
        db = spire_db
        output = ".env.paired.a3m"
    else:
        db = uniref_db
        output = ".paired.a3m"

    # ---- Shared param blocks (upstream lines 255-267) -----------------
    search_param: List[str] = [
        "--num-iterations", "3",
        "--db-load-mode", str(db_load_mode),
        "-a",
        "-e", "0.1",
        "--max-seqs", "10000",
    ]
    if gpu:
        search_param += ["--gpu", str(gpu), "--prefilter-mode", "1"]
    else:
        search_param += ["--prefilter-mode", str(prefilter_mode)]
        if s is not None:
            search_param += ["-s", "{:.1f}".format(s)]
        else:
            search_param += ["--k-score", "'seq:96,prof:80'"]
    if gpu_server:
        search_param += ["--gpu-server", str(gpu_server)]

    expand_param = [
        "--expansion-mode", "0",
        "-e", "inf",
        "--expand-filter-clusters", "0",
        "--max-seq-id", "0.95",
    ]
    filter_param = [
        "--filter-msa", str(1 if filter else 0),
        "--filter-min-enable", "1000",
        "--diff", "3000",
        "--qid", "0.2,0.4,0.6,0.8,1.0",
        "--qsc", "0",
        "--max-seq-id", "0.95",
    ]

    # --- Pair-Step 1/9: profile search vs pairing DB ---
    # Why:    same role as monomer step 1, but against either UniRef30 (no
    #         env pairing) or SPIRE (env pairing).
    # Input:  base/qdb, dbbase/<db>
    # Output: base/res, base/tmp/latest/profile_1
    # Why these params: see monomer step 1 (identical search_param block)
    # Upstream reference: colabfold/mmseqs/search.py:268
    run_mmseqs(mmseqs, ["search",
                        base.joinpath("qdb"),
                        dbbase.joinpath(db),
                        base.joinpath("res"),
                        base.joinpath("tmp"),
                        "--threads", str(threads)] + search_param)

    # --- Pair-Step 2/9: claim iter-1 profile out of tmp/ ---
    # Why:    keep the profile alive past the tmp/ cleanup at end of run.
    # Input:  base/tmp/latest/profile_1
    # Output: base/prof_res
    # Why these params: none
    # Upstream reference: colabfold/mmseqs/search.py:269
    run_mmseqs(mmseqs, ["mvdb",
                        base.joinpath("tmp/latest/profile_1"),
                        base.joinpath("prof_res")])

    # --- Pair-Step 3/9: symlink query headers onto profile DB ---
    # Why:    profile DB needs a header companion (see monomer step 3).
    # Input:  base/qdb_h
    # Output: base/prof_res_h
    # Why these params: none
    # Upstream reference: colabfold/mmseqs/search.py:270
    run_mmseqs(mmseqs, ["lndb",
                        base.joinpath("qdb_h"),
                        base.joinpath("prof_res_h")])

    # --- Pair-Step 4/9: expand alignments through cluster members ---
    # Why:    see monomer step 4; same logic.
    # Input:  base/qdb, dbbase/<db>.idx (seq + aln), base/res
    # Output: base/res_exp
    # Why these params: ``-e inf`` keeps every cluster member (pairing
    #         filtering happens later in pairaln, not here).
    # Upstream reference: colabfold/mmseqs/search.py:271
    run_mmseqs(mmseqs, ["expandaln",
                        base.joinpath("qdb"),
                        dbbase.joinpath(f"{db}{dbSuffix1}"),
                        base.joinpath("res"),
                        dbbase.joinpath(f"{db}{dbSuffix2}"),
                        base.joinpath("res_exp"),
                        "--db-load-mode", str(db_load_mode),
                        "--threads", str(threads)] + expand_param)

    # --- Pair-Step 5/9: align expanded hits against profile ---
    # Why:    pairaln needs alignment scores + e-values; align populates
    #         them via Smith-Waterman over the profile.
    # Input:  base/prof_res, dbbase/<db>.idx, base/res_exp
    # Output: base/res_exp_realign
    # Why these params:
    #   --alignment-mode 1 = score + end_pos only (cheap; we re-align with
    #                        backtrace in step 7 after pairing prunes hits)
    #   -e 0.001           = tighter e-value than monomer (paired search
    #                        is more selective)
    #   --max-accept 1e6   = accept many alignments (pairaln will prune)
    # Upstream reference: colabfold/mmseqs/search.py:272
    run_mmseqs(mmseqs, ["align",
                        base.joinpath("prof_res"),
                        dbbase.joinpath(f"{db}{dbSuffix1}"),
                        base.joinpath("res_exp"),
                        base.joinpath("res_exp_realign"),
                        "--db-load-mode", str(db_load_mode),
                        "--alignment-mode", "1",
                        "-e", "0.001",
                        "--max-accept", "1000000",
                        "--threads", str(threads)])

    # --- Pair-Step 6/9: pair alignments by species (first pass) ---
    # Why:    keep only hits whose species are present for every query
    #         chain — this is the actual "pairing" step that ColabFold's
    #         multimer mode depends on for co-evolution signal.
    # Input:  base/qdb, dbbase/<db>, base/res_exp_realign
    # Output: base/res_exp_realign_pair (per-chain pruned hit DB,
    #         consumed by step 7)
    # Why these params:
    #   --pairing-mode <strategy>  0 = pair maximal per species (greedy),
    #                              1 = pair only if all chains covered
    #   --pairing-dummy-mode 0     do NOT insert dummy rows yet (we want
    #                              clean re-alignments in step 7)
    # Upstream reference: colabfold/mmseqs/search.py:273
    run_mmseqs(mmseqs, ["pairaln",
                        base.joinpath("qdb"),
                        dbbase.joinpath(f"{db}"),
                        base.joinpath("res_exp_realign"),
                        base.joinpath("res_exp_realign_pair"),
                        "--db-load-mode", str(db_load_mode),
                        "--pairing-mode", str(pairing_strategy),
                        "--pairing-dummy-mode", "0",
                        "--threads", str(threads)])

    # --- Pair-Step 7/9: re-align paired hits with backtrace ---
    # Why:    step 5's alignments lacked backtraces (alignment-mode 1);
    #         we need them now to materialise the paired MSA in step 9.
    # Input:  base/prof_res, dbbase/<db>.idx, base/res_exp_realign_pair
    # Output: base/res_exp_realign_pair_bt (BT = with backtrace; input
    #         to the second pairaln in step 8)
    # Why these params:
    #   -e inf = accept every paired hit (pairing already pruned)
    #   -a     = produce backtrace
    # Upstream reference: colabfold/mmseqs/search.py:274
    run_mmseqs(mmseqs, ["align",
                        base.joinpath("prof_res"),
                        dbbase.joinpath(f"{db}{dbSuffix1}"),
                        base.joinpath("res_exp_realign_pair"),
                        base.joinpath("res_exp_realign_pair_bt"),
                        "--db-load-mode", str(db_load_mode),
                        "-e", "inf",
                        "-a",
                        "--threads", str(threads)])

    # --- Pair-Step 8/9: pair alignments (second pass, with dummy rows) ---
    # Why:    insert gap-only rows for species missing a chain so the
    #         row-aligned a3m written by result2msa has consistent length
    #         across query chains (required by ColabFold/Boltz multimer
    #         input format).
    # Input:  base/qdb, dbbase/<db>, base/res_exp_realign_pair_bt
    # Output: base/res_final (input to step 9)
    # Why these params:
    #   --pairing-mode <strategy>  (same as step 6)
    #   --pairing-dummy-mode 1     YES insert dummy rows for missing chains
    # Upstream reference: colabfold/mmseqs/search.py:275
    run_mmseqs(mmseqs, ["pairaln",
                        base.joinpath("qdb"),
                        dbbase.joinpath(f"{db}"),
                        base.joinpath("res_exp_realign_pair_bt"),
                        base.joinpath("res_final"),
                        "--db-load-mode", str(db_load_mode),
                        "--pairing-mode", str(pairing_strategy),
                        "--pairing-dummy-mode", "1",
                        "--threads", str(threads)])

    # --- Pair-Step 9/9: build the paired a3m MSA ---
    # Why:    write out the row-aligned paired MSA, ready for unpack.
    # Input:  base/qdb, dbbase/<db>.idx, base/res_final
    # Output: base/pair.a3m (DB; unpacked below)
    # Why these params:
    #   --msa-format-mode 5 = A3M (no alignment-info comment lines —
    #                         pair_sequences in _colabfold_helpers will
    #                         glue per-chain a3m's so we don't want the
    #                         per-row metadata that mode 6 emits)
    #   filter_param         = diversity filter applied during MSA build
    # Upstream reference: colabfold/mmseqs/search.py:276
    run_mmseqs(mmseqs, ["result2msa",
                        base.joinpath("qdb"),
                        dbbase.joinpath(f"{db}{dbSuffix1}"),
                        base.joinpath("res_final"),
                        base.joinpath("pair.a3m"),
                        "--db-load-mode", str(db_load_mode),
                        "--msa-format-mode", "5",
                        "--threads", str(threads)] + filter_param)

    if unpack:
        # --- Unpack pair.a3m into loose files ---
        # Why:    callers want one ``<id>.paired.a3m`` (or
        #         ``.env.paired.a3m``) per query.
        # Input:  base/pair.a3m
        # Output: base/<id><output> for each entry in qdb.lookup
        # Why these params: see monomer step 14
        # Upstream reference: colabfold/mmseqs/search.py:278
        run_mmseqs(mmseqs, ["unpackdb",
                            base.joinpath("pair.a3m"),
                            base.joinpath("."),
                            "--unpack-name-mode", "0",
                            "--unpack-suffix", output])

        # --- Cleanup: drop pair.a3m DB ---
        # Why:    loose files are now source of truth.
        # Input:  base/pair.a3m
        # Output: none
        # Why these params: none
        # Upstream reference: colabfold/mmseqs/search.py:279
        run_mmseqs(mmseqs, ["rmdb", base.joinpath("pair.a3m")])

    # --- Cleanup: drop all intermediate pair-side result DBs + profile ---
    # Why:    free disk; not needed after unpack.
    # Input:  the intermediates from steps 1, 4-8 + prof_res
    # Output: none
    # Why these params: none
    # Upstream reference: colabfold/mmseqs/search.py:280-287
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp_realign")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp_realign_pair")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp_realign_pair_bt")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_final")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("prof_res")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("prof_res_h")])

    shutil.rmtree(base.joinpath("tmp"))


# ---------------------------------------------------------------------------
# CLI — invoked by tools.py:colabfold_search_argv()
# ---------------------------------------------------------------------------

def _build_query_db(
    mmseqs: Path,
    base: Path,
    queries_unique: list,
) -> None:
    """Create the qdb sequence DB + lookup file from a single query FASTA.

    Upstream reference: colabfold/mmseqs/search.py:444-475

    Writes a deduplicated FASTA whose sequence headers are integers (the
    upstream convention starts at 101 — kept here for compatibility), then
    runs ``mmseqs createdb`` to produce the qdb DB consumed by every
    subsequent step.  Finally rewrites qdb.lookup so each row maps back to
    the *original* jobname (used by ``unpackdb`` to name output files).
    """
    base.mkdir(exist_ok=True, parents=True)
    # Dedupe + write the canonical FASTA mmseqs consumes.
    canonical_fasta = base.joinpath("query.fas")
    with canonical_fasta.open("w") as f:
        for _job_number, (
            _raw_jobname,
            query_sequences,
            _query_seqs_cardinality,
            _other,
        ) in enumerate(queries_unique):
            for j, seq in enumerate(query_sequences):
                # Upstream convention: first seq header = 101.
                query_seq_headername = 101 + j
                f.write(f">{query_seq_headername}\n{seq}\n")

    # --- Build the sequence DB used as input to every search step ---
    # Why:    mmseqs needs an indexed DB (not a FASTA) as input.
    # Input:  canonical_fasta
    # Output: base/qdb + base/qdb_h (+ index/lookup helpers)
    # Why these params:
    #   --shuffle 0  = preserve input order (otherwise qdb.lookup indexing
    #                  below would be off-by-one)
    #   --dbtype 1   = amino-acid DB
    # Upstream reference: colabfold/mmseqs/search.py:458-461
    run_mmseqs(mmseqs, ["createdb",
                        canonical_fasta,
                        base.joinpath("qdb"),
                        "--shuffle", "0",
                        "--dbtype", "1"])

    # Rewrite qdb.lookup so the integer DB keys map back to the original
    # job names — unpackdb consumes this to name output files.
    # Upstream reference: colabfold/mmseqs/search.py:462-475
    with base.joinpath("qdb.lookup").open("w") as f:
        idx = 0
        file_number = 0
        for _job_number, (
            raw_jobname,
            query_sequences,
            _query_seqs_cardinality,
            _other,
        ) in enumerate(queries_unique):
            for _seq in query_sequences:
                raw_jobname_first = raw_jobname.split()[0]
                f.write(f"{idx}\t{raw_jobname_first}\t{file_number}\n")
                idx += 1
            file_number += 1

    # The canonical FASTA was only needed for createdb.
    canonical_fasta.unlink(missing_ok=True)


def _dedupe_queries(queries: list) -> list:
    """Dedupe per-job query sequences while preserving cardinality counts.

    Upstream reference: colabfold/mmseqs/search.py:427-442
    """
    queries_unique = []
    for _job_number, (raw_jobname, query_sequences, _a3m, other_molecules) in enumerate(queries):
        query_sequences = (
            [query_sequences] if isinstance(query_sequences, str) else query_sequences
        )
        query_seqs_unique: List[str] = []
        for x in query_sequences:
            if x not in query_seqs_unique:
                query_seqs_unique.append(x)
        query_seqs_cardinality = [0] * len(query_seqs_unique)
        for seq in query_sequences:
            seq_idx = query_seqs_unique.index(seq)
            query_seqs_cardinality[seq_idx] += 1
        queries_unique.append(
            [raw_jobname, query_seqs_unique, query_seqs_cardinality, other_molecules]
        )
    return queries_unique


def _emit_complex_outputs(
    base: Path,
    queries_unique: list,
    *,
    keep_unpaired: bool,
    keep_paired: bool,
    unpack: bool,
) -> None:
    """For complex (multimer) queries, concatenate per-chain a3m files
    into the final ``<job>.a3m`` consumed by downstream folding tools.

    Upstream reference: colabfold/mmseqs/search.py:547-591

    Diverges from upstream:
      - Drops the ``--af3-json`` / ``args.af3_json`` branch.
      - Drops the ``use_env_pairing`` env-paired concat (we don't expose
        env pairing in our CLI contract).
    """
    if not unpack:
        return
    idx = 0
    for job_number, (
        _raw_jobname,
        query_sequences,
        query_seqs_cardinality,
        _other,
    ) in enumerate(queries_unique):
        # NB: ``pair_msa`` distinguishes "not provided" from "provided but
        # empty" via ``is None``.  If we default to ``[]`` when the flag is
        # False, pair_msa's branch selector routes to the both-provided path
        # and calls pad_sequences([], ...) which raises IndexError on
        # ``a3m_lines[0]``.  Keep these as None unless the corresponding
        # side was actually collected below.
        unpaired_msa: List[str] | None = [] if keep_unpaired else None
        paired_msa: list | None = (
            [] if (keep_paired and len(query_seqs_cardinality) > 1) else None
        )
        for _seq in query_sequences:
            if keep_unpaired and unpaired_msa is not None:
                with base.joinpath(f"{idx}.a3m").open("r") as f:
                    unpaired_msa.append(f.read())
                base.joinpath(f"{idx}.a3m").unlink(missing_ok=True)
            if keep_paired and paired_msa is not None:
                with base.joinpath(f"{idx}.paired.a3m").open("r") as f:
                    paired_msa.append(f.read())
                base.joinpath(f"{idx}.paired.a3m").unlink(missing_ok=True)
            idx += 1
        msa = msa_to_str(
            unpaired_msa, paired_msa, query_sequences, query_seqs_cardinality
        )
        base.joinpath(f"{job_number}.a3m").write_text(msa)


def main() -> None:
    """CLI entry: parse args, prep qdb, dispatch to monomer / pair, finalise.

    Diverges from upstream's argparse:
      - Uses long-only flags matching the contract in
        ``services/mmseqs2-server/tools.py:colabfold_search_argv()``.
      - Drops ``--use-templates`` / ``--db2`` / ``--db4`` /
        ``--use-env-pairing`` / ``--af3-json`` / ``--af3-msa-as-path``.
      - Requires explicit ``--db-dir`` + ``--output-dir`` + ``--query``
        (no positional args — easier to wrap from Python).
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--query", type=Path, required=True,
                        help="Single FASTA query file.")
    parser.add_argument("--db-dir", type=Path, required=True,
                        help="Root directory containing pre-built MMseqs2 DBs.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write .a3m output + intermediate state.")
    parser.add_argument("--mmseqs", type=Path,
                        default=Path("/opt/mmseqs-gpu/bin/mmseqs"),
                        help="Path to the mmseqs binary.")
    parser.add_argument("--db1", type=Path, required=True,
                        help="UniRef30 DB name (relative to --db-dir).")
    parser.add_argument("--db3", type=Path, default=None,
                        help="Env DB name (required when --use-env 1).")
    parser.add_argument("--use-env", type=int, default=0, choices=[0, 1],
                        help="Whether to run the env-DB block (extends MSA).")
    parser.add_argument("--filter", type=int, default=1, choices=[0, 1],
                        help="Whether to apply the strict align/qsc filter.")
    parser.add_argument("--pair-mode", type=str, default="unpaired",
                        choices=["unpaired", "paired"],
                        help="MSA mode: unpaired (monomer) or paired (multimer).")
    parser.add_argument("--pairing-strategy", type=int, default=None,
                        choices=[0, 1],
                        help="pairaln strategy (required when pair-mode=paired). "
                             "0 = greedy per species, 1 = only if all chains covered.")
    parser.add_argument("--gpu", type=int, default=0, choices=[0, 1],
                        help="Use mmseqs-GPU (1) or CPU (0).")
    parser.add_argument("--threads", type=int, default=4,
                        help="CPU threads handed to each mmseqs invocation.")
    parser.add_argument("--db-load-mode", type=int, default=2,
                        help="DB load mode (0 auto, 1 fread, 2 mmap, 3 mmap+touch).")
    parser.add_argument("--unpack", type=int, default=1, choices=[0, 1],
                        help="Unpack final DBs to loose .a3m files.")

    args = parser.parse_args()

    # Validation -------------------------------------------------------
    if args.pair_mode == "paired" and args.pairing_strategy is None:
        parser.error("--pair-mode paired requires --pairing-strategy")
    if args.use_env == 1 and args.db3 is None:
        parser.error("--use-env 1 requires --db3")

    logging.basicConfig(level=logging.INFO)

    # Load + dedupe queries -------------------------------------------
    queries, is_complex = get_queries(args.query)
    queries_unique = _dedupe_queries(queries)

    # Build qdb + lookup ----------------------------------------------
    _build_query_db(args.mmseqs, args.output_dir, queries_unique)

    keep_paired = args.pair_mode == "paired"
    keep_unpaired = args.pair_mode == "unpaired"

    # Dispatch ---------------------------------------------------------
    if keep_unpaired:
        logger.info("Dispatching to mmseqs_search_monomer (unpaired mode)")
        mmseqs_search_monomer(
            mmseqs=args.mmseqs,
            dbbase=args.db_dir,
            base=args.output_dir,
            uniref_db=args.db1,
            metagenomic_db=(args.db3 if args.db3 is not None else Path("")),
            use_env=bool(args.use_env),
            filter=bool(args.filter),
            db_load_mode=args.db_load_mode,
            threads=args.threads,
            gpu=args.gpu,
            unpack=bool(args.unpack),
        )

    if is_complex and keep_paired:
        logger.info("Dispatching to mmseqs_search_pair (paired mode)")
        mmseqs_search_pair(
            mmseqs=args.mmseqs,
            dbbase=args.db_dir,
            base=args.output_dir,
            uniref_db=args.db1,
            filter=False,
            db_load_mode=args.db_load_mode,
            threads=args.threads,
            gpu=args.gpu,
            pairing_strategy=args.pairing_strategy if args.pairing_strategy is not None else 0,
            pair_env=False,
            unpack=bool(args.unpack),
        )
    elif keep_paired and not is_complex:
        logger.warning(
            "pair-mode=paired requested but query is a monomer; "
            "skipping paired search."
        )

    # Multimer post-processing: concatenate per-chain a3m's ------------
    if is_complex:
        _emit_complex_outputs(
            args.output_dir,
            queries_unique,
            keep_unpaired=keep_unpaired,
            keep_paired=keep_paired,
            unpack=bool(args.unpack),
        )

    # Rename numeric a3m files → <safe_filename(jobname)>.a3m ----------
    output_paths: List[Path] = []
    if args.unpack:
        for job_number, (raw_jobname, _qs, _qsc, _other) in enumerate(queries_unique):
            src = args.output_dir.joinpath(f"{job_number}.a3m")
            dst = args.output_dir.joinpath(f"{safe_filename(raw_jobname)}.a3m")
            if src.exists() and src != dst:
                os.rename(src, dst)
            output_paths.append(dst)

        # --- Cleanup: drop the query DB + its header companion ---
        # Why:    qdb / qdb_h were only needed as inputs to search /
        #         expandaln / align / etc.; once the loose .a3m files are
        #         renamed they have no further role.
        # Input:  output_dir/qdb, output_dir/qdb_h
        # Output: none (DBs deleted)
        # Why these params: none (rmdb takes one positional)
        # Upstream reference: colabfold/mmseqs/search.py:627-628
        run_mmseqs(args.mmseqs, ["rmdb", args.output_dir.joinpath("qdb")])
        run_mmseqs(args.mmseqs, ["rmdb", args.output_dir.joinpath("qdb_h")])

    logger.info("colabfold-search complete; pair_mode=%s outputs=%s",
                args.pair_mode,
                [str(p) for p in output_paths])
    for p in output_paths:
        print(str(p))


if __name__ == "__main__":
    main()
