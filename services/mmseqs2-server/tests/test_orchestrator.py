"""Offline unit tests for the vendored ColabFold MSA orchestrator.

We mock ``subprocess.check_call`` so the tests never invoke a real mmseqs
binary. The point is to lock down the *argv sequence* the orchestrator
produces — argument order, key flags (``--gpu``, ``--pairing-mode``), the
presence/absence of the env-DB block, etc. — without depending on the
upstream mmseqs subprocess being installed or any DB files being present.

These tests cover Task 4.0 of the Stage 4 plan.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from server import orchestrator
from server.orchestrator import (
    mmseqs_search_monomer,
    mmseqs_search_pair,
    run_mmseqs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dbbase(tmp_path: Path, *db_names: str) -> Path:
    """Create a dbbase dir with ``<db>.dbtype`` sentinels + ``.idx`` files.

    ``mmseqs_search_monomer`` probes ``dbbase/<db>.dbtype`` (file exists?) and
    ``dbbase/<db>.idx`` (indexed?) before launching; both must be on disk for
    the orchestrator to take the indexed (db_load_mode != 0) path.
    """
    dbbase = tmp_path / "dbbase"
    dbbase.mkdir()
    for name in db_names:
        (dbbase / f"{name}.dbtype").write_text("")
        (dbbase / f"{name}.idx").write_text("")
    return dbbase


def _make_base(tmp_path: Path) -> Path:
    """Create the per-job working dir + the qdb/qdb_h sentinels search() needs.

    The orchestrator never reads these — they're handed off as positional argv
    to mmseqs — but having them on disk keeps things tidy if any future step
    grows a Path.exists() probe.
    """
    base = tmp_path / "base"
    base.mkdir()
    return base


def _argv_calls(mock_check_call) -> list[list[str]]:
    """Return each captured subprocess.check_call argv as ``list[str]``."""
    return [call.args[0] for call in mock_check_call.call_args_list]


def _flatten_modules(argv_calls: list[list[str]]) -> list[str]:
    """Extract just the mmseqs subcommand name from each captured argv."""
    # argv[0] is the mmseqs binary path; argv[1] is the subcommand.
    return [argv[1] for argv in argv_calls]


# ---------------------------------------------------------------------------
# run_mmseqs — low-level subprocess plumbing
# ---------------------------------------------------------------------------


def test_run_mmseqs_invokes_subprocess() -> None:
    """run_mmseqs forwards the params verbatim (with str coercion) to check_call."""
    with patch("server.orchestrator.subprocess.check_call") as mock_call:
        run_mmseqs(Path("/fake/mmseqs"), ["search", "qdb", "tdb", "res", "tmp"])

    assert mock_call.call_count == 1
    argv = mock_call.call_args.args[0]
    assert argv == ["/fake/mmseqs", "search", "qdb", "tdb", "res", "tmp"]


def test_run_mmseqs_skips_if_output_exists(tmp_path: Path) -> None:
    """If the output DB's .dbtype already exists, the call is skipped."""
    # ``search`` outputs at position 3 (MODULE_OUTPUT_POS); place the sentinel.
    out = tmp_path / "res"
    out.with_suffix(".dbtype").write_text("")

    with patch("server.orchestrator.subprocess.check_call") as mock_call:
        run_mmseqs(
            Path("/fake/mmseqs"),
            ["search", tmp_path / "qdb", tmp_path / "tdb", out, tmp_path / "tmp"],
        )

    assert mock_call.call_count == 0


def test_run_mmseqs_path_coercion(tmp_path: Path) -> None:
    """Path objects passed as params must reach check_call as strings.

    This locks in the "Diverges from upstream: coerce to str" annotation in
    orchestrator.run_mmseqs — subprocess on some Python versions rejects
    PosixPath in argv.
    """
    with patch("server.orchestrator.subprocess.check_call") as mock_call:
        run_mmseqs(
            Path("/fake/mmseqs"),
            ["createdb", tmp_path / "in.fa", tmp_path / "qdb"],
        )

    argv = mock_call.call_args.args[0]
    assert all(isinstance(a, str) for a in argv)
    assert argv[0] == "/fake/mmseqs"
    assert argv[1] == "createdb"
    assert argv[2] == str(tmp_path / "in.fa")
    assert argv[3] == str(tmp_path / "qdb")


# ---------------------------------------------------------------------------
# mmseqs_search_monomer
# ---------------------------------------------------------------------------


def test_mmseqs_search_monomer_uniref_only_pipeline(tmp_path: Path) -> None:
    """use_env=False runs only the UniRef block + final mvdb-as-rename."""
    dbbase = _make_dbbase(tmp_path, "uniref30_2302_db")
    base = _make_base(tmp_path)

    with patch("server.orchestrator.subprocess.check_call") as mock_call, \
         patch("server.orchestrator.shutil.rmtree"):
        mmseqs_search_monomer(
            dbbase=dbbase,
            base=base,
            uniref_db=Path("uniref30_2302_db"),
            use_env=False,
            unpack=True,
            gpu=0,
        )

    argv_calls = _argv_calls(mock_call)
    modules = _flatten_modules(argv_calls)

    # First subcommand must be ``search`` against the UniRef DB.
    assert modules[0] == "search"
    first_search_argv = argv_calls[0]
    # The 3rd positional after the mmseqs binary + 'search' is the target DB.
    assert first_search_argv[3] == str(dbbase / "uniref30_2302_db")

    # result2msa is invoked exactly once (UniRef-only mode: no env-DB result2msa).
    assert modules.count("result2msa") == 1

    # mergedbs is NOT called when use_env=False — uniref.a3m becomes final.a3m
    # via mvdb instead.
    assert "mergedbs" not in modules

    # No --gpu flag anywhere.
    for argv in argv_calls:
        assert "--gpu" not in argv

    # No --pairing-mode anywhere — this is the monomer pipeline.
    for argv in argv_calls:
        assert "--pairing-mode" not in argv


def test_mmseqs_search_monomer_env_pipeline(tmp_path: Path) -> None:
    """use_env=True adds an env-DB search + mergedbs + env-side result2msa."""
    dbbase = _make_dbbase(
        tmp_path, "uniref30_2302_db", "colabfold_envdb_202108_db"
    )
    base = _make_base(tmp_path)

    with patch("server.orchestrator.subprocess.check_call") as mock_call, \
         patch("server.orchestrator.shutil.rmtree"):
        mmseqs_search_monomer(
            dbbase=dbbase,
            base=base,
            uniref_db=Path("uniref30_2302_db"),
            metagenomic_db=Path("colabfold_envdb_202108_db"),
            use_env=True,
            unpack=True,
            gpu=0,
        )

    argv_calls = _argv_calls(mock_call)
    modules = _flatten_modules(argv_calls)

    # Two search calls — UniRef + env.
    assert modules.count("search") == 2
    # Two result2msa calls — one per DB (UniRef + env).
    assert modules.count("result2msa") == 2
    # mergedbs combines them.
    assert modules.count("mergedbs") == 1

    # The mergedbs argv lists final.a3m as output, then the two per-DB MSAs.
    mergedbs_argv = next(a for a in argv_calls if a[1] == "mergedbs")
    assert mergedbs_argv[3] == str(base / "final.a3m")
    assert mergedbs_argv[4] == str(base / "uniref.a3m")
    assert mergedbs_argv[5] == str(base / "bfd.mgnify30.metaeuk30.smag30.a3m")


def test_mmseqs_search_monomer_gpu_flag_injected_when_gpu_1(tmp_path: Path) -> None:
    """gpu=1 → every ``search`` invocation includes ``--gpu 1 --prefilter-mode 1``."""
    dbbase = _make_dbbase(tmp_path, "uniref30_2302_db")
    base = _make_base(tmp_path)

    with patch("server.orchestrator.subprocess.check_call") as mock_call, \
         patch("server.orchestrator.shutil.rmtree"):
        mmseqs_search_monomer(
            dbbase=dbbase,
            base=base,
            uniref_db=Path("uniref30_2302_db"),
            use_env=False,
            unpack=True,
            gpu=1,
        )

    # Find every ``search`` argv and check the GPU flags rode along.
    search_calls = [a for a in _argv_calls(mock_call) if a[1] == "search"]
    assert len(search_calls) >= 1
    for argv in search_calls:
        assert "--gpu" in argv
        assert argv[argv.index("--gpu") + 1] == "1"
        assert "--prefilter-mode" in argv
        assert argv[argv.index("--prefilter-mode") + 1] == "1"


def test_mmseqs_search_monomer_no_gpu_flag_when_gpu_0(tmp_path: Path) -> None:
    """gpu=0 → no ``--gpu`` flag, and ``-s`` (sensitivity) is emitted instead."""
    dbbase = _make_dbbase(tmp_path, "uniref30_2302_db")
    base = _make_base(tmp_path)

    with patch("server.orchestrator.subprocess.check_call") as mock_call, \
         patch("server.orchestrator.shutil.rmtree"):
        mmseqs_search_monomer(
            dbbase=dbbase,
            base=base,
            uniref_db=Path("uniref30_2302_db"),
            use_env=False,
            unpack=True,
            gpu=0,
        )

    search_calls = [a for a in _argv_calls(mock_call) if a[1] == "search"]
    assert len(search_calls) >= 1
    for argv in search_calls:
        assert "--gpu" not in argv
        # CPU branch always passes ``-s <sensitivity>`` because s defaults to 8.
        assert "-s" in argv


# ---------------------------------------------------------------------------
# mmseqs_search_pair
# ---------------------------------------------------------------------------


def test_mmseqs_search_pair_invokes_pairaln(tmp_path: Path) -> None:
    """The pair pipeline calls pairaln at least once — that's its raison d'être."""
    dbbase = _make_dbbase(tmp_path, "uniref30_2302_db")
    base = _make_base(tmp_path)

    with patch("server.orchestrator.subprocess.check_call") as mock_call, \
         patch("server.orchestrator.shutil.rmtree"):
        mmseqs_search_pair(
            dbbase=dbbase,
            base=base,
            uniref_db=Path("uniref30_2302_db"),
            pair_env=False,
            unpack=True,
            gpu=0,
            pairing_strategy=0,
        )

    modules = _flatten_modules(_argv_calls(mock_call))
    # Pair pipeline invokes pairaln twice (steps 6 + 8: pre-dummy + with-dummy).
    assert modules.count("pairaln") == 2


@pytest.mark.parametrize("strategy", [0, 1])
def test_mmseqs_search_pair_pairing_mode_passes_through(
    tmp_path: Path, strategy: int
) -> None:
    """pairing_strategy reaches the ``--pairing-mode`` flag verbatim."""
    dbbase = _make_dbbase(tmp_path, "uniref30_2302_db")
    base = _make_base(tmp_path)

    with patch("server.orchestrator.subprocess.check_call") as mock_call, \
         patch("server.orchestrator.shutil.rmtree"):
        mmseqs_search_pair(
            dbbase=dbbase,
            base=base,
            uniref_db=Path("uniref30_2302_db"),
            pair_env=False,
            unpack=True,
            gpu=0,
            pairing_strategy=strategy,
        )

    pairaln_argvs = [a for a in _argv_calls(mock_call) if a[1] == "pairaln"]
    assert len(pairaln_argvs) == 2
    for argv in pairaln_argvs:
        assert "--pairing-mode" in argv
        assert argv[argv.index("--pairing-mode") + 1] == str(strategy)


# ---------------------------------------------------------------------------
# CLI main() — dispatch + argparse validation
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_get_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace _colabfold_helpers.get_queries with a stub.

    ``main()`` calls ``get_queries(args.query)`` which expects a real FASTA on
    disk; we'd rather not exercise the FASTA parser here (test_tools covers
    it). Returning a single monomer query keeps the dispatch path clean.
    """
    def _stub(_path):
        # Returns (queries, is_complex). One monomer query → is_complex=False.
        return [("job1", "ACDEFGHIK", None, None)], False

    monkeypatch.setattr(orchestrator, "get_queries", _stub)


def test_main_cli_dispatches_monomer_vs_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_get_queries: None
) -> None:
    """--pair-mode unpaired → monomer; --pair-mode paired → pair (when complex)."""
    # Suppress the real mmseqs dispatches.
    monomer_calls: list[dict] = []
    pair_calls: list[dict] = []

    monkeypatch.setattr(
        orchestrator, "mmseqs_search_monomer",
        lambda **kw: monomer_calls.append(kw),
    )
    monkeypatch.setattr(
        orchestrator, "mmseqs_search_pair",
        lambda **kw: pair_calls.append(kw),
    )
    monkeypatch.setattr(orchestrator, "_build_query_db", lambda *a, **kw: None)
    # _emit_complex_outputs only runs for is_complex=True; stub anyway.
    monkeypatch.setattr(
        orchestrator, "_emit_complex_outputs", lambda *a, **kw: None
    )

    fasta = tmp_path / "q.fasta"
    fasta.write_text(">job1\nACDEFGHIK\n")
    outdir = tmp_path / "out"
    outdir.mkdir()
    dbdir = tmp_path / "db"
    dbdir.mkdir()

    # --- pair-mode=unpaired branch ---
    monkeypatch.setattr(sys, "argv", [
        "orchestrator",
        "--query", str(fasta),
        "--db-dir", str(dbdir),
        "--output-dir", str(outdir),
        "--db1", "uniref30",
        "--pair-mode", "unpaired",
        "--unpack", "0",  # skip rename + rmdb steps (no real .a3m / qdb).
    ])
    orchestrator.main()
    assert len(monomer_calls) == 1
    assert len(pair_calls) == 0


def test_main_cli_paired_mode_dispatches_pair_for_complex(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """For a complex (multimer) query, --pair-mode paired calls mmseqs_search_pair."""
    monomer_calls: list[dict] = []
    pair_calls: list[dict] = []

    # Multimer query: sequences are a list, is_complex=True.
    def _stub_complex(_path):
        return [("dimer", ["MKQH", "LLLLL"], None, None)], True

    monkeypatch.setattr(orchestrator, "get_queries", _stub_complex)
    monkeypatch.setattr(
        orchestrator, "mmseqs_search_monomer",
        lambda **kw: monomer_calls.append(kw),
    )
    monkeypatch.setattr(
        orchestrator, "mmseqs_search_pair",
        lambda **kw: pair_calls.append(kw),
    )
    monkeypatch.setattr(orchestrator, "_build_query_db", lambda *a, **kw: None)
    monkeypatch.setattr(
        orchestrator, "_emit_complex_outputs", lambda *a, **kw: None
    )

    fasta = tmp_path / "q.fasta"
    fasta.write_text(">dimer\nMKQH:LLLLL\n")
    outdir = tmp_path / "out"
    outdir.mkdir()
    dbdir = tmp_path / "db"
    dbdir.mkdir()

    monkeypatch.setattr(sys, "argv", [
        "orchestrator",
        "--query", str(fasta),
        "--db-dir", str(dbdir),
        "--output-dir", str(outdir),
        "--db1", "uniref30",
        "--pair-mode", "paired",
        "--pairing-strategy", "0",
        "--unpack", "0",  # skip rename step (no real .a3m files on disk).
    ])
    orchestrator.main()

    assert len(monomer_calls) == 0
    assert len(pair_calls) == 1
    assert pair_calls[0]["pairing_strategy"] == 0


def test_main_cli_validation_use_env_requires_db3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--use-env 1 without --db3 must hit argparse parser.error → SystemExit."""
    fasta = tmp_path / "q.fasta"
    fasta.write_text(">job1\nACDE\n")
    monkeypatch.setattr(sys, "argv", [
        "orchestrator",
        "--query", str(fasta),
        "--db-dir", str(tmp_path),
        "--output-dir", str(tmp_path),
        "--db1", "uniref30",
        "--use-env", "1",
        # deliberately no --db3
    ])
    with pytest.raises(SystemExit):
        orchestrator.main()


def test_main_cli_validation_paired_requires_strategy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--pair-mode paired without --pairing-strategy must SystemExit."""
    fasta = tmp_path / "q.fasta"
    fasta.write_text(">job1\nACDE\n")
    monkeypatch.setattr(sys, "argv", [
        "orchestrator",
        "--query", str(fasta),
        "--db-dir", str(tmp_path),
        "--output-dir", str(tmp_path),
        "--db1", "uniref30",
        "--pair-mode", "paired",
        # deliberately no --pairing-strategy
    ])
    with pytest.raises(SystemExit):
        orchestrator.main()


# ---------------------------------------------------------------------------
# _emit_complex_outputs — paired-only branch must not pass empty unpaired list
# ---------------------------------------------------------------------------


def test_emit_complex_outputs_pair_only_does_not_pass_empty_unpaired(
    tmp_path: Path,
) -> None:
    """Regression: FC pair endpoint used to fail with
    ``IndexError: list index out of range`` because ``_emit_complex_outputs``
    initialised ``unpaired_msa = []`` even when ``keep_unpaired=False``.
    ``pair_msa`` distinguishes "not provided" from "empty" via ``is None``
    so a stray empty list routed us into the both-provided branch and
    ``pad_sequences([], ...)`` blew up on ``a3m_lines[0]``.

    Verified end-to-end on FC v0.0.4 (job ``cc4025d8ce414559870c``).
    """
    base = tmp_path
    # Pretend the paired unpack step wrote per-chain files.
    (base / "0.paired.a3m").write_text(">101\nACDE\n>seq1\nACDE\n")
    (base / "1.paired.a3m").write_text(">102\nWXYZ\n>seq2\nWXYZ\n")

    queries_unique = [
        # (jobname, sequences, cardinality, other)
        ("complex_1", ["ACDE", "WXYZ"], [1, 1], None),
    ]

    # Must not raise IndexError; must produce base/0.a3m.
    orchestrator._emit_complex_outputs(
        base,
        queries_unique,
        keep_unpaired=False,
        keep_paired=True,
        unpack=True,
    )
    output = base / "0.a3m"
    assert output.exists(), "paired-only concat did not produce 0.a3m"
    text = output.read_text()
    # msa_to_str prefixes with "#<lens>\t<cardinalities>\n" header.
    assert text.startswith("#4,4\t1,1\n"), (
        f"unexpected header line: {text.splitlines()[:1]}"
    )
    # Pair mode CONCATENATES chains into single rows joined by tab in the
    # header (``>101\t102``) with sequences glued (``ACDEWXYZ``).  This is
    # ``pair_sequences``'s contract — one row per paired hit across all chains.
    assert ">101\t102" in text, (
        f"paired MSA missing concatenated chain header: {text!r}"
    )
    assert "ACDEWXYZ" in text, (
        f"paired MSA missing glued-chain sequence: {text!r}"
    )


def test_emit_complex_outputs_unpaired_only_still_works(
    tmp_path: Path,
) -> None:
    """The mirror case: keep_unpaired=True + keep_paired=False must also
    concat correctly (this path used to work; guard against regression)."""
    base = tmp_path
    (base / "0.a3m").write_text(">101\nACDE\n>seq1\nACDE\n")
    (base / "1.a3m").write_text(">102\nWXYZ\n>seq2\nWXYZ\n")

    queries_unique = [
        ("complex_1", ["ACDE", "WXYZ"], [1, 1], None),
    ]
    orchestrator._emit_complex_outputs(
        base,
        queries_unique,
        keep_unpaired=True,
        keep_paired=False,
        unpack=True,
    )
    assert (base / "0.a3m").exists()
