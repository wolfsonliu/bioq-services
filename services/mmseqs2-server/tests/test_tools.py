"""Offline unit tests for server.tools — mode parsing, argv builder, FASTA parser.

No subprocess / network / filesystem use. Pure parse + map logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from server.settings import MMseqs2Settings
from server.tools import (
    ModeConfig,
    ParsedSequence,
    colabfold_search_argv,
    parse_mode_flags,
    parse_query_fasta,
)


# ---------------------------------------------------------------------------
# parse_mode_flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("env",              ModeConfig(use_env=1, filter=1, pair_mode=None,     pairing_strategy=None)),
        ("all",              ModeConfig(use_env=0, filter=1, pair_mode=None,     pairing_strategy=None)),
        ("env-nofilter",     ModeConfig(use_env=1, filter=0, pair_mode=None,     pairing_strategy=None)),
        ("nofilter",         ModeConfig(use_env=0, filter=0, pair_mode=None,     pairing_strategy=None)),
        ("pairgreedy",       ModeConfig(use_env=0, filter=0, pair_mode="paired", pairing_strategy=0)),
        ("paircomplete",     ModeConfig(use_env=0, filter=0, pair_mode="paired", pairing_strategy=1)),
        ("pairgreedy-env",   ModeConfig(use_env=1, filter=0, pair_mode="paired", pairing_strategy=0)),
        ("paircomplete-env", ModeConfig(use_env=1, filter=0, pair_mode="paired", pairing_strategy=1)),
    ],
)
def test_parse_mode_flags_valid_modes(mode: str, expected: ModeConfig) -> None:
    assert parse_mode_flags(mode) == expected


def test_parse_mode_flags_empty_string_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported mode"):
        parse_mode_flags("")


def test_parse_mode_flags_unknown_string_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported mode"):
        parse_mode_flags("random-string")


def test_parse_mode_flags_case_sensitive() -> None:
    # "ENV" must be rejected — uppercase is not silently normalised.
    with pytest.raises(ValueError, match="unsupported mode"):
        parse_mode_flags("ENV")


# ---------------------------------------------------------------------------
# colabfold_search_argv
# ---------------------------------------------------------------------------


def _settings(**overrides: object) -> MMseqs2Settings:
    """Build settings without depending on environment variables."""
    base: dict[str, object] = {
        "jobs_base_dir": Path("/tmp/mmseqs2_jobs"),
        "mmseqs_binary": "/opt/mmseqs-gpu/bin/mmseqs",
        "db_dir": Path("/data/models/mmseqs2"),
        "default_db": "uniref30_subset_4090_gpu",
        "env_db": "colabfold_envdb_gpu",
        "gpu_enabled": True,
        "threads": 4,
    }
    base.update(overrides)
    return MMseqs2Settings(**base)  # type: ignore[arg-type]


def test_argv_starts_with_python_module_invocation() -> None:
    argv = colabfold_search_argv(
        query_path=Path("/tmp/query.fasta"),
        output_dir=Path("/tmp/out"),
        mode_config=parse_mode_flags("all"),
        settings=_settings(),
    )
    assert argv[:3] == ["python", "-m", "server.orchestrator"]


def test_argv_includes_query_and_output_paths_as_str() -> None:
    qp = Path("/tmp/query.fasta")
    od = Path("/tmp/out")
    argv = colabfold_search_argv(
        query_path=qp,
        output_dir=od,
        mode_config=parse_mode_flags("all"),
        settings=_settings(),
    )
    assert str(qp) in argv
    assert str(od) in argv


def test_argv_env_mode_appends_db3() -> None:
    argv = colabfold_search_argv(
        query_path=Path("/tmp/query.fasta"),
        output_dir=Path("/tmp/out"),
        mode_config=parse_mode_flags("env"),
        settings=_settings(env_db="colabfold_envdb_gpu"),
    )
    assert "--use-env" in argv
    use_env_value = argv[argv.index("--use-env") + 1]
    assert use_env_value == "1"
    assert "--db3" in argv
    assert argv[argv.index("--db3") + 1] == "colabfold_envdb_gpu"


def test_argv_non_env_mode_omits_db3() -> None:
    argv = colabfold_search_argv(
        query_path=Path("/tmp/query.fasta"),
        output_dir=Path("/tmp/out"),
        mode_config=parse_mode_flags("all"),
        settings=_settings(),
    )
    assert "--db3" not in argv
    assert argv[argv.index("--use-env") + 1] == "0"


def test_argv_env_mode_raises_when_env_db_unset() -> None:
    with pytest.raises(ValueError, match="env_db is unset"):
        colabfold_search_argv(
            query_path=Path("/tmp/query.fasta"),
            output_dir=Path("/tmp/out"),
            mode_config=parse_mode_flags("env"),
            settings=_settings(env_db=None),
        )


def test_argv_env_mode_raises_when_env_db_empty_string() -> None:
    with pytest.raises(ValueError, match="env_db is unset"):
        colabfold_search_argv(
            query_path=Path("/tmp/query.fasta"),
            output_dir=Path("/tmp/out"),
            mode_config=parse_mode_flags("env"),
            settings=_settings(env_db=""),
        )


def test_argv_paired_mode_appends_pairing_strategy() -> None:
    argv = colabfold_search_argv(
        query_path=Path("/tmp/query.fasta"),
        output_dir=Path("/tmp/out"),
        mode_config=parse_mode_flags("paircomplete"),
        settings=_settings(),
    )
    assert "--pair-mode" in argv
    assert argv[argv.index("--pair-mode") + 1] == "paired"
    assert "--pairing-strategy" in argv
    assert argv[argv.index("--pairing-strategy") + 1] == "1"


def test_argv_paired_greedy_strategy_value() -> None:
    argv = colabfold_search_argv(
        query_path=Path("/tmp/query.fasta"),
        output_dir=Path("/tmp/out"),
        mode_config=parse_mode_flags("pairgreedy"),
        settings=_settings(),
    )
    assert argv[argv.index("--pairing-strategy") + 1] == "0"


def test_argv_unpaired_mode_omits_pairing_strategy() -> None:
    argv = colabfold_search_argv(
        query_path=Path("/tmp/query.fasta"),
        output_dir=Path("/tmp/out"),
        mode_config=parse_mode_flags("all"),
        settings=_settings(),
    )
    assert "--pairing-strategy" not in argv
    assert argv[argv.index("--pair-mode") + 1] == "unpaired"


def test_argv_gpu_enabled_emits_one() -> None:
    argv = colabfold_search_argv(
        query_path=Path("/tmp/query.fasta"),
        output_dir=Path("/tmp/out"),
        mode_config=parse_mode_flags("all"),
        settings=_settings(gpu_enabled=True),
    )
    assert argv[argv.index("--gpu") + 1] == "1"


def test_argv_gpu_disabled_emits_zero() -> None:
    argv = colabfold_search_argv(
        query_path=Path("/tmp/query.fasta"),
        output_dir=Path("/tmp/out"),
        mode_config=parse_mode_flags("all"),
        settings=_settings(gpu_enabled=False),
    )
    assert argv[argv.index("--gpu") + 1] == "0"


def test_argv_propagates_threads_and_db_paths() -> None:
    argv = colabfold_search_argv(
        query_path=Path("/tmp/query.fasta"),
        output_dir=Path("/tmp/out"),
        mode_config=parse_mode_flags("all"),
        settings=_settings(
            threads=8,
            db_dir=Path("/mnt/nas/mmseqs2"),
            default_db="uniref30_custom",
            mmseqs_binary="/usr/local/bin/mmseqs",
        ),
    )
    assert argv[argv.index("--threads") + 1] == "8"
    assert argv[argv.index("--db-dir") + 1] == "/mnt/nas/mmseqs2"
    assert argv[argv.index("--db1") + 1] == "uniref30_custom"
    assert argv[argv.index("--mmseqs") + 1] == "/usr/local/bin/mmseqs"


# ---------------------------------------------------------------------------
# parse_query_fasta
# ---------------------------------------------------------------------------


def test_parse_single_sequence() -> None:
    out = parse_query_fasta(">1\nMKQH\n")
    assert out == [ParsedSequence(header="1", sequence="MKQH")]


def test_parse_multi_sequence_preserves_order() -> None:
    out = parse_query_fasta(">A\nMKQH\n>B\nLLLL\n")
    assert out == [
        ParsedSequence(header="A", sequence="MKQH"),
        ParsedSequence(header="B", sequence="LLLL"),
    ]


def test_parse_strips_whitespace_within_sequence_block() -> None:
    out = parse_query_fasta(">1\nMK QH\n  AA\t\n")
    assert out == [ParsedSequence(header="1", sequence="MKQHAA")]


def test_parse_uppercases_lowercase_input() -> None:
    out = parse_query_fasta(">1\nmkqh\n")
    assert out == [ParsedSequence(header="1", sequence="MKQH")]


def test_parse_wraps_multi_line_sequence_into_one() -> None:
    out = parse_query_fasta(">A\nMKQH\nLLLL\n")
    assert out == [ParsedSequence(header="A", sequence="MKQHLLLL")]


def test_parse_handles_crlf_line_endings() -> None:
    out = parse_query_fasta(">A\r\nMKQH\r\n>B\r\nLLLL\r\n")
    assert out == [
        ParsedSequence(header="A", sequence="MKQH"),
        ParsedSequence(header="B", sequence="LLLL"),
    ]


def test_parse_accepts_unknown_and_stop_codons() -> None:
    out = parse_query_fasta(">1\nMKXH*\n")
    assert out == [ParsedSequence(header="1", sequence="MKXH*")]


@pytest.mark.parametrize("bad_char", ["Z", "B", "0", "@"])
def test_parse_rejects_invalid_amino_acid(bad_char: str) -> None:
    with pytest.raises(ValueError, match="invalid amino acid character"):
        parse_query_fasta(f">1\nMK{bad_char}H\n")


def test_parse_empty_query_rejected() -> None:
    with pytest.raises(ValueError, match="empty query"):
        parse_query_fasta("")


def test_parse_whitespace_only_query_rejected() -> None:
    with pytest.raises(ValueError, match="empty query"):
        parse_query_fasta("   \n\n   ")


def test_parse_header_without_body_rejected() -> None:
    with pytest.raises(ValueError, match="empty body"):
        parse_query_fasta(">1\n")


def test_parse_header_without_body_between_records_rejected() -> None:
    # First record is valid, second header has no body.
    with pytest.raises(ValueError, match="empty body"):
        parse_query_fasta(">A\nMKQH\n>B\n")


def test_parse_sequence_before_header_rejected() -> None:
    with pytest.raises(ValueError, match="before any FASTA header"):
        parse_query_fasta("MKQH\n>1\nLLLL\n")


def test_parse_enforces_max_len() -> None:
    # max_len=10, sequence has 11 residues -> reject.
    with pytest.raises(ValueError, match="length 11 > 10"):
        parse_query_fasta(">A\nMKQHMKQHMKQ\n", max_len=10)


def test_parse_at_max_len_boundary_accepted() -> None:
    out = parse_query_fasta(">A\nMKQHMKQHMK\n", max_len=10)
    assert out[0].sequence == "MKQHMKQHMK"
    assert len(out[0].sequence) == 10
