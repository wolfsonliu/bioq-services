"""HTTP-form mode/query parsing + orchestrator argv builder.

Translates the ColabFold-protocol fields (`q`, `mode`) that arrive on
`POST /ticket/msa` or `POST /ticket/pair` into:
  - ParsedSequence list (after FASTA parse + per-residue validation)
  - ModeConfig (after mode-string -> flag mapping)
  - argv list ready for SubprocessRunner to invoke `python -m server.orchestrator`

This is the only module that needs to know both the ColabFold wire
protocol AND the local orchestrator's CLI surface — keeps routes thin
and the orchestrator decoupled from HTTP concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:  # pragma: no cover — import only used for type hints
    from server.settings import MMseqs2Settings


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModeConfig:
    """Orchestrator CLI flag values derived from a ColabFold `mode` string."""

    use_env: int                                  # 0 or 1
    filter: int                                   # 0 or 1
    pair_mode: Optional[Literal["paired"]]        # None -> monomer/unpaired path
    pairing_strategy: Optional[int]               # None | 0 (greedy) | 1 (complete)


@dataclass
class ParsedSequence:
    """One FASTA record extracted from the form-data `q` field."""

    header: str    # the original header line (without the leading >)
    sequence: str  # uppercase, contiguous, no whitespace


# ---------------------------------------------------------------------------
# Mode parsing
# ---------------------------------------------------------------------------

# Static table — see engineering/decisions/2026-06-23-mmseqs2-server-design.md
# "mode 映射 + orchestrator CLI" section for derivation.
_MODE_TABLE: dict[str, ModeConfig] = {
    "env":              ModeConfig(use_env=1, filter=1, pair_mode=None,     pairing_strategy=None),
    "all":              ModeConfig(use_env=0, filter=1, pair_mode=None,     pairing_strategy=None),
    "env-nofilter":     ModeConfig(use_env=1, filter=0, pair_mode=None,     pairing_strategy=None),
    "nofilter":         ModeConfig(use_env=0, filter=0, pair_mode=None,     pairing_strategy=None),
    "pairgreedy":       ModeConfig(use_env=0, filter=0, pair_mode="paired", pairing_strategy=0),
    "paircomplete":     ModeConfig(use_env=0, filter=0, pair_mode="paired", pairing_strategy=1),
    "pairgreedy-env":   ModeConfig(use_env=1, filter=0, pair_mode="paired", pairing_strategy=0),
    "paircomplete-env": ModeConfig(use_env=1, filter=0, pair_mode="paired", pairing_strategy=1),
}


def parse_mode_flags(mode: str) -> ModeConfig:
    """Map a ColabFold-protocol `mode` form-data string to orchestrator flags.

    Case-sensitive: "ENV" is rejected, only "env" is valid. The route layer
    catches ValueError and returns ColabFold's `{"status": "ERROR"}`.
    """
    try:
        return _MODE_TABLE[mode]
    except KeyError as e:
        raise ValueError(f"unsupported mode: {mode!r}") from e


# ---------------------------------------------------------------------------
# Orchestrator argv builder
# ---------------------------------------------------------------------------


def colabfold_search_argv(
    *,
    query_path: Path,
    output_dir: Path,
    mode_config: ModeConfig,
    settings: "MMseqs2Settings",
) -> list[str]:
    """Build the argv to invoke the vendored orchestrator as a subprocess.

    Flag names + order match `orchestrator.main()`'s argparse setup. Keeping
    the order stable makes subprocess command lines in FC logs easy to read.
    """
    argv: list[str] = [
        "python", "-m", "server.orchestrator",
        "--query", str(query_path),
        "--db-dir", str(settings.db_dir),
        "--output-dir", str(output_dir),
        "--mmseqs", settings.mmseqs_binary,
        "--db1", settings.default_db,
    ]

    # --use-env always emitted (orchestrator default is 0 but explicit is
    # easier to audit from log lines).
    argv += ["--use-env", str(mode_config.use_env)]
    if mode_config.use_env == 1:
        if not settings.env_db:
            raise ValueError("--use-env 1 requested but settings.env_db is unset")
        argv += ["--db3", settings.env_db]

    argv += ["--filter", str(mode_config.filter)]

    if mode_config.pair_mode == "paired":
        argv += ["--pair-mode", "paired"]
        # pairing_strategy is guaranteed non-None for the paired modes by
        # the _MODE_TABLE, but the orchestrator's argparse also enforces it.
        argv += ["--pairing-strategy", str(mode_config.pairing_strategy)]
    else:
        argv += ["--pair-mode", "unpaired"]

    argv += ["--gpu", "1" if settings.gpu_enabled else "0"]
    argv += ["--threads", str(settings.threads)]

    return argv


# ---------------------------------------------------------------------------
# FASTA query parsing
# ---------------------------------------------------------------------------

# 20 standard amino acids + X (unknown) + * (stop codon, accepted by ColabFold).
_VALID_AA_ALPHABET: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWYX*")

# Boltz single-chain limit. Override per-call in tests for malformed-input cases.
_DEFAULT_MAX_LEN = 1023


def parse_query_fasta(q: str, *, max_len: int = _DEFAULT_MAX_LEN) -> list[ParsedSequence]:
    """Parse the FASTA-like body sent in the form-data `q` field.

    ColabFold protocol: `q` is like `">1\\nMKQHKAM...\\n>2\\nLLLL...\\n"`.
    Multiple `>` blocks are allowed for multimers.

    Validation:
      - Empty `q` -> ValueError.
      - Any residue outside the 20 standard AAs + `X` + `*` -> ValueError.
      - Per-sequence length > `max_len` -> ValueError.
      - Sequences are uppercased before validation.
      - Headers without any sequence body -> ValueError (malformed input).

    Returns records in input order; dedup is the orchestrator's responsibility.
    Handles both `\\n` and `\\r\\n` line endings (Windows clients).
    """
    if not q or not q.strip():
        raise ValueError("empty query")

    records: list[ParsedSequence] = []
    current_header: Optional[str] = None
    current_seq_parts: list[str] = []

    def _flush() -> None:
        if current_header is None:
            return
        # Strip whitespace inside the joined seq (line breaks etc.) and upper.
        seq = "".join(current_seq_parts).strip().upper()
        # Strip any remaining whitespace within (defensive — split below
        # already removes line breaks but tabs/spaces inside a line need this).
        seq = "".join(ch for ch in seq if not ch.isspace())
        if not seq:
            raise ValueError(f"sequence {current_header!r} has empty body")
        if len(seq) > max_len:
            raise ValueError(
                f"sequence {current_header!r} length {len(seq)} > {max_len}"
            )
        for ch in seq:
            if ch not in _VALID_AA_ALPHABET:
                raise ValueError(
                    f"invalid amino acid character {ch!r} in sequence {current_header!r}"
                )
        records.append(ParsedSequence(header=current_header, sequence=seq))

    # splitlines() handles both \n and \r\n transparently.
    for line in q.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            _flush()
            current_header = line[1:].strip()
            current_seq_parts = []
        else:
            if current_header is None:
                # Sequence data before any header — malformed.
                raise ValueError("sequence data found before any FASTA header")
            current_seq_parts.append(line)

    _flush()

    if not records:
        raise ValueError("empty query")

    return records
