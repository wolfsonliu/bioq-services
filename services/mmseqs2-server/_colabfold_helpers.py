"""Vendored ColabFold input helpers.

Source: opensource/ColabFold/colabfold/input.py
Upstream commit: 26c0d46e12a98603a190231d643f6cafa49566b4
Upstream date:   2026-03-07
Upstream license: MIT (Copyright (c) 2021 Sergey Ovchinnikov)
                  see opensource/ColabFold/LICENSE

Scope of this vendor (vs full upstream module):
  - Only what `orchestrator.py` needs from `colabfold.input`:
      * safe_filename     — sanitise FASTA header into a filesystem name
      * parse_fasta       — minimal FASTA splitter (sequences + descriptions)
      * get_queries       — read a single FASTA file into the standard
                            ``(jobname, sequence(s), a3m_lines, other_mols)``
                            tuple list + ``is_complex`` flag.  We only need
                            the FASTA branch — CSV/TSV/A3M/PDB/CIF code paths
                            and the colabfold.utils.MolType / pandas / alphafold
                            imports are dropped (out-of-scope for the MSA
                            service).
      * msa_to_str        — combine paired + unpaired MSA into a single string
                            for complex outputs (used by orchestrator after
                            mmseqs unpacks per-chain a3m files).
      * pair_sequences / pad_sequences / pair_msa — internal helpers reachable
                            from msa_to_str.

Local changes (vs upstream):
  - Drop dependency on `colabfold.utils.MolType` and the `classify_molecules`
    helper.  We treat any ``A:B`` style multimer FASTA as a list of plain
    protein sequences and surface non-protein components as the raw string
    (we do not invoke AF3 JSON generation).  ``other_molecules`` is therefore
    always None in our get_queries output.
  - Drop CSV/TSV/A3M/PDB/CIF input branches in ``get_queries`` — the service
    only accepts a single ``.fasta`` / ``.fa`` / ``.faa`` query file produced
    by ``tools.py``.
  - Drop the directory-input branch in ``get_queries`` — orchestrator hands
    a single file path.
  - Drop ``sort_queries_by`` / random shuffling — one query per invocation.

This module has no ``__main__`` block — it is a helper library imported by
``orchestrator.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def safe_filename(file: str) -> str:
    """Sanitise an arbitrary string into a filesystem-safe filename.

    Upstream reference: colabfold/input.py:8-9

    Replaces every character that is not alphanumeric or one of ``_ . -``
    with ``_``.  Used to turn FASTA headers (which may contain spaces,
    pipes, slashes, etc.) into output file stems.
    """
    return "".join(
        c if c.isalnum() or c in ["_", ".", "-"] else "_" for c in file
    )


def parse_fasta(fasta_string: str) -> Tuple[List[str], List[str]]:
    """Parse a FASTA blob into ``(sequences, descriptions)``.

    Upstream reference: colabfold/input.py:88-116

    Lines starting with ``#`` are treated as comments and skipped.
    Blank lines between records are tolerated.
    """
    sequences: List[str] = []
    descriptions: List[str] = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # drop leading '>'
            sequences.append("")
            continue
        elif not line:
            continue
        sequences[index] += line
    return sequences, descriptions


def pair_sequences(
    a3m_lines: List[str],
    query_sequences: List[str],
    query_cardinality: List[int],
) -> str:
    """Concatenate per-chain a3m blocks line-by-line for paired MSA output.

    Upstream reference: colabfold/input.py:11-24

    Each input a3m has the same number of rows (because pairing already
    matched rows by species); we glue chain n's row i onto the right of
    chain 0's row i, separated by a tab where the second '>' would be.
    """
    a3m_line_paired = [""] * len(a3m_lines[0].splitlines())
    for n, _seq in enumerate(query_sequences):
        lines = a3m_lines[n].splitlines()
        for i, line in enumerate(lines):
            if line.startswith(">"):
                if n != 0:
                    line = line.replace(">", "\t", 1)
                a3m_line_paired[i] = a3m_line_paired[i] + line
            else:
                a3m_line_paired[i] = (
                    a3m_line_paired[i] + line * query_cardinality[n]
                )
    return "\n".join(a3m_line_paired)


def pad_sequences(
    a3m_lines: List[str],
    query_sequences: List[str],
    query_cardinality: List[int],
) -> str:
    """Pad each chain's unpaired a3m with gaps for the other chains.

    Upstream reference: colabfold/input.py:26-49

    Produces a single MSA where each row corresponds to one chain's hit
    plus gap fillers for every other chain in the complex.
    """
    _blank_seq = [
        ("-" * len(seq))
        for n, seq in enumerate(query_sequences)
        for _ in range(query_cardinality[n])
    ]
    a3m_lines_combined: List[str] = []
    pos = 0
    for n, _seq in enumerate(query_sequences):
        for _j in range(0, query_cardinality[n]):
            lines = a3m_lines[n].split("\n")
            for a3m_line in lines:
                if len(a3m_line) == 0:
                    continue
                if a3m_line.startswith(">"):
                    a3m_lines_combined.append(a3m_line)
                else:
                    a3m_lines_combined.append(
                        "".join(_blank_seq[:pos] + [a3m_line] + _blank_seq[pos + 1:])
                    )
            pos += 1
    return "\n".join(a3m_lines_combined)


def pair_msa(
    query_seqs_unique: List[str],
    query_seqs_cardinality: List[int],
    paired_msa: Optional[List[str]],
    unpaired_msa: Optional[List[str]],
) -> str:
    """Combine paired + unpaired MSAs into a single a3m string.

    Upstream reference: colabfold/input.py:51-73

    Three cases: only unpaired (pad), both (paired then padded unpaired
    appended), only paired (pair sequences).
    """
    if paired_msa is None and unpaired_msa is not None:
        return pad_sequences(unpaired_msa, query_seqs_unique, query_seqs_cardinality)
    if paired_msa is not None and unpaired_msa is not None:
        return (
            pair_sequences(paired_msa, query_seqs_unique, query_seqs_cardinality)
            + "\n"
            + pad_sequences(unpaired_msa, query_seqs_unique, query_seqs_cardinality)
        )
    if paired_msa is not None and unpaired_msa is None:
        return pair_sequences(paired_msa, query_seqs_unique, query_seqs_cardinality)
    raise ValueError("Invalid pairing: paired and unpaired MSAs are both None")


def msa_to_str(
    unpaired_msa: List[str],
    paired_msa: Optional[List[str]],
    query_seqs_unique: List[str],
    query_seqs_cardinality: List[int],
) -> str:
    """Compose the final complex-mode a3m string written to ``<job>.a3m``.

    Upstream reference: colabfold/input.py:75-86

    The leading ``#<lens>\\t<cardinalities>`` comment line carries chain
    length / copy information for downstream folding tools (ColabFold,
    Boltz, etc.).  Cardinality is reset to 1 internally before pairing
    because each unique sequence is represented once in the alignment.
    """
    msa = "#" + ",".join(map(str, map(len, query_seqs_unique))) + "\t"
    msa += ",".join(map(str, query_seqs_cardinality)) + "\n"
    query_seqs_cardinality = [1 for _ in query_seqs_cardinality]
    msa += pair_msa(query_seqs_unique, query_seqs_cardinality, paired_msa, unpaired_msa)
    return msa


# Type alias for the returned tuple — kept as plain typing rather than a
# TypedDict / dataclass to mirror upstream signature exactly.
QueryTuple = Tuple[str, Union[str, List[str]], Optional[List[str]], Optional[list]]


def get_queries(
    input_path: Union[str, Path],
) -> Tuple[List[QueryTuple], bool]:
    """Read a single FASTA file into the canonical ColabFold query list.

    Upstream reference: colabfold/input.py:267-405 (FASTA branch only)

    Output tuple per query:
        (jobname, sequence_or_sequences, a3m_lines_or_None, other_mols_or_None)

    For a monomer FASTA record the sequence is a plain string; for a
    multimer (``:``-separated) record it becomes a list of strings.  The
    ``is_complex`` flag is True iff any record contains ``:``.

    Diverges from upstream:
      - Only ``.fasta``/``.fa``/``.faa`` is accepted (no CSV/TSV/A3M/PDB/CIF
        — those would pull pandas / alphafold imports we deliberately
        avoid; tools.py is responsible for ensuring its input is FASTA).
      - Directory input is not accepted (one query file per orchestrator
        invocation).
      - ``other_molecules`` is always None: we don't classify non-protein
        sub-sequences (out of scope for the MSA pipeline; the downstream
        folding tool consumes them via its own input parser).
      - No ``sort_queries_by`` parameter — single query, no sort needed.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise OSError(f"{input_path} could not be found")
    if not input_path.is_file():
        raise ValueError(
            f"Expected a single FASTA file, got directory: {input_path}"
        )
    if input_path.suffix.lower() not in (".fasta", ".fa", ".faa"):
        raise ValueError(
            f"Unsupported query suffix {input_path.suffix!r} "
            "(only .fasta/.fa/.faa supported by mmseqs2-server)"
        )

    sequences, headers = parse_fasta(input_path.read_text())
    if len(sequences) == 0:
        raise ValueError(f"{input_path} is empty")

    queries: List[QueryTuple] = []
    for sequence, header in zip(sequences, headers):
        sequence = sequence.upper()
        if sequence.count(":") == 0:
            queries.append((header, sequence, None, None))
        else:
            # Multimer: split by ':' into per-chain sequences.  We do not
            # honour the ``MOLTYPE|SEQ|COPIES`` extended syntax — any chain
            # containing ``|`` is passed through as-is and will be flagged
            # by downstream validation.
            protein_queries = sequence.split(":")
            queries.append((header, protein_queries, None, None))

    is_complex = any(isinstance(q[1], list) for q in queries)
    return queries, is_complex
