# Vendored upstream — ColabFold MSA orchestration

This service vendors two functions and a few helpers from the ColabFold
project rather than depending on the upstream Python package.  Rationale,
sync procedure and locked parameters are recorded below.

## Upstream

| Field            | Value |
| ---------------- | ----- |
| Project          | [ColabFold](https://github.com/sokrypton/ColabFold) |
| Vendored commit  | `26c0d46e12a98603a190231d643f6cafa49566b4` |
| Commit date      | 2026-03-07 |
| License          | MIT (Copyright (c) 2021 Sergey Ovchinnikov) |
| Vendored files   | `opensource/ColabFold/colabfold/mmseqs/search.py`, `opensource/ColabFold/colabfold/input.py` (subset) |

The license text is preserved at `opensource/ColabFold/LICENSE`; every
vendored file in this service carries a per-file provenance docstring
that points back to this document.

## What lives where

| Vendored to (in this service)   | Upstream origin                                  |
| ------------------------------- | ------------------------------------------------ |
| `orchestrator.py`               | `colabfold/mmseqs/search.py` (subset, see below) |
| `_colabfold_helpers.py`         | `colabfold/input.py` (subset, see below)         |

## What we kept

From `colabfold/mmseqs/search.py`:

- `MODULE_OUTPUT_POS` constant — drives the resume-by-output-exists logic.
- `run_mmseqs(mmseqs, params)` — subprocess wrapper.
- `mmseqs_search_monomer(...)` — UniRef30 (+ optional env DB) → final.a3m.
- `mmseqs_search_pair(...)`    — multimer pairing variant.

From `colabfold/input.py`:

- `safe_filename`
- `parse_fasta`
- `pair_sequences`, `pad_sequences`, `pair_msa`
- `msa_to_str`
- `get_queries` (FASTA-only branch)

## What we dropped (vs upstream)

| Dropped                                       | Why                                                                                                |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `run_mmseqs2(...)` (remote API client)        | We **are** the server; never call `api.colabfold.com`.                                             |
| `--af3-json` / `--af3-msa-as-path` + `AF3Utils` | Out of scope — this service emits a3m only.                                                       |
| `--use-templates` / `--db2` / template path   | Phase 2 will add a separate templates service.                                                     |
| `--use-env-pairing` / `--db4` / SPIRE         | Not exposed in our CLI contract; can be re-added later by surfacing `mmseqs_search_pair(pair_env=True)`. |
| Multi-query batching in upstream `main()`     | One query per orchestrator invocation (HTTP / CLI front-end issues one job per request).           |
| CSV / TSV / A3M / PDB / CIF query parsers     | Avoids pulling `pandas` + `alphafold` dependencies; `tools.py` is responsible for FASTA conversion. |
| `colabfold.utils.MolType` / `classify_molecules` | Tied to AF3 JSON output, which we don't produce.                                                |
| `--sort-queries-by` (random shuffle)          | Single query — no sort needed.                                                                     |

## Local additions

- New `main()` matching the CLI contract in `tools.py:colabfold_search_argv()`
  (long-only flags, validation: `--pair-mode paired` ⇒ `--pairing-strategy`,
  `--use-env 1` ⇒ `--db3`).
- Detailed inline comments on every `run_mmseqs(...)` call: each one carries
  a **Step N/M: title** block with *Why*, *Input*, *Output*, *Why these
  params*, and *Upstream reference* lines.

## Locked MMseqs2 parameters (Task 0.1 verification)

The MMseqs2 binary we ship in the Docker image is GPU-enabled; the parameter
combinations below must match upstream `mmseqs <subcmd> --help` output and
must not drift on a re-vendor.  See
`engineering/decisions/2026-06-23-mmseqs2-server-design.md`#MMseq2 for the
full `--help` listings used to verify these.

| Pipeline stage                  | Locked params                                              |
| ------------------------------- | ---------------------------------------------------------- |
| GPU search                      | `--gpu 1 --prefilter-mode 1`                               |
| Pair greedy (per species)       | `pairaln --pairing-mode 0`                                 |
| Pair complete (all chains)      | `pairaln --pairing-mode 1`                                 |
| A3M output (monomer / env)      | `result2msa --msa-format-mode 6`                           |
| A3M output (paired)             | `result2msa --msa-format-mode 5`                           |

If any of these change in upstream, update the table together with the
SHA bump and re-run the offline tests.

## Sync procedure — bumping the upstream version

1. Update the local checkout:

   ```bash
   cd opensource/ColabFold && git pull
   git log -1 --format='%H %ci' -- colabfold/mmseqs/search.py colabfold/input.py
   ```

2. 3-way diff the two upstream files against our vendored copies.  Pay
   attention to:

   - new flags inside `search_param` / `filter_param` / `expand_param` blocks,
   - changes to `MODULE_OUTPUT_POS`,
   - any new `run_mmseqs(...)` step inserted into either function,
   - any rename of an intermediate DB (e.g. `res_exp_realign_pair`).

3. Cherry-pick the diff into our `orchestrator.py` (and
   `_colabfold_helpers.py` if `colabfold/input.py` moved).  When in doubt,
   keep the upstream algorithmic structure verbatim so future diffs stay
   trivial.

4. Update the provenance docstring header in each vendored file (SHA + date)
   and the "Upstream" table at the top of this document.

5. Re-run the offline test suite:

   ```bash
   uv run python -m pytest services/mmseqs2-server/tests/ -v
   ```

6. If any `run_mmseqs(...)` call changed, refresh its 4-field comment block.
   The whole point of the per-step comments is that nobody has to re-derive
   what a step does from upstream the next time we sync.

7. Commit with `chore(mmseqs2-server): bump vendored ColabFold to <SHA>`.

## Why vendor (and not pip install)

- ColabFold's release cadence is research-grade; pinning to a SHA is more
  stable than pinning to a PyPI version.
- The full package drags `biopython`, `numpy`, `alphafold` (for PDB/CIF
  input parsing) and the AF3 output JSON code path — none of which are
  needed for an a3m-only server.
- Vendoring keeps our service Dockerfile decoupled from the `opensource/`
  tree at build time (consistent with the `rfdiffusion2-server` precedent
  — see `engineering/decisions/2026-05-19-rfdiffusion2-server-vendor.md`).
