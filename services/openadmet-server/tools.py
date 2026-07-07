"""Argv assembly + input preparation for openadmet-server.

Wraps the upstream ``openadmet`` CLI (`click`-based, three subcommands):

    openadmet predict --input-path ... --input-col ... --model-dir ... --output-csv ...
    openadmet compare --model-dirs ... --label-types ... --output-dir ...

We subprocess into the CLI rather than importing the Python API (see §6.1 of
the design doc): heavy imports (torch + PyG + chemprop + molfeat) cost 30 s
per import; subprocess isolation keeps uvicorn worker memory clean.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .models import CompareRequest, PredictRequest
from .settings import ModelInfo, OpenAdmetSettings


# ---------------------------------------------------------------------------
# SMILES + CSV preparation
# ---------------------------------------------------------------------------

_SMILES_SPLIT_RE = re.compile(r"[\s,]+")


def split_inline_smiles(input_smiles: str, *, max_n: int = 200) -> list[str]:
    """Parse ``input_smiles`` (comma / whitespace separated) into a SMILES list.

    Empty tokens are skipped. Raises ValueError on empty input or on > max_n.
    """
    tokens = [t.strip() for t in _SMILES_SPLIT_RE.split(input_smiles) if t.strip()]
    if not tokens:
        raise ValueError("input_smiles is empty after splitting")
    if len(tokens) > max_n:
        raise ValueError(
            f"input_smiles contains {len(tokens)} molecules (max {max_n}); "
            f"use `input_csv` upload for larger batches."
        )
    return tokens


def write_alias_csv(
    smiles: Iterable[str],
    dest: Path,
    aliases: Iterable[str],
) -> Path:
    """Write a CSV with all alias columns filled with the same SMILES values.

    Solves the "which input_col name should I use" problem — subsequent
    subprocess calls just point ``--input-col`` at whichever alias matches
    the current model's data.yaml. See §4.1 of design doc.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    smiles_list = list(smiles)
    alias_list = list(aliases)
    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(alias_list)
        for s in smiles_list:
            writer.writerow([s] * len(alias_list))
    return dest


def augment_csv_with_aliases(
    input_csv: Path,
    dest: Path,
    aliases: Iterable[str],
    detected_col: str,
) -> Path:
    """Copy ``input_csv`` to ``dest`` adding any missing alias columns.

    The SMILES column identified as ``detected_col`` is duplicated under each
    alias name so downstream subprocess calls with any ``--input-col`` from
    ``aliases`` will hit the same values.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(input_csv, "r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        original_cols = reader.fieldnames or []
        if detected_col not in original_cols:
            raise ValueError(
                f"detected_col '{detected_col}' not in CSV columns: {original_cols}"
            )
        alias_list = list(aliases)
        # Compute new field list preserving order + appending unseen aliases.
        new_cols = list(original_cols)
        for a in alias_list:
            if a not in new_cols:
                new_cols.append(a)

        with open(dest, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=new_cols)
            writer.writeheader()
            for row in reader:
                val = row[detected_col]
                for a in alias_list:
                    if a not in row or not row[a]:
                        row[a] = val
                writer.writerow(row)
    return dest


def sniff_smiles_column(csv_path: Path, candidates: Iterable[str]) -> str | None:
    """Return the first column in ``candidates`` present in the CSV header, or None."""
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
    for c in candidates:
        if c in header:
            return c
    return None


# ---------------------------------------------------------------------------
# input_col grouping for multi-model predict
# ---------------------------------------------------------------------------


def group_models_by_input_col(
    models: list[ModelInfo],
    override_col: str | None,
) -> dict[str, list[ModelInfo]]:
    """Return {input_col_name: [ModelInfo, ...]} groups.

    If ``override_col`` is set, all models are grouped under that single key
    (client explicitly overrode auto-derivation).
    """
    if override_col:
        return {override_col: list(models)}
    groups: dict[str, list[ModelInfo]] = defaultdict(list)
    for m in models:
        groups[m.input_col].append(m)
    return dict(groups)


# ---------------------------------------------------------------------------
# argv builders
# ---------------------------------------------------------------------------


def predict_argv(
    req: PredictRequest,
    *,
    input_path: Path,
    input_col: str,
    output_csv: Path,
    model_dirs: list[Path],
    settings: OpenAdmetSettings,
) -> list[str]:
    """Compose one ``openadmet predict`` invocation for a group of models sharing an input_col.

    Uses the setuptools-generated ``openadmet`` script (settings.cli_binary),
    NOT ``python -m openadmet.models.cli.cli`` — see the settings.py comment
    on why ``python -m`` silently no-ops for click groups without a
    ``__main__`` guard.
    """
    argv: list[str] = [
        settings.cli_binary, "predict",
        "--input-path", str(input_path),
        "--input-col", input_col,
        "--output-csv", str(output_csv),
        "--accelerator", req.accelerator,
    ]
    for mp in model_dirs:
        argv += ["--model-dir", str(mp)]
    for aq in req.aq_fxns:
        argv += ["--aq-fxn", aq]
    for b in req.beta:
        argv += ["--beta", str(b)]
    for y in req.best_y:
        argv += ["--best-y", str(y)]
    for x in req.xi:
        argv += ["--xi", str(x)]
    if req.debug:
        argv += ["--debug"]
    return argv


def compare_argv_mode_a(
    req: CompareRequest,
    *,
    output_dir: Path,
    model_dirs: list[Path],
    settings: OpenAdmetSettings,
) -> list[str]:
    """`openadmet compare` in Mode A (from model dirs + label_types)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    argv: list[str] = [
        settings.cli_binary, "compare",
        "--output-dir", str(output_dir),
    ]
    for mp in model_dirs:
        argv += ["--model-dirs", str(mp)]
    for lt in req.label_types:
        argv += ["--label-types", lt]
    if req.mt_id:
        argv += ["--mt-id", req.mt_id]
    if req.report:
        argv += ["--report", "true"]
    return argv


def compare_argv_mode_b(
    req: CompareRequest,
    *,
    output_dir: Path,
    stats_files: list[Path],
    settings: OpenAdmetSettings,
) -> list[str]:
    """`openadmet compare` in Mode B (from JSON stats)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    argv: list[str] = [
        settings.cli_binary, "compare",
        "--output-dir", str(output_dir),
    ]
    for sf in stats_files:
        argv += ["--model-stats-fns", str(sf)]
    for label in req.labels:
        argv += ["--labels", label]
    for tn in req.task_names:
        argv += ["--task-names", tn]
    if req.mt_id:
        argv += ["--mt-id", req.mt_id]
    if req.report:
        argv += ["--report", "true"]
    return argv


# ---------------------------------------------------------------------------
# Compound argv: predict with input_col grouping
# ---------------------------------------------------------------------------


def predict_composite_argv(
    req: PredictRequest,
    *,
    input_csv: Path,
    job_dir: Path,
    settings: OpenAdmetSettings,
    models: list[ModelInfo],
) -> list[list[str]]:
    """Build the list of argv calls for one predict job.

    Groups ``models`` by input_col; one subprocess per group. Each group
    writes to ``output/predictions_group_<i>.csv``; a merge step (done by the
    JobRunner via a bash wrapper) consolidates into ``output/predictions.csv``.

    Returns a list of argv lists — the caller composes them into a single
    shell pipeline via ``build_predict_shell``.
    """
    override = req.input_col
    groups = group_models_by_input_col(models, override)

    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    argvs: list[list[str]] = []
    for idx, (col, group_models) in enumerate(groups.items()):
        group_csv = output_dir / f"predictions_group_{idx}.csv"
        argvs.append(
            predict_argv(
                req,
                input_path=input_csv,
                input_col=col,
                output_csv=group_csv,
                model_dirs=[m.path for m in group_models],
                settings=settings,
            )
        )
    return argvs


def build_predict_shell(argvs: list[list[str]], *, output_dir: Path) -> list[str]:
    """Wrap the argv list into a single bash -c pipeline.

    Runs each ``openadmet predict`` invocation sequentially (single-GPU
    concurrency), then merges all ``predictions_group_*.csv`` into a final
    ``predictions.csv`` via a Python one-liner (pandas is a hard dep of the
    conda env).

    Two separate quoting concerns are involved:

    * **Path literals inside the Python code** — must be valid Python
      string literals.  Use ``repr()`` (yields ``'…'`` with proper escaping)
      NOT ``shlex.quote()`` which is for shell tokens and leaves plain
      paths unquoted (breaking Python parsing).
    * **The whole Python source when embedded in ``bash -c 'python -c …'``**
      — must survive one round of shell parsing.  Use ``shlex.quote()`` on
      the whole ``merge_py`` string.
    """
    import shlex

    parts: list[str] = []
    for argv in argvs:
        parts.append(shlex.join(argv))

    # Merge step — outer join by row index on the group CSVs. Simple
    # concatenation of PRED/STD columns works because all groups share the
    # same row ordering (upstream keeps input CSV row order in predictions).
    output_dir_pylit = repr(str(output_dir))
    predictions_pylit = repr(str(output_dir / "predictions.csv"))
    merge_py = f"""\
import glob, os, pandas as pd
paths = sorted(glob.glob({output_dir_pylit} + "/predictions_group_*.csv"))
if not paths:
    # Adopt fallback: some upstream builds write predictions.csv directly
    # in output_dir (single-model call, no per-group suffix).  Salvage it.
    fallback = sorted(glob.glob({output_dir_pylit} + "/*.csv"))
    if fallback:
        paths = fallback
    else:
        try:
            listing = sorted(os.listdir({output_dir_pylit}))
        except Exception as _e:
            listing = f"(cannot list: {{_e}})"
        raise SystemExit(
            f"no per-group prediction CSVs produced. output_dir contents: {{listing!r}}"
        )
dfs = [pd.read_csv(p) for p in paths]
merged = dfs[0]
# For subsequent groups, only take OADMET_* columns (predictions/std/aq),
# keeping index alignment to the first group's rows.
for df in dfs[1:]:
    extra = [c for c in df.columns if c.startswith("OADMET_") and c not in merged.columns]
    if extra:
        merged = pd.concat([merged, df[extra]], axis=1)
merged.to_csv({predictions_pylit}, index=False)
print(f"merged {{len(paths)}} group CSV(s) -> predictions.csv "
      f"({{len(merged)}} rows, {{len(merged.columns)}} cols)")
"""
    parts.append(f"python -c {shlex.quote(merge_py)}")
    # Use ; not && so downstream errors surface even if a subprocess printed
    # a warning to stderr but returned 0.  We still want the pipeline to
    # short-circuit if openadmet predict itself fails (rc != 0), so wrap
    # with `set -e` up front (`set -e` propagates to each command, including
    # merge; SystemExit yields rc != 0 so pipeline stops there too).
    script = "set -e\n" + "\n".join(parts)
    return ["bash", "-c", script]


# ---------------------------------------------------------------------------
# Request archival (job manifest)
# ---------------------------------------------------------------------------


def archive_request(job_dir: Path, name: str, payload: dict) -> Path:
    """Write ``payload`` to ``<job_dir>/input/<name>.json`` for reproducibility."""
    dest = job_dir / "input" / f"{name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, default=str))
    return dest


def csv_from_smiles_list(smiles: list[str]) -> str:
    """In-memory CSV string with a single 'smiles' column. Used only in tests."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["smiles"])
    for s in smiles:
        writer.writerow([s])
    return buf.getvalue()
