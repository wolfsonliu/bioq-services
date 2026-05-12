"""Dataset zip extraction with problem-JSON path normalization.

genie3's problem JSONs reference data files (target PDBs, MSAs, motif PDBs, etc.)
via paths that are *relative to the upstream repo's cwd*, e.g.
`data/design/binder_design/binderbench/targets/pdb/01_bhrf1.pdb`.

When clients upload a problem-set zip into a job-local dir, those paths no
longer resolve. We rewrite them to absolute paths anchored at the just-extracted
`dataset_root` so genie3 finds the files regardless of subprocess cwd.

The rewriter also chases `binder_framework`, which points at a *nested* motif-
config JSON whose own `motif_filepaths` list needs the same treatment.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Path-bearing keys we know about in problem JSONs.
_PATH_KEYS_SCALAR = (
    "target_pdb_filepath",
    "target_fasta_filepath",
    "target_msa_filepath",
    # Points at a nested motif-config JSON; that file gets a follow-up rewrite.
    "binder_framework",
)
_PATH_KEYS_LIST = (
    "target_pdb_filepath_by_chain",
    "target_fasta_filepath_by_chain",
    "target_msa_filepath_by_chain",
    "motif_filepaths",
)


def extract_dataset(zip_path: Path, dest_dir: Path) -> Path:
    """Extract a problem-set zip and rewrite paths inside `problems/*.json`.

    The zip may wrap everything in a single top-level dir (`binderbench/`) or
    have `problems/` + `targets/` at the root — both layouts work. Returns the
    resolved dataset root (the dir containing `problems/`).

    Raises `zipfile.BadZipFile` or `ValueError`; the caller maps these to HTTP 422.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    # The dataset root is the dir containing `problems/`. Multiple candidates can
    # appear if archives nest oddly; pick the shallowest.
    candidates = [p.parent for p in dest_dir.rglob("problems") if p.is_dir()]
    if not candidates:
        raise ValueError(
            f"Dataset zip does not contain a 'problems/' directory: {zip_path.name}"
        )
    dataset_root = sorted(candidates, key=lambda p: len(p.parts))[0].resolve()

    _rewrite_problem_paths(dataset_root / "problems", dataset_root)
    return dataset_root


def _rewrite_problem_paths(problems_dir: Path, dataset_root: Path) -> None:
    if not problems_dir.exists():
        return

    for json_path in problems_dir.glob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("skipping unparseable JSON %s: %s", json_path, e)
            continue

        changed = False
        for key in _PATH_KEYS_SCALAR:
            if key in data and isinstance(data[key], str):
                new = _resolve_path(data[key], dataset_root)
                if new != data[key]:
                    data[key] = new
                    changed = True
        for key in _PATH_KEYS_LIST:
            if key in data and isinstance(data[key], list):
                new_list = [
                    _resolve_path(v, dataset_root) if isinstance(v, str) else v
                    for v in data[key]
                ]
                if new_list != data[key]:
                    data[key] = new_list
                    changed = True
        if changed:
            json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        # binder_framework → follow into the nested motif-config JSON.
        bfw = data.get("binder_framework")
        if isinstance(bfw, str):
            _rewrite_motif_config_paths(Path(bfw), dataset_root)


def _resolve_path(value: str, dataset_root: Path) -> str:
    """Map a relative-ish path in a problem JSON to an absolute path under dataset_root."""
    p = Path(value)
    # Already absolute and exists → trust it.
    if p.is_absolute() and p.exists():
        return str(p)

    # Direct join.
    candidate = dataset_root / value
    if candidate.exists():
        return str(candidate.resolve())

    # Strip prefix up to and including the conventional sub-roots.
    for marker in ("targets/", "motifs/"):
        idx = value.find(marker)
        if idx != -1:
            tail = value[idx:]
            candidate = dataset_root / tail
            if candidate.exists():
                return str(candidate.resolve())

    # Last resort: basename match in well-known subdirs.
    basename = p.name
    for sub in ("targets/pdb", "targets/fasta", "targets/msa", "motifs"):
        candidate = dataset_root / sub / basename
        if candidate.exists():
            return str(candidate.resolve())

    logger.warning("could not resolve dataset path %r under %s", value, dataset_root)
    return value


def _rewrite_motif_config_paths(config_path: Path, dataset_root: Path) -> None:
    """Rewrite `motif_filepaths` inside a nested motif-config JSON."""
    if not config_path.exists():
        logger.warning("motif config not found, skipping path rewrite: %s", config_path)
        return
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("skipping unparseable motif config %s: %s", config_path, e)
        return
    motif_paths = data.get("motif_filepaths")
    if not isinstance(motif_paths, list):
        return

    def _resolve_motif(value: str) -> str:
        p = Path(value)
        if p.is_absolute() and p.exists():
            return str(p)
        candidate = dataset_root / value
        if candidate.exists():
            return str(candidate.resolve())
        idx = value.find("motifs/")
        if idx != -1:
            candidate = dataset_root / value[idx:]
            if candidate.exists():
                return str(candidate.resolve())
        candidate = dataset_root / "motifs" / p.name
        if candidate.exists():
            return str(candidate.resolve())
        logger.warning("could not resolve motif path %r under %s", value, dataset_root)
        return value

    new_list = [_resolve_motif(v) if isinstance(v, str) else v for v in motif_paths]
    if new_list != motif_paths:
        data["motif_filepaths"] = new_list
        config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
