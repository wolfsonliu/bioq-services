"""Dataset zip extraction + problem-JSON path normalization."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from server.datasets import extract_dataset


def _make_zip(zip_path: Path, layout: dict[str, bytes | str]) -> Path:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, content in layout.items():
            data = content.encode() if isinstance(content, str) else content
            zf.writestr(name, data)
    return zip_path


def test_extract_dataset_with_nested_root(tmp_path: Path) -> None:
    # Layout: binderbench/problems/foo.json, binderbench/targets/pdb/foo.pdb
    problem = {"target_pdb_filepath": "data/binderbench/targets/pdb/foo.pdb"}
    _make_zip(
        tmp_path / "ds.zip",
        {
            "binderbench/problems/foo.json": json.dumps(problem),
            "binderbench/targets/pdb/foo.pdb": "ATOM",
        },
    )

    dataset_root = extract_dataset(tmp_path / "ds.zip", tmp_path / "out")
    assert dataset_root.name == "binderbench"

    # The problem JSON's filepath was rewritten to an absolute path under dataset_root.
    rewritten = json.loads((dataset_root / "problems" / "foo.json").read_text())
    assert rewritten["target_pdb_filepath"] == str((dataset_root / "targets" / "pdb" / "foo.pdb").resolve())


def test_extract_dataset_with_top_level_layout(tmp_path: Path) -> None:
    """The other valid layout is problems/ + targets/ directly at the zip root."""
    problem = {"target_pdb_filepath": "foo.pdb"}
    _make_zip(
        tmp_path / "ds.zip",
        {
            "problems/foo.json": json.dumps(problem),
            "targets/pdb/foo.pdb": "ATOM",
        },
    )
    dataset_root = extract_dataset(tmp_path / "ds.zip", tmp_path / "out")
    rewritten = json.loads((dataset_root / "problems" / "foo.json").read_text())
    # Resolved by basename fallback into the canonical sub-dir.
    assert rewritten["target_pdb_filepath"].endswith("targets/pdb/foo.pdb")


def test_extract_dataset_rewrites_motif_filepaths(tmp_path: Path) -> None:
    problem = {"motif_filepaths": ["motifs/m1.pdb", "motifs/m2.pdb"]}
    _make_zip(
        tmp_path / "ds.zip",
        {
            "problems/p.json": json.dumps(problem),
            "motifs/m1.pdb": "ATOM 1",
            "motifs/m2.pdb": "ATOM 2",
        },
    )
    dataset_root = extract_dataset(tmp_path / "ds.zip", tmp_path / "out")
    rewritten = json.loads((dataset_root / "problems" / "p.json").read_text())
    assert all(Path(p).is_absolute() and Path(p).exists() for p in rewritten["motif_filepaths"])


def test_extract_dataset_follows_binder_framework_chain(tmp_path: Path) -> None:
    """`binder_framework` points at a nested motif config whose paths also need rewriting."""
    nested_config = {"motif_filepaths": ["motifs/m.pdb"]}
    problem = {"binder_framework": "framework_config.json"}
    _make_zip(
        tmp_path / "ds.zip",
        {
            "problems/p.json": json.dumps(problem),
            "framework_config.json": json.dumps(nested_config),
            "motifs/m.pdb": "ATOM",
        },
    )
    dataset_root = extract_dataset(tmp_path / "ds.zip", tmp_path / "out")

    # Problem JSON's binder_framework now points at the extracted file (absolute path).
    p = json.loads((dataset_root / "problems" / "p.json").read_text())
    assert Path(p["binder_framework"]).is_absolute()

    # And the nested motif config got its own filepath rewritten.
    nested = json.loads(Path(p["binder_framework"]).read_text())
    assert Path(nested["motif_filepaths"][0]).is_absolute()
    assert Path(nested["motif_filepaths"][0]).exists()


def test_extract_dataset_rejects_zip_without_problems_dir(tmp_path: Path) -> None:
    _make_zip(tmp_path / "ds.zip", {"random/file.txt": "x"})
    with pytest.raises(ValueError, match="problems/"):
        extract_dataset(tmp_path / "ds.zip", tmp_path / "out")
