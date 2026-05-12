"""Config builders must produce the same YAML shape that legacy `tasks.py` did."""

from __future__ import annotations

from pathlib import Path

from server.configs import (
    build_binder_config,
    build_motif_config,
    build_unconditional_config,
    rewrite_custom_paths,
)
from server.models import BinderRequest, MotifRequest, UnconditionalRequest


def test_unconditional_config(tmp_path: Path) -> None:
    cfg = build_unconditional_config(
        rootdir=tmp_path / "out",
        req=UnconditionalRequest(min_length=50, max_length=200, n_sample=8, direction_scale=0.5),
    )
    assert cfg["experiment"]["name"] == "unconditional"
    assert cfg["generation"]["dataset"]["source"] == "unconditional"
    assert cfg["generation"]["dataset"]["min_length"] == 50
    assert cfg["generation"]["dataset"]["max_length"] == 200
    assert cfg["generation"]["dataset"]["n_sample"] == 8
    assert cfg["generation"]["sampler"]["sampler"]["direction_scale"] == 0.5
    assert cfg["paths"]["rootdir"].endswith("out")


def test_motif_config_includes_selections(tmp_path: Path) -> None:
    cfg = build_motif_config(
        rootdir=tmp_path / "out",
        dataset_root=tmp_path / "dataset",
        req=MotifRequest(selections="prob1,prob2", n_sample=16),
    )
    assert cfg["experiment"]["name"] == "motif"
    assert cfg["generation"]["dataset"]["source"] == "motif"
    assert cfg["generation"]["dataset"]["selections"] == "prob1,prob2"
    assert cfg["paths"]["dataset"].endswith("dataset")


def test_motif_config_omits_selections_when_unset(tmp_path: Path) -> None:
    cfg = build_motif_config(
        rootdir=tmp_path / "out",
        dataset_root=tmp_path / "dataset",
        req=MotifRequest(),
    )
    assert "selections" not in cfg["generation"]["dataset"]


def test_binder_config_uses_target_source(tmp_path: Path) -> None:
    cfg = build_binder_config(
        rootdir=tmp_path / "out",
        dataset_root=tmp_path / "dataset",
        req=BinderRequest(n_sample=64, batch_size=4, direction_scale=0.0),
    )
    assert cfg["experiment"]["name"] == "binder"
    assert cfg["generation"]["dataset"]["source"] == "target"
    assert cfg["generation"]["dataset"]["batch_size"] == 4
    assert cfg["generation"]["sampler"]["sampler"]["direction_scale"] == 0.0


def test_rewrite_custom_paths_injects_rootdir(tmp_path: Path) -> None:
    user = {"experiment": {"name": "x"}, "generation": {"dataset": {"source": "unconditional"}}}
    out = rewrite_custom_paths(user, rootdir=tmp_path / "out")
    assert out["paths"]["rootdir"].endswith("out")
    assert "dataset" not in out["paths"]


def test_rewrite_custom_paths_injects_both_when_dataset_given(tmp_path: Path) -> None:
    user: dict = {"experiment": {"name": "x"}}
    out = rewrite_custom_paths(
        user, rootdir=tmp_path / "out", dataset_root=tmp_path / "ds",
    )
    assert out["paths"]["dataset"].endswith("ds")
