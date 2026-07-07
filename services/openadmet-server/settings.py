"""Env-driven config for openadmet-server.

All values via pydantic-settings; env_prefix=`OPENADMET_`.

Design doc: engineering/decisions/2026-07-05-openadmet-server-design.md
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from bioagent_service import ServiceSettings
from pydantic import Field, computed_field
from pydantic_settings import SettingsConfigDict


class ModelInfo:
    """Lightweight NAS-registered model descriptor (populated by `list_models`)."""

    __slots__ = (
        "name", "path", "input_col", "target_cols", "biotargets",
        "tag", "description", "model_type", "feat_type", "build_number",
    )

    def __init__(
        self,
        name: str,
        path: Path,
        input_col: str,
        target_cols: list[str],
        biotargets: list[str],
        tag: str,
        description: str,
        model_type: str,
        feat_type: str,
        build_number: int,
    ) -> None:
        self.name = name
        self.path = path
        self.input_col = input_col
        self.target_cols = target_cols
        self.biotargets = biotargets
        self.tag = tag
        self.description = description
        self.model_type = model_type
        self.feat_type = feat_type
        self.build_number = build_number

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "input_col": self.input_col,
            "target_cols": self.target_cols,
            "biotargets": self.biotargets,
            "tag": self.tag,
            "description": self.description,
            "model_type": self.model_type,
            "feat_type": self.feat_type,
            "build_number": self.build_number,
        }


class OpenAdmetSettings(ServiceSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENADMET_",
        env_file=".env",
        extra="ignore",
    )

    jobs_base_dir: Path = Field(default=Path("/data/openadmet_jobs"))

    root: Path = Field(
        default=Path("/opt/openadmet/upstream"),
        description="Upstream OpenADMET-models source root (subprocess cwd).",
    )

    python: str = Field(
        default="/opt/conda/envs/openadmet-models/bin/python",
        description="Python interpreter inside the conda env.",
    )

    # Setuptools entry-point script created by `pip install -e ./upstream`.
    # Upstream defines `openadmet = "openadmet.models.cli.cli:cli"` in
    # pyproject.toml → this exact path is generated.
    #
    # DO NOT switch to `python -m openadmet.models.cli.cli` — that module
    # has no `if __name__ == '__main__': cli()` guard, so it just imports
    # (30 s of chemprop/molfeat/torch import) and exits rc=0 without ever
    # invoking the click group.  Discovered 2026-07-07 during v0.0.4
    # diagnostic — subprocess ran 36 s, log empty, output_dir empty.
    cli_binary: str = Field(
        default="/opt/conda/envs/openadmet-models/bin/openadmet",
        description="OpenADMET CLI entry-point script (from setuptools).",
    )

    # Weight root — see §6.7 of the design doc.  Layout:
    #   weights_dir/models/<name>/{model.pth,model.json,recipe_components/,...}
    #   weights_dir/foundations/.chemprop/chemeleon_mp.pt
    weights_dir: Path = Field(
        default=Path("/data/models/openadmet"),
        description="Root for pre-trained anvil model dirs + CheMeleon foundation cache.",
    )

    # Alias columns written into the temp CSV when a client submits inline
    # SMILES (see §4.1 input_col auto-derive rules).  All aliases receive
    # the same values so any registered model's input_col will match.
    default_input_col_aliases: list[str] = Field(
        default_factory=lambda: [
            "OPENADMET_SMILES",
            "OPENADMET_CANONICAL_SMILES",
            "SMILES",
            "canonical_smiles",
            "smiles",
        ],
    )

    # GPU single-card; FC session affinity handles per-instance concurrency.
    max_concurrent_jobs: int = Field(default=1, ge=1, le=8)

    oss_region: str = Field(default="cn-hangzhou")

    session_header_name: str = Field(default="bioagent-session-id")

    # ---- Computed weight-dir paths ----

    @computed_field  # type: ignore[misc]
    @property
    def models_root(self) -> Path:
        """Where individual pre-trained model_dirs live."""
        return self.weights_dir / "models"

    @computed_field  # type: ignore[misc]
    @property
    def chemeleon_foundation(self) -> Path:
        """CheMeleon foundation checkpoint.

        Upstream ``ChemPropModel._download_chemeleon`` expects
        ``Path.home() / ".chemprop" / "chemeleon_mp.pt"`` — Dockerfile sets
        ``HOME=<weights_dir>/foundations`` so this path resolves right.
        """
        return self.weights_dir / "foundations" / ".chemprop" / "chemeleon_mp.pt"

    # ---- Model registry helpers ----

    def model_path(self, name: str) -> Path:
        """Return the resolved model_dir path (with ``anvil_training/`` suffix if nested).

        Delegates to the registry so both flat and nested NAS layouts are
        transparently supported — see ``_read_model_info``.
        """
        for info in self.list_models():
            if info.name == name:
                return info.path
        raise ValueError(
            f"Model '{name}' not registered under {self.models_root}. "
            f"Available: {[m.name for m in self.list_models()]}"
        )

    def list_models(self) -> list[ModelInfo]:
        """Scan `models_root` for anvil-trained model_dirs and return metadata.

        Reads ``recipe_components/{metadata,data,procedure}.yaml`` from each.
        Silently skips directories missing required yaml files (partial rsyncs).
        Cached by mtime of the models_root (§6.5 of design doc).
        """
        if not self.models_root.is_dir():
            return []
        # LRU-cached at module level, keyed by (models_root, mtime).
        return _cached_list_models(
            self.models_root,
            _dir_mtime_ns(self.models_root),
        )


def _dir_mtime_ns(path: Path) -> int:
    """Coarse cache key: newest mtime among direct children + own mtime."""
    try:
        mtimes = [path.stat().st_mtime_ns]
        for child in path.iterdir():
            try:
                mtimes.append(child.stat().st_mtime_ns)
            except FileNotFoundError:
                continue
        return max(mtimes)
    except FileNotFoundError:
        return 0


@lru_cache(maxsize=1)
def _cached_list_models(models_root: Path, _mtime_key: int) -> list[ModelInfo]:
    infos: list[ModelInfo] = []
    for child in sorted(models_root.iterdir()):
        if not child.is_dir():
            continue
        info = _read_model_info(child)
        if info is not None:
            infos.append(info)
    return infos


def _read_model_info(model_dir: Path) -> Optional[ModelInfo]:
    """Recognize an anvil model directory in either of two layouts.

    * **Flat**:   ``<model_dir>/recipe_components/{metadata,data,procedure}.yaml``
                  + ``<model_dir>/model.pth``
    * **Nested**: ``<model_dir>/anvil_training/recipe_components/...``
                  + ``<model_dir>/anvil_training/model.pth``
                  (this is HuggingFace's default download layout for
                  openadmet/* model repos)

    The nested layout is common if models were rsync'd straight from
    ``opensource/openadmet-models/hc/`` without stripping ``anvil_training/``.
    The returned ``ModelInfo.path`` always points at the directory that
    contains ``model.pth`` (i.e. the anvil_training subdir if nested), so
    downstream ``openadmet predict --model-dir`` gets the right path.
    """
    candidates = [model_dir, model_dir / "anvil_training"]
    for candidate in candidates:
        rc = candidate / "recipe_components"
        meta_p = rc / "metadata.yaml"
        data_p = rc / "data.yaml"
        proc_p = rc / "procedure.yaml"
        if not (meta_p.is_file() and data_p.is_file() and proc_p.is_file()):
            continue

        try:
            meta = yaml.safe_load(meta_p.read_text()) or {}
            data = yaml.safe_load(data_p.read_text()) or {}
            proc = yaml.safe_load(proc_p.read_text()) or {}
        except yaml.YAMLError:
            continue

        target_cols = data.get("target_cols") or []
        if isinstance(target_cols, str):
            target_cols = [target_cols]

        return ModelInfo(
            name=model_dir.name,     # outer dir name — the registry key
            path=candidate,          # actual model.pth dir (with anvil_training suffix if nested)
            input_col=data.get("input_col") or "OPENADMET_CANONICAL_SMILES",
            target_cols=list(target_cols),
            biotargets=list(meta.get("biotargets") or []),
            tag=meta.get("tag") or "",
            description=meta.get("description") or "",
            model_type=(proc.get("model") or {}).get("type") or "",
            feat_type=(proc.get("feat") or {}).get("type") or "",
            build_number=int(meta.get("build_number") or 0),
        )
    return None
